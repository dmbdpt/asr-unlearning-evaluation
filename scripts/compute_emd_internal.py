#!/usr/bin/env python
"""Forget-vs-test and forget-vs-retain EMD around each unlearned checkpoint, for every pre/post..."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import Subset

from compute_emd_vs_retrained import collect_per_utt_losses, forget_subset
from mia_common import (DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, FORGET_SET, MODEL_TAG,
                        discover_checkpoints, select_gpu)
from mia_common import load_attack_model

from src.models.espnet import ESPnetASRWrapper

from src.data.datasets.csv_dataset import CSVDataset


def sample_subset(dataset: CSVDataset, indices, n, seed):
    indices = np.asarray(indices)
    if n and len(indices) > n:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=n, replace=False))
    return Subset(dataset, indices.tolist())


def retain_indices(train_raw: CSVDataset, subject: str, exclude_other_forgotten: bool):
    spk = train_raw.data["speaker_id"].astype(str)
    mask = spk != subject
    if exclude_other_forgotten:
        mask &= ~spk.isin(FORGET_SET)
    return np.flatnonzero(mask.to_numpy())


PAIRS = (  # name, (numerator set, denominator set) as (phase, split) keys
    ("emd_forgetpre_testpre", ("pre_forget", "pre_test")),
    ("emd_forgetpre_retainpre", ("pre_forget", "pre_retain")),
    ("emd_forgetpost_testpre", ("post_forget", "pre_test")),
    ("emd_forgetpost_retainpre", ("post_forget", "pre_retain")),
    ("emd_forgetpost_testpost", ("post_forget", "post_test")),
    ("emd_forgetpost_retainpost", ("post_forget", "post_retain")),
)


def stats(x):
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "median": float(np.median(x))}


def is_done(path: Path) -> bool:
    """Only a result that finished ok counts; a stored failure is retried."""
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def write_summary(out_dir: Path):
    rows = []
    for path in sorted(out_dir.glob("*/*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["method", "subject"])
    out = out_dir / "forget_emd_internal_summary.csv"
    df.to_csv(out, index=False)
    return out, len(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-root", default=str(Path.home() / "unlearned_ckps"))
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results" / "forget_emd_internal"))
    p.add_argument("--methods", "--method", dest="methods", nargs="+")
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--device")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--n-test", type=int, default=1000, help="test utterances to score (0 = all)")
    p.add_argument("--n-retain", type=int, default=1000, help="retain utterances to score (0 = all)")
    p.add_argument("--exclude-other-forgotten", action="store_true",
                   help="also drop the other nine forgotten speakers from retain (automatic for *_multi methods)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-n", "--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    # from_pretrained stats the bare model tag as a relative path, so a cwd with an...
    os.chdir(REPO_ROOT)
    args = parse_args(argv)
    root, out_dir = Path(args.checkpoints_root).expanduser(), Path(args.out_dir).expanduser()
    for f in (args.train_csv, args.test_csv):
        if not Path(f).is_file():
            raise SystemExit(f"CSV not found: {f}")
    if not root.is_dir():
        raise SystemExit(f"Checkpoint root not found: {root}")

    work = discover_checkpoints(root, set(args.methods or []), set(args.subjects or []))
    if not work:
        raise SystemExit(f"No checkpoints under {root} for those filters.")
    pending = [(m, s, c) for m, s, c in work
               if args.overwrite or not is_done(out_dir / m / f"{s}.json")]
    print(f"{len(work)} checkpoint(s); {len(pending)} to run, {len(work) - len(pending)} already done")
    if args.dry_run:
        for m, s, c in pending:
            print(f"  {m}/{s}  <- {c}")
        return 0
    if not pending:
        print(f"Nothing to do. Summary: {write_summary(out_dir)[0]}")
        return 0

    gpu_index, device = select_gpu(args.device)
    print(f"Device: {device}" + (f" (physical GPU {gpu_index})" if gpu_index is not None else ""))

    train_raw, test_raw = CSVDataset(args.train_csv), CSVDataset(args.test_csv)
    test_set = sample_subset(test_raw, np.arange(len(test_raw)), args.n_test, args.seed)
    print(f"Train pool {len(train_raw)}, test pool {len(test_raw)} (scoring {len(test_set)})")

    pre_model = None
    pre_dir = out_dir / "_pre"
    pre_dir.mkdir(parents=True, exist_ok=True)

    started, failures = time.time(), []
    for i, (method, subject, ckpt) in enumerate(pending, start=1):
        print(f"\n[{i}/{len(pending)}] {method}/{subject}")
        t0 = time.time()
        payload = {"method": method, "subject": subject, "checkpoint": str(ckpt), "status": "failed"}
        model = None
        try:
            excl = args.exclude_other_forgotten or "multi" in method
            fset = forget_subset(train_raw, subject)
            rset = sample_subset(train_raw, retain_indices(train_raw, subject, excl),
                                 args.n_retain, args.seed)
            print(f"  forget={len(fset)}  retain={len(rset)} (other forgotten excluded: {excl})  test={len(test_set)}")

            sets = {"forget": fset, "retain": rset, "test": test_set}
            loss = {}

            pre_cache = pre_dir / f"{subject}_excl{int(excl)}_n{len(rset)}_{len(test_set)}_s{args.seed}.npz"
            if pre_cache.is_file():
                with np.load(pre_cache) as z:
                    loss.update({f"pre_{k}": z[k] for k in sets})
                print(f"  pre losses: loaded {pre_cache.name}")
            else:
                if pre_model is None:
                    pre_model = ESPnetASRWrapper.from_pretrained(args.model_tag)
                    pre_model.to(device)
                    pre_model.eval()
                pre = {k: collect_per_utt_losses(pre_model, ds, device, args.num_workers)
                       for k, ds in sets.items()}
                np.savez(pre_cache, **pre)
                loss.update({f"pre_{k}": v for k, v in pre.items()})

            model = load_attack_model(ckpt, args.model_tag, device)
            loss.update({f"post_{k}": collect_per_utt_losses(model, ds, device, args.num_workers)
                         for k, ds in sets.items()})
            for name, x in loss.items():
                print(f"  {name:<12} mean={x.mean():.4f} std={x.std():.4f} n={len(x)}")
            if min(len(x) for x in loss.values()) < 2:
                raise ValueError("Not enough samples to compute EMD")

            emds = {name: float(wasserstein_distance(loss[a], loss[b])) for name, (a, b) in PAIRS}
            for name, v in emds.items():
                print(f"  {name:<28} {v:.4f}")

            method_dir = out_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            np.savez(method_dir / f"{subject}_losses.npz", **loss)
            payload.update({
                "status": "ok", "other_forgotten_excluded": bool(excl), **emds,
                **{f"{k}_{s}": v for k, x in loss.items() for s, v in stats(x).items()},
            })
        except Exception as exc:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            payload["error"] = repr(exc)
            failures.append((method, subject, repr(exc)))
        finally:
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        payload["runtime_sec"] = round(time.time() - t0, 1)
        (out_dir / method).mkdir(parents=True, exist_ok=True)
        (out_dir / method / f"{subject}.json").write_text(json.dumps(payload, indent=2))
        print(f"    ({payload['runtime_sec']}s)")

    summary, n_rows = write_summary(out_dir)
    print(f"\nRan {len(pending)} checkpoint(s) in {(time.time() - started) / 60:.1f} min")
    for m, s, why in failures:
        print(f"  FAILED {m}/{s}: {why}")
    print(f"Summary over {n_rows} row(s): {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
