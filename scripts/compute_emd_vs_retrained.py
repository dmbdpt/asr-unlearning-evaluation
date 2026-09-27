#!/usr/bin/env python
"""Earth Mover's Distance between each unlearned checkpoint's forget-set loss distribution and..."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, Subset

from mia_common import (
    DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, MODEL_TAG, discover_checkpoints, select_gpu,
)
from mia_common import find_model_spec, load_attack_model

torch.multiprocessing.set_sharing_strategy('file_system')

from src.models.espnet import ESPnetASRWrapper

from src.data.datasets.csv_dataset import CSVDataset
from src.data.datasets.librispeech import collate_fn

DEFAULT_RETRAIN_ROOT = Path.home() / "unlearned_ckps_retrain" / "retrained"


def forget_subset(train_raw: CSVDataset, subject: str) -> Subset:
    idx = train_raw.get_subj_indices(int(subject))
    if not idx:
        raise ValueError(f"No training utterances found for speaker {subject!r}")
    return Subset(train_raw, idx)


def sample_subset(dataset: CSVDataset, n: int, seed: int) -> Subset:
    indices = np.arange(len(dataset))
    if n and len(indices) > n:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=n, replace=False))
    return Subset(dataset, indices.tolist())


def collect_per_utt_losses(model, dataset, device, num_workers=8) -> np.ndarray:
    """Per-utterance total loss (ESPnetASRWrapper.forward's att+ctc combination)."""
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                        num_workers=num_workers, shuffle=False)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
            loss = model(batch_on_device)
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu()
            if loss.ndim == 0:
                loss = loss.unsqueeze(0)
            losses.extend(loss.numpy().tolist())
    return np.array(losses, dtype=float)


def is_done(path: Path) -> bool:
    """Only a result that finished ok counts; a stored failure is retried."""
    try:
        return json.loads(path.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def write_summary(out_dir: Path):
    rows = []
    for path in sorted(out_dir.glob("*/*.json")):
        if path.name in ("summary.json",) or path.parent.name == "_retrained":
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(payload)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["method", "subject"])
    summary_csv = out_dir / "forget_emd_vs_retrained_summary.csv"
    df.to_csv(summary_csv, index=False)
    return summary_csv, len(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-root", default=str(Path.home() / "unlearned_ckps"))
    p.add_argument("--retrain-root", default=str(DEFAULT_RETRAIN_ROOT))
    p.add_argument("-o", "--out-dir",
                   default=str(REPO_ROOT / "results" / "forget_emd_vs_retrained"))
    p.add_argument("--methods", nargs="+", help="only these method folders")
    p.add_argument("--method", dest="methods", nargs="+", help="alias for --methods")
    p.add_argument("--subjects", nargs="+", help="only these forget subjects")
    p.add_argument("--device", help="torch device (default: CUDA card with most free memory)")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--n-test", type=int, default=1000,
                   help="test utterances to score for the retrained model's own "
                        "forget-vs-test EMD (0 = all; same sample for every subject)")
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="rerun checkpoints that already have an ok result")
    p.add_argument("-n", "--dry-run", action="store_true", help="list the work and exit")
    return p.parse_args(argv)


def main(argv=None):
    # from_pretrained stats the bare model tag as a relative path, so a cwd with an...
    os.chdir(REPO_ROOT)
    args = parse_args(argv)
    root = Path(args.checkpoints_root).expanduser()
    retrain_root = Path(args.retrain_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Checkpoint root not found: {root}")
    if not retrain_root.is_dir():
        raise SystemExit(f"Retrained model root not found: {retrain_root}")
    if not Path(args.train_csv).is_file():
        raise SystemExit(f"Train CSV not found: {args.train_csv}")
    if not Path(args.test_csv).is_file():
        raise SystemExit(f"Test CSV not found: {args.test_csv}")

    work = discover_checkpoints(root, set(args.methods or []), set(args.subjects or []))
    if not work:
        raise SystemExit(f"No <method>/<subject>/{{last.ckpt,espnet_model.json}} under {root} "
                         f"for those filters.")

    pending = [(m, s, c) for m, s, c in work
              if args.overwrite or not is_done(out_dir / m / f"{s}.json")]
    print(f"{len(work)} checkpoint(s); {len(pending)} to run, "
          f"{len(work) - len(pending)} already done")
    if args.dry_run:
        for m, s, c in pending:
            print(f"  {m}/{s}  forget={s}  <- {c}  vs retrained/{s}")
        return 0
    if not pending:
        summary, n = write_summary(out_dir)
        print(f"Nothing to do. Summary over {n} row(s): {summary}")
        return 0

    gpu_index, device = select_gpu(args.device)
    print(f"Device: {device} (physical GPU {gpu_index})" if gpu_index is not None
          else f"Device: {device}")
    print(f"Train CSV: {args.train_csv}")

    train_raw = CSVDataset(args.train_csv)
    test_raw = CSVDataset(args.test_csv)
    test_set = sample_subset(test_raw, args.n_test, args.seed)
    print(f"Train pool: {len(train_raw)} utterances; test pool: {len(test_raw)} "
          f"(scoring {len(test_set)} against the retrained model)")

    out_dir.mkdir(parents=True, exist_ok=True)
    retrained_dir = out_dir / "_retrained"
    retrained_dir.mkdir(parents=True, exist_ok=True)
    retrained_cache: dict[str, np.ndarray] = {}
    retrained_test_losses = None
    retrained_forget_vs_test: dict[str, float] = {}
    retrained_forget_vs_pretrained: dict[str, float] = {}
    pretrained_model = None

    started, failures = time.time(), []
    for n_done, (method, subject, ckpt) in enumerate(pending, start=1):
        print(f"\n[{n_done}/{len(pending)}] {method}/{subject}  (forget speaker {subject})")
        t0 = time.time()
        payload = {"method": method, "subject": subject, "checkpoint": str(ckpt),
                   "status": "failed"}
        model = None
        try:
            fset = forget_subset(train_raw, subject)
            print(f"  forget set: {len(fset)} utterance(s)")

            if subject not in retrained_cache:
                retrain_spec = find_model_spec(retrain_root / subject)
                if retrain_spec is None:
                    raise FileNotFoundError(
                        f"No retrained model spec under {retrain_root / subject}")
                retrain_model = load_attack_model(retrain_spec, args.model_tag, device)
                retrained_cache[subject] = collect_per_utt_losses(
                    retrain_model, fset, device, args.num_workers)
                rl = retrained_cache[subject]
                print(f"  retrained forget losses: n={len(rl)} "
                      f"mean={rl.mean():.4f} std={rl.std():.4f}")

                # Test losses only depend on the retrained WEIGHTS, which are byte-identical across every...
                if retrained_test_losses is None:
                    retrained_test_losses = collect_per_utt_losses(
                        retrain_model, test_set, device, args.num_workers)
                    tl = retrained_test_losses
                    print(f"  retrained test losses:   n={len(tl)} "
                          f"mean={tl.mean():.4f} std={tl.std():.4f}")

                # Pretrained (pre-unlearning) forget losses on the SAME utterances -- the starting point...
                if pretrained_model is None:
                    pretrained_model = ESPnetASRWrapper.from_pretrained(args.model_tag)
                    pretrained_model.to(device)
                    pretrained_model.eval()
                pretrained_forget_losses = collect_per_utt_losses(
                    pretrained_model, fset, device, args.num_workers)
                pl = pretrained_forget_losses
                print(f"  pretrained forget losses: n={len(pl)} "
                      f"mean={pl.mean():.4f} std={pl.std():.4f}")

                extra = {}
                if len(rl) >= 2 and len(retrained_test_losses) >= 2:
                    ft_emd = float(wasserstein_distance(rl, retrained_test_losses))
                    retrained_forget_vs_test[subject] = ft_emd
                    print(f"  EMD(forget[retrained], test[retrained]) = {ft_emd:.4f}")
                    extra.update({
                        "n_test_utts": int(len(retrained_test_losses)),
                        "retrained_test_loss_mean": float(retrained_test_losses.mean()),
                        "retrained_test_loss_std": float(retrained_test_losses.std()),
                        "emd_retrained_forget_vs_test": ft_emd,
                    })
                if len(rl) >= 2 and len(pl) >= 2:
                    fp_emd = float(wasserstein_distance(rl, pl))
                    retrained_forget_vs_pretrained[subject] = fp_emd
                    print(f"  EMD(forget[retrained], forget[pretrained]) = {fp_emd:.4f}")
                    extra.update({
                        "pretrained_forget_loss_mean": float(pl.mean()),
                        "pretrained_forget_loss_std": float(pl.std()),
                        "emd_retrained_forget_vs_pretrained_forget": fp_emd,
                    })
                if extra:
                    (retrained_dir / f"{subject}.json").write_text(json.dumps({
                        "subject": subject,
                        "n_forget_utts": int(len(rl)),
                        "retrained_forget_loss_mean": float(rl.mean()),
                        "retrained_forget_loss_std": float(rl.std()),
                        **extra,
                    }, indent=2))

            model = load_attack_model(ckpt, args.model_tag, device)
            unlearned_losses = collect_per_utt_losses(model, fset, device, args.num_workers)
            print(f"  {method:<12} forget losses: n={len(unlearned_losses)} "
                  f"mean={unlearned_losses.mean():.4f} std={unlearned_losses.std():.4f}")

            retrained_losses = retrained_cache[subject]
            if len(unlearned_losses) < 2 or len(retrained_losses) < 2:
                raise ValueError(
                    f"Not enough samples to compute EMD (unlearned={len(unlearned_losses)}, "
                    f"retrained={len(retrained_losses)})")
            emd = float(wasserstein_distance(unlearned_losses, retrained_losses))
            print(f"  EMD(forget[{method}], forget[retrained]) = {emd:.4f}")

            payload.update({
                "status": "ok",
                "n_forget_utts": int(len(unlearned_losses)),
                "unlearned_loss_mean": float(unlearned_losses.mean()),
                "unlearned_loss_std": float(unlearned_losses.std()),
                "retrained_loss_mean": float(retrained_losses.mean()),
                "retrained_loss_std": float(retrained_losses.std()),
                "emd_forget_vs_retrained": emd,
                "retrained_forget_vs_test_emd": retrained_forget_vs_test.get(subject),
                "retrained_forget_vs_pretrained_emd": retrained_forget_vs_pretrained.get(subject),
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
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / f"{subject}.json").write_text(json.dumps(payload, indent=2))
        print(f"    ({payload['runtime_sec']}s)" if payload["status"] == "ok"
              else f"    FAILED after {payload['runtime_sec']}s")

    summary, n_rows = write_summary(out_dir)
    print(f"\nRan {len(pending)} checkpoint(s) in {(time.time() - started) / 60:.1f} min")
    if failures:
        print(f"{len(failures)} failed:")
        for method, subject, why in failures:
            print(f"  {method}/{subject}: {why}")
    print(f"Summary over {n_rows} row(s): {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
