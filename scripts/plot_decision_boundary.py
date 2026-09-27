#!/usr/bin/env python
"""Plot the loss-feature MIA classifier's decision boundary for every organized unlearning..."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from mia_common import (
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    MODEL_TAG,
    align_mia_sets_to_forget,
    build_train_eval_csvs,
    discover_checkpoints,
    load_unlearned_model,
    select_gpu,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("espnet", "espnet2", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("plot_decision_boundary")

torch.multiprocessing.set_sharing_strategy('file_system')

FEATURE_NAMES = ["loss_att", "loss_ctc"]


# ---------------------------------------------------------------------------
# Plotting — ported from the decision-boundary cell in mia_loo.ipynb
# ---------------------------------------------------------------------------

def plot_decision_boundary(X, y, classifier, feat_names, title, save_path,
                           grid_res, pad):
    x0_min, x0_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    x1_min, x1_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, grid_res),
                         np.linspace(x1_min, x1_max, grid_res))

    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.75, vmin=0, vmax=1)
    fig.colorbar(cf, ax=ax, label="P(member)")
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)

    colors = {0: "tab:blue", 1: "tab:orange"}
    labels_map = {0: "Non-member", 1: "Member"}
    y = np.asarray(y)
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls], label=labels_map[cls],
                  s=18, alpha=0.7, edgecolors="none")

    ax.set_xlabel(feat_names[0] if len(feat_names) > 0 else "feature 0")
    ax.set_ylabel(feat_names[1] if len(feat_names) > 1 else "feature 1")
    ax.set_title(title)
    ax.legend(markerscale=1.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-root", default=str(Path.home() / "unlearned_ckps"))
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results" / "decision_boundary"))
    p.add_argument("--methods", nargs="+", help="only these method folders")
    p.add_argument("--subjects", nargs="+", help="only these forget subjects")
    p.add_argument("--device", help="torch device (default: CUDA card with most free memory)")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--class-balance", choices=("none", "downsample", "class-weight"),
                   default="class-weight",
                   help="how to handle the member/non-member imbalance the duration "
                        "matching leaves in the TRAINING set (default: class-weight, "
                        "matching run_mia.py's default)")
    p.add_argument("--no-match-durations", dest="match_durations", action="store_false",
                   help="skip the duration alignment entirely (for an ablation)")
    p.add_argument("--duration-bins", type=int, default=20,
                   help="quantile bins for the duration histogram (default: 20)")
    p.add_argument("--nonmember-split", choices=("speaker", "utterance"), default="speaker",
                   help="how the test pool is halved into the attacker's training "
                        "negatives and its evaluation negatives (default: speaker)")
    p.add_argument("--no-per-corpus-match", dest="per_corpus_match", action="store_false",
                   help="duration-match the pooled non-member set instead of matching "
                        "test-clean and test-other to the forget speaker separately")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--mia-train-split-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-resolution", type=int, default=400,
                   help="side length of the P(member) contour grid (default: 400)")
    p.add_argument("--pad", type=float, default=0.5,
                   help="axis padding beyond the eval features' min/max (default: 0.5)")
    p.add_argument("--reuse-features", action="store_true",
                   help="reuse a previously written feature cache (sets override=False)")
    p.add_argument("--overwrite", action="store_true",
                   help="rerun checkpoints that already have a decision_boundary.png")
    p.add_argument("-n", "--dry-run", action="store_true", help="list the work and exit")
    return p.parse_args(argv)


def build_cfg(args):
    return OmegaConf.create({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "override": not args.reuse_features,
        "cache_save_frequency": 1,
        "mia_train_split_ratio": args.mia_train_split_ratio,
        "loss_att": True,
        "loss_ctc": True,
        "loss_cer": False,               # decision boundary is drawn in 2D only
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "cache_dir": None,
        "random_state": args.seed,
    })


def write_summary(out_dir):
    rows = []
    for path in sorted(out_dir.glob("*/*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        row = {k: payload.get(k) for k in
              ("method", "forget_subject", "checkpoint", "status", "runtime_sec", "png")}
        row.update(payload.get("metrics") or {})
        rows.append(row)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["method", "forget_subject"])
    summary = out_dir / "decision_boundary_summary.csv"
    df.to_csv(summary, index=False)
    return summary, len(rows)


def main(argv=None):
    args = parse_args(argv)
    root, out_dir = Path(args.checkpoints_root).expanduser(), Path(args.out_dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Checkpoint root not found: {root}")
    for csv in (args.train_csv, args.test_csv):
        if not Path(csv).is_file():
            raise SystemExit(f"CSV not found: {csv}")

    work = discover_checkpoints(root, set(args.methods or []), set(args.subjects or []))
    if not work:
        raise SystemExit(f"No <method>/<subject>/last.ckpt under {root} for those filters.")

    pending = [(m, s, c) for m, s, c in work
              if args.overwrite or not (out_dir / m / s / "decision_boundary.png").exists()]
    print(f"{len(work)} checkpoint(s); {len(pending)} to run, "
          f"{len(work) - len(pending)} already done")
    print(f"Duration matching: {'on' if args.match_durations else 'OFF'}"
          f"{' (per corpus)' if args.per_corpus_match else ' (pooled)'} | "
          f"class balance: {args.class_balance} | non-member split: by {args.nonmember_split}")
    if args.dry_run:
        for m, s, c in pending:
            print(f"  {m}/{s}  forget={s}  <- {c}")
        return 0
    if not pending:
        summary, n = write_summary(out_dir)
        print(f"Nothing to do. Summary over {n} row(s): {summary}")
        return 0

    gpu_index, device = select_gpu(args.device)

    from src.data.datasets.csv_dataset import CSVDataset
    from src.evaluation.evaluate_mia import _build_classifier, _build_feature_extractor, _prepare_forget_eval_csvs
    from src.membership_inference.modules.abs_attacker import Attacker
    from src.membership_inference.utils import compute_binary_mia_metrics

    cfg = build_cfg(args)
    print(f"Device: {device} (physical GPU {gpu_index})" if gpu_index is not None
          else f"Device: {device}")
    print(f"Train CSV: {args.train_csv}\nTest CSV:  {args.test_csv}")

    train_raw = CSVDataset(args.train_csv)
    test_raw = CSVDataset(args.test_csv)
    print(f"Pools: {len(train_raw)} train utterances, {len(test_raw)} test utterances")

    started, failures = time.time(), []
    for n_done, (method, subject, ckpt) in enumerate(pending, start=1):
        print(f"\n[{n_done}/{len(pending)}] {method}/{subject}  (forget speaker {subject})")
        artifact_dir = out_dir / method / subject
        temp_dir = artifact_dir / "mia_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        png_path = artifact_dir / "decision_boundary.png"
        t0 = time.time()

        model, metrics, counts = None, {}, {}
        try:
            csv_sets = _prepare_forget_eval_csvs(
                datasets={"train_raw": train_raw, "test": test_raw},
                temp_dir=str(temp_dir),
                forget_speakers=[subject],
                seed=args.seed,
                train_split_ratio=cfg.mia_train_split_ratio,
                nonmember_split_by=args.nonmember_split,
            )
            aligned = (align_mia_sets_to_forget(csv_sets, temp_dir, args.duration_bins,
                                                args.seed, args.per_corpus_match)
                       if args.match_durations else csv_sets)
            train_csv, eval_csv, counts = build_train_eval_csvs(
                aligned, temp_dir, args.class_balance, args.seed)
            print(f"  train: {counts['n_train_members']} utt (members)  vs  "
                  f"{counts['n_train_nonmembers']} utt (non-members)")
            print(f"  eval:  {counts['n_eval_members']} utt (members)  vs  "
                  f"{counts['n_eval_nonmembers']} utt (non-members)")
            if min(v for k, v in counts.items() if k != "subject_ids") == 0:
                raise ValueError(f"A split came out empty after matching: {counts}")

            model = load_unlearned_model(args.model_tag, ckpt, device)
            feature_extractor = _build_feature_extractor(
                model=model, cfg=cfg, feature_dir=str(temp_dir / "features"),
                name=f"loss_extractor_boundary_{method}_{subject}")
            classifier = _build_classifier(cfg=cfg, seed=args.seed)
            if args.class_balance == "class-weight":
                classifier.set_params(class_weight="balanced")

            attacker = Attacker({
                "datasets": {"train": train_csv, "dev": eval_csv, "test": eval_csv},
                "classifier": classifier,
                "feature_extractors": [feature_extractor],
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers,
                "label": "utt_in_set",
                "class_label": "utt_in_set",
            })

            log.info("  training classifier (override=%s)...", cfg.override)
            attacker.train_classifier(override=cfg.override)

            log.info("  extracting eval features...")
            X_eval, y_eval = attacker.extract_features("test", override=cfg.override)
            probs = attacker.classifier.predict_proba(X_eval)[:, 1]
            preds = attacker.classifier.predict(X_eval)
            metrics = compute_binary_mia_metrics(y_eval, probs, preds)
            log.info("  %s", "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                       if isinstance(v, (int, float))))

            # title = (f"{method}/{subject} — decision boundary\n" f"auc={metrics['auc']:.3f}...
            title = "MIA Classifier - decision boundary"
            plot_decision_boundary(X_eval, y_eval, attacker.classifier, FEATURE_NAMES,
                                   title, png_path, args.grid_resolution, args.pad)
        except Exception as exc:                           # noqa: BLE001
            log.exception("%s/%s failed", method, subject)
            failures.append((method, subject, repr(exc)))
        finally:
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        ok = bool(metrics) and png_path.exists()
        if not ok and not any(f[:2] == (method, subject) for f in failures):
            failures.append((method, subject, "no plot produced"))

        payload = {
            "method": method, "forget_subject": subject, "checkpoint": str(ckpt),
            "status": "ok" if ok else "failed",
            "class_balance": args.class_balance, "match_durations": args.match_durations,
            "duration_bins": args.duration_bins, "nonmember_split": args.nonmember_split,
            "per_corpus_match": args.per_corpus_match, "model_tag": args.model_tag,
            "seed": args.seed, "n_estimators": args.n_estimators,
            "splits": counts, "runtime_sec": round(time.time() - t0, 1),
            "metrics": metrics, "png": str(png_path) if ok else None,
        }
        (artifact_dir / "results.json").write_text(json.dumps(payload, indent=2))

        if ok:
            print(f"    -> {png_path}  ({payload['runtime_sec']}s)")
        else:
            print(f"    FAILED after {payload['runtime_sec']}s — see log above")

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
