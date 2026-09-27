#!/usr/bin/env python
"""Plot the LOO *informed* MIA classifier's decision boundary for every organized unlearning..."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from copy import deepcopy
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

from plot_decision_boundary import FEATURE_NAMES, plot_decision_boundary
from run_informed_mia import (
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    MODEL_TAG,
    discover_by_method,
    load_into,
    make_classifier,
    sample_retain_members,
    select_gpu,
)
from mia_common import align_mia_sets_to_forget, duration_stats, match_to_reference_distribution

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("espnet", "espnet2", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("plot_decision_boundary_informed")

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-root", default=str(Path.home() / "unlearned_ckps"))
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results" / "decision_boundary_informed"))
    p.add_argument("--methods", nargs="+", help="only these method folders")
    p.add_argument("--subjects", nargs="+",
                   help="only these subjects as TARGETS. Every subject of the method is "
                        "still available as a shadow model")
    p.add_argument("--device", help="torch device (default: CUDA card with most free memory)")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--min-shadows", type=int, default=2,
                   help="skip a method with fewer shadow models than this (default: 2)")
    p.add_argument("--no-match-distributions", dest="match_distributions",
                   action="store_false",
                   help="skip duration alignment of retain/test and of each shadow's forget "
                        "set onto the target's distribution")
    p.add_argument("--align-n-bins", type=int, default=20)
    p.add_argument("--nonmember-split", choices=("speaker", "utterance"), default="speaker",
                   help="how the test pool is halved into the attacker's training "
                        "negatives and its evaluation negatives (default: speaker)")
    p.add_argument("--no-per-corpus-match", dest="per_corpus_match", action="store_false",
                   help="duration-match the pooled non-member set instead of matching "
                        "test-clean and test-other to the forget speaker separately")
    p.add_argument("--hybrid-retain", type=int, default=0, metavar="N",
                   help="hybrid mode: add N of the TARGET's retain-speaker utterances per "
                        "shadow model as extra members (0 = pure LOO, default)")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--class-weight", default="balanced",
                   help="forest class_weight; 'none' disables (default: balanced)")
    p.add_argument("--train-split-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-save-frequency", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-resolution", type=int, default=400,
                   help="side length of the P(member) contour grid (default: 400)")
    p.add_argument("--pad", type=float, default=0.5,
                   help="axis padding beyond the eval features' min/max (default: 0.5)")
    p.add_argument("--reuse-features", action="store_true",
                   help="keep any feature cache from a previous run instead of clearing it "
                        "before each model's pass")
    p.add_argument("--overwrite", action="store_true",
                   help="rerun targets that already have a decision_boundary.png")
    p.add_argument("-n", "--dry-run", action="store_true", help="list the work and exit")
    return p.parse_args(argv)


def write_summary(out_dir):
    rows = []
    for path in sorted(out_dir.glob("*/*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        row = {k: payload.get(k) for k in
              ("method", "forget_subject", "status", "num_shadow_models",
               "n_shadow_features", "runtime_sec", "png")}
        row["shadow_subjects"] = ",".join(payload.get("shadow_subjects", []))
        row.update(payload.get("metrics") or {})
        rows.append(row)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["method", "forget_subject"])
    summary = out_dir / "decision_boundary_informed_summary.csv"
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

    by_method = discover_by_method(root, set(args.methods or []), set(args.subjects or []))
    if not by_method:
        raise SystemExit(f"No <method>/<subject>/last.ckpt under {root} for those filters.")

    plan, skipped_methods = [], []
    for method, ckpts in by_method.items():
        n_shadows = len(ckpts) - 1
        if n_shadows < args.min_shadows:
            skipped_methods.append((method, n_shadows))
            continue
        targets = [s for s in ckpts if not args.subjects or s in set(args.subjects)]
        pending = [s for s in targets
                   if args.overwrite or not (out_dir / method / s / "decision_boundary.png").exists()]
        if pending:
            plan.append((method, ckpts, pending))

    total_pending = sum(len(p) for _, _, p in plan)
    print(f"{len(by_method)} method(s) discovered; {total_pending} target(s) to run")
    for method, ckpts, pending in plan:
        print(f"  {method}: {len(pending)} target(s), {len(ckpts) - 1} shadow model(s) each "
              f"-> {len(ckpts)} checkpoint load(s) for shadows + {len(pending)} for targets")
    for method, n in skipped_methods:
        print(f"  {method}: SKIPPED — only {n} shadow model(s), below --min-shadows="
              f"{args.min_shadows}")
    if args.dry_run:
        return 0
    if not plan:
        summary, n = write_summary(out_dir)
        print(f"Nothing to do. Summary over {n} row(s): {summary}")
        return 0

    gpu_index, device = select_gpu(args.device)

    from src.data.datasets.csv_dataset import CSVDataset
    from src.evaluation.evaluate_mia import _concat_csvs, _prepare_forget_eval_csvs
    from src.membership_inference.modules import Attacker, LossFeatureExtractorEspnetASR
    from src.membership_inference.utils import compute_binary_mia_metrics
    from src.models.espnet import ESPnetASRWrapper

    print(f"Device: {device} (physical GPU {gpu_index})" if gpu_index is not None
          else f"Device: {device}")
    train_ds, test_ds = CSVDataset(args.train_csv), CSVDataset(args.test_csv)
    datasets = {"train_raw": train_ds, "test_raw": test_ds}
    print(f"Pools: {len(train_ds)} train utterances, {len(test_ds)} test utterances")

    fe_cfg_base = {
        "feature_extractor_name": "LossExtractor",
        "cache_save_frequency": args.cache_save_frequency,
        "loss_att": True, "loss_ctc": True, "loss_cer": False,  # 2D plot only
    }
    att_cfg_base = {
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "class_label": "utt_in_set", "label": "utt_in_set",
        "classifier": make_classifier(args), "feature_extractors": [None],
    }

    log.info("Building base ASR wrapper (%s)...", args.model_tag)
    base_model = ESPnetASRWrapper.from_pretrained(args.model_tag)
    base_model.to(device)

    started, failures, n_done = time.time(), [], 0
    for method, ckpts, pending in plan:
        print(f"\n{'=' * 70}\nMETHOD {method}  —  targets {pending}\n"
              f"shadow pool: {sorted(ckpts)}\n{'=' * 70}")
        method_dir = out_dir / method

        # -- Phase 0 (CPU): per-target splits, aligned to that target's forget speaker.
        splits = {}
        for target in pending:
            tdir = method_dir / target / "target_splits"
            tdir.mkdir(parents=True, exist_ok=True)
            s = _prepare_forget_eval_csvs(datasets, str(tdir), [target], seed=args.seed,
                                          train_split_ratio=args.train_split_ratio,
                                          nonmember_split_by=args.nonmember_split)
            if args.match_distributions:
                log.info("[%s] aligning target %s splits", method, target)
                s = align_mia_sets_to_forget(s, tdir, args.align_n_bins, args.seed,
                                             args.per_corpus_match)
                s["train_csv"] = str(_concat_csvs(
                    [s["retain_train_csv"], s["test_train_csv"]], str(tdir / "mia_train.csv")))
                s["eval_csv"] = str(_concat_csvs(
                    [s["forget_eval_csv"], s["test_eval_csv"]], str(tdir / "mia_eval.csv")))
            splits[target] = s

        # -- Phase 1 (model-major): shadow features, each checkpoint loaded once.
        pooled_X = {t: [] for t in pending}
        pooled_y = {t: [] for t in pending}
        shadow_used = {t: [] for t in pending}
        for shadow in sorted(ckpts):
            consumers = [t for t in pending if t != shadow]
            if not consumers:
                continue
            sdir = method_dir / "_shadow" / shadow
            sdir.mkdir(parents=True, exist_ok=True)
            inner = _prepare_forget_eval_csvs(datasets, str(sdir), [shadow], seed=args.seed,
                                              train_split_ratio=args.train_split_ratio,
                                              nonmember_split_by=args.nonmember_split)
            shadow_forget = pd.read_csv(inner["forget_eval_csv"])

            feats_dir = sdir / "features"
            if not args.reuse_features and feats_dir.exists():
                for f in feats_dir.glob("*.h5"):
                    f.unlink()
            feats_dir.mkdir(parents=True, exist_ok=True)

            log.info("[%s] shadow %s -> %d target(s)", method, shadow, len(consumers))
            shadow_model = load_into(base_model, ckpts[shadow], device)
            fe_cfg = dict(fe_cfg_base, save_features_folder=str(feats_dir))
            extractor = LossFeatureExtractorEspnetASR(shadow_model, fe_cfg)

            for target in consumers:
                pair_dir = sdir / f"for_{target}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                forget_csv = inner["forget_eval_csv"]
                if args.match_distributions:
                    target_forget = pd.read_csv(splits[target]["forget_eval_csv"])
                    matched = match_to_reference_distribution(
                        shadow_forget, target_forget, n_bins=args.align_n_bins,
                        seed=args.seed, label=f"shadow {shadow}->{target}")
                    forget_csv = str(pair_dir / "forget_eval_matched.csv")
                    matched.to_csv(forget_csv, index=False)
                    log.info("    shadow %s forget set for target %s: %s",
                             shadow, target, duration_stats(matched))

                member_csvs = [forget_csv]
                if args.hybrid_retain:
                    extra = sample_retain_members(splits[target]["retain_train_csv"],
                                                  shadow, args.hybrid_retain, args.seed)
                    extra_path = pair_dir / "retain_members.csv"
                    extra.to_csv(extra_path, index=False)
                    member_csvs.append(str(extra_path))
                    log.info("    + %d retain member(s) from %d speaker(s)",
                             len(extra), extra["speaker_id"].nunique() if len(extra) else 0)
                train_csv = _concat_csvs(member_csvs + [splits[target]["test_train_csv"]],
                                         str(pair_dir / "train.csv"))
                att_cfg = deepcopy(att_cfg_base)
                att_cfg["datasets"] = {"train": train_csv,
                                       "dev": splits[target]["test_eval_csv"],
                                       "test": splits[target]["test_eval_csv"]}
                attacker = Attacker(att_cfg)
                attacker.feature_extractors = [extractor]
                X, y = attacker.extract_features(split="train", override=False)
                pooled_X[target].append(X)
                pooled_y[target].append(y)
                shadow_used[target].append(shadow)
                del attacker

            del extractor
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # -- Phase 2 (model-major): fit on pooled shadows, score+plot the target's own model.
        for target in pending:
            n_done += 1
            print(f"\n[{n_done}/{total_pending}] {method}/{target}  "
                  f"({len(shadow_used[target])} shadow model(s))")
            tdir = method_dir / target
            png_path = tdir / "decision_boundary.png"
            t0 = time.time()
            metrics, n_feats = {}, None
            try:
                X_all = np.vstack(pooled_X[target])
                y_all = np.concatenate(pooled_y[target])
                n_feats = int(X_all.shape[0])
                log.info("  pooled shadow features: %s  (members %d / non-members %d)",
                         X_all.shape, int((y_all == 1).sum()), int((y_all == 0).sum()))
                clf = make_classifier(args)
                clf.fit(X_all, y_all)

                target_model = load_into(base_model, ckpts[target], device)
                tfeats = tdir / "target_features"
                tfeats.mkdir(parents=True, exist_ok=True)
                if not args.reuse_features:
                    for f in tfeats.glob("*.h5"):
                        f.unlink()
                fe_cfg = dict(fe_cfg_base, save_features_folder=str(tfeats))

                att_cfg = deepcopy(att_cfg_base)
                att_cfg["datasets"] = {"train": splits[target]["train_csv"],
                                       "dev": splits[target]["eval_csv"],
                                       "test": splits[target]["eval_csv"]}   # mixed: forget + test
                evaluator = Attacker(att_cfg)
                evaluator.feature_extractors = [
                    LossFeatureExtractorEspnetASR(target_model, fe_cfg)]
                evaluator.classifier = clf

                X_eval, y_eval = evaluator.extract_features("test", override=False)
                probs = evaluator.classifier.predict_proba(X_eval)[:, 1]
                preds = evaluator.classifier.predict(X_eval)
                metrics = compute_binary_mia_metrics(y_eval, probs, preds)
                log.info("  %s", "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                           if isinstance(v, (int, float))))

                # title = (f"{method}/{target} — informed decision boundary "...
                title = "MIA Classifier - decision boundary"
                plot_decision_boundary(X_eval, y_eval, evaluator.classifier, FEATURE_NAMES,
                                       title, png_path, args.grid_resolution, args.pad)
                del evaluator
            except Exception as exc:                        # noqa: BLE001
                log.exception("%s/%s failed", method, target)
                failures.append((method, target, repr(exc)))
            finally:
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            ok = bool(metrics) and png_path.exists()
            if not ok and not any(f[:2] == (method, target) for f in failures):
                failures.append((method, target, "no plot produced"))

            payload = {
                "method": method, "forget_subject": target,
                "checkpoint": str(ckpts[target]),
                "status": "ok" if ok else "failed",
                "num_shadow_models": len(shadow_used[target]),
                "shadow_subjects": shadow_used[target],
                "n_shadow_features": n_feats,
                "match_distributions": args.match_distributions,
                "align_n_bins": args.align_n_bins,
                "nonmember_split": args.nonmember_split,
                "per_corpus_match": args.per_corpus_match,
                "hybrid_retain": args.hybrid_retain,
                "class_weight": args.class_weight, "n_estimators": args.n_estimators,
                "seed": args.seed, "train_split_ratio": args.train_split_ratio,
                "runtime_sec": round(time.time() - t0, 1),
                "metrics": metrics, "png": str(png_path) if ok else None,
            }
            (tdir / "results.json").write_text(json.dumps(payload, indent=2))

            if ok:
                print(f"    -> {png_path}  ({payload['runtime_sec']}s)")
            else:
                print(f"    FAILED after {payload['runtime_sec']}s — see log above")

        pooled_X.clear()
        pooled_y.clear()
        gc.collect()

    summary, n_rows = write_summary(out_dir)
    print(f"\nRan {total_pending} target(s) in {(time.time() - started) / 60:.1f} min")
    if failures:
        print(f"{len(failures)} failed:")
        for method, target, why in failures:
            print(f"  {method}/{target}: {why}")
    print(f"Summary over {n_rows} row(s): {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
