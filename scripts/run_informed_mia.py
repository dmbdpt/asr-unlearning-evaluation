#!/usr/bin/env python
"""Leave-one-out *informed* MIA over every organized unlearning checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
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
from sklearn.ensemble import RandomForestClassifier

from mia_common import (align_mia_sets_to_forget, duration_stats,
                              evaluate_by_corpus, match_to_reference_distribution)
from mia_common import find_model_spec, load_attack_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("espnet", "espnet2", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("run_informed_mia")

MODEL_TAG = "asapp/e_branchformer_librispeech"
DATA_DIR = Path(os.environ.get("LEAF_DATA_DIR", REPO_ROOT / "data"))
DEFAULT_TRAIN_CSV = DATA_DIR / "ls_train_all.csv"
DEFAULT_TEST_CSV = DATA_DIR / "ls_test_all.csv"

EVAL_SETS = ("mixed", "forget_only")


def forget_detection_rate(forget_only_metrics):
    """Fraction of the forgotten speaker's utterances still classified as members."""
    uar = (forget_only_metrics or {}).get("uar")
    if uar is None or not isinstance(uar, (int, float)) or np.isnan(uar):
        return None
    return 1.0 if uar > 0.5 else 2.0 * uar
METRIC_COLUMNS = [
    "auc", "uar", "eer",
    "tpr@fpr=10%", "fpr@fpr=10%",
    "tpr@fpr=1%", "fpr@fpr=1%",
    "tpr@fpr=0.1%", "fpr@fpr=0.1%",
    # budget-respecting operating points (fpr <= target); the "@fpr=" keys above take the...
    "tpr@maxfpr=10%", "fpr@maxfpr=10%",
    "tpr@maxfpr=1%", "fpr@maxfpr=1%",
    "tpr@maxfpr=0.1%", "fpr@maxfpr=0.1%",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints-root", default=str(Path.home() / "unlearned_ckps"))
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results" / "informed_mia"))
    p.add_argument("--methods", nargs="+", help="only these method folders")
    p.add_argument("--subjects", nargs="+",
                   help="only these subjects as TARGETS. Every subject of the method is "
                        "still available as a shadow model")
    p.add_argument("--device", help="torch device (default: CUDA card with most free memory)")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--min-shadows", type=int, default=2,
                   help="skip a method with fewer shadow models than this; the pool size is "
                        "(subjects in the method - 1), so methods are not equally informed "
                        "(default: 2)")
    p.add_argument("--no-match-distributions", dest="match_distributions",
                   action="store_false",
                   help="skip duration alignment of retain/test and of each shadow's forget "
                        "set onto the target's distribution")
    p.add_argument("--align-n-bins", type=int, default=20)
    p.add_argument("--nonmember-split", choices=("speaker", "utterance"), default="speaker",
                   help="how the test pool is halved into the attacker's training "
                        "negatives and its evaluation negatives; 'speaker' (default) "
                        "keeps the halves speaker-disjoint, 'utterance' is the old "
                        "behaviour that put every test speaker on both sides")
    p.add_argument("--no-per-corpus-match", dest="per_corpus_match", action="store_false",
                   help="duration-match the pooled non-member set instead of matching "
                        "test-clean and test-other to the forget speaker separately")
    p.add_argument("--hybrid-retain", type=int, default=0, metavar="N",
                   help="hybrid mode: add N of the TARGET's retain-speaker utterances per "
                        "shadow model as extra members, sampled one-per-speaker first. The "
                        "pure LOO pool has one member speaker per shadow, which underpowers "
                        "the attack on methods that leave no unlearning signature. 0 (default) "
                        "reproduces run_mia_loo.py exactly")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--class-weight", default="balanced",
                   help="forest class_weight; 'none' disables (default: balanced, as in "
                        "run_mia_loo.py)")
    p.add_argument("--loss-cer", action="store_true",
                   help="add CER to the loss feature vector (off by default)")
    p.add_argument("--train-split-ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-save-frequency", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse-features", action="store_true",
                   help="keep any shadow feature cache from a previous run instead of "
                        "clearing it before each shadow model's pass")
    p.add_argument("--overwrite", action="store_true",
                   help="rerun targets that already have results.json")
    p.add_argument("-n", "--dry-run", action="store_true", help="list the work and exit")
    return p.parse_args(argv)


def discover_by_method(root, methods, subjects):
    """{method: {subject: spec}} for every attackable <root>/<method>/<subject>/."""
    found = {}
    for method_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and not p.name.startswith(("_", "."))):
        if methods and method_dir.name not in methods:
            continue
        ckpts = {}
        for subj_dir in sorted(method_dir.iterdir(),
                               key=lambda p: (not p.name.isdigit(), p.name)):
            if not subj_dir.is_dir():
                continue
            spec = find_model_spec(subj_dir)
            if spec is not None:
                ckpts[subj_dir.name] = spec
        if ckpts:
            found[method_dir.name] = ckpts
    if subjects:
        # Subjects filter targets only — every checkpoint stays available as a shadow.
        found = {m: c for m, c in found.items() if set(c) & set(subjects)}
    return found


def select_gpu(requested):
    """Pin one physical GPU via CUDA_VISIBLE_DEVICES and report it as cuda:0."""
    if requested == "cpu":
        return None, "cpu"
    if requested:
        index = requested.split(":")[-1]
        if not index.isdigit():
            raise SystemExit(f"--device must be 'cpu' or 'cuda:N', got {requested!r}")
    else:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            log.warning("nvidia-smi unavailable; falling back to CPU")
            return None, "cpu"
        cards = [(int(free), idx.strip()) for idx, free in
                 (line.split(",") for line in out.strip().splitlines())]
        if not cards:
            return None, "cpu"
        index = max(cards)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    return index, "cuda:0"


def load_into(base_model, spec, device):
    """Eval-mode wrapper for `spec`, reusing `base_model` where that is valid."""
    return load_attack_model(spec, MODEL_TAG, device, base_model=base_model)


def sample_retain_members(retain_csv, exclude_speaker, n, seed):
    """Extra member rows for the hybrid pool, drawn from the target's retain speakers."""
    df = pd.read_csv(retain_csv)
    df = df[df["speaker_id"].astype(str) != str(exclude_speaker)]
    if df.empty or n <= 0:
        return df.iloc[0:0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order]
    first = df.groupby("speaker_id", sort=False).head(1)
    if len(first) >= n:
        return first.iloc[:n]
    rest = df.drop(index=first.index)
    return pd.concat([first, rest.iloc[: n - len(first)]], ignore_index=True)


def make_classifier(args):
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight=None if args.class_weight == "none" else args.class_weight,
    )


def write_summary(out_dir):
    """Rebuild the summary CSV from every results.json currently on disk."""
    rows = []
    for path in sorted(out_dir.glob("*/*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        base = {k: payload.get(k) for k in
                ("method", "forget_subject", "status", "num_shadow_models",
                 "match_distributions", "align_n_bins", "hybrid_retain",
                 "nonmember_split", "per_corpus_match",
                 "n_shadow_features", "runtime_sec", "forget_detection_rate")}
        base["shadow_subjects"] = ",".join(payload.get("shadow_subjects", []))
        # Iterate the metrics dict, not EVAL_SETS: it also carries the per-corpus slices...
        for eval_set, metrics in (payload.get("metrics") or {}).items():
            if not metrics:
                continue
            row = dict(base, eval_set=eval_set)
            row.update({k: metrics.get(k) for k in METRIC_COLUMNS})
            row.update({k: metrics.get(k) for k in ("n_members", "n_nonmembers")})
            rows.append(row)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["eval_set", "method", "forget_subject"])
    summary = out_dir / "informed_mia_summary.csv"
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
                   if args.overwrite or not (out_dir / method / s / "results.json").exists()]
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
    from src.models.espnet import ESPnetASRWrapper

    print(f"Device: {device} (physical GPU {gpu_index})" if gpu_index is not None
          else f"Device: {device}")
    train_ds, test_ds = CSVDataset(args.train_csv), CSVDataset(args.test_csv)
    datasets = {"train_raw": train_ds, "test_raw": test_ds}
    print(f"Pools: {len(train_ds)} train utterances, {len(test_ds)} test utterances")

    fe_cfg_base = {
        "feature_extractor_name": "LossExtractor",
        "cache_save_frequency": args.cache_save_frequency,
        "loss_att": True, "loss_ctc": True, "loss_cer": args.loss_cer,
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
                # Scoped to this shadow model; cleared so no other model's cache can leak in, then shared...
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

        # -- Phase 2 (model-major): fit on pooled shadows, score the target's own model.
        for target in pending:
            n_done += 1
            print(f"\n[{n_done}/{total_pending}] {method}/{target}  "
                  f"({len(shadow_used[target])} shadow model(s))")
            tdir = method_dir / target
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

                for eval_set, test_csv in (("mixed", splits[target]["eval_csv"]),
                                           ("forget_only", splits[target]["forget_eval_csv"])):
                    att_cfg = deepcopy(att_cfg_base)
                    att_cfg["datasets"] = {"train": splits[target]["train_csv"],
                                           "dev": splits[target]["eval_csv"],
                                           "test": test_csv}
                    evaluator = Attacker(att_cfg)
                    evaluator.feature_extractors = [
                        LossFeatureExtractorEspnetASR(target_model, fe_cfg)]
                    evaluator.classifier = deepcopy(clf)
                    if eval_set == "mixed":
                        # forget_only has no non-members, so there is nothing to break down by corpus there.
                        by_corpus = evaluate_by_corpus(evaluator, "test", False, test_csv)
                        metrics[eval_set] = by_corpus["overall"]
                        for name, m in by_corpus.items():
                            if name != "overall":
                                metrics[f"{eval_set}:{name}"] = m
                    else:
                        metrics[eval_set] = evaluator.evaluate(split="test", override=False)
                    del evaluator
            except Exception as exc:                        # noqa: BLE001
                log.exception("%s/%s failed", method, target)
                failures.append((method, target, repr(exc)))
            finally:
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            ok = bool(metrics.get("mixed"))
            if not ok and not any(f[:2] == (method, target) for f in failures):
                failures.append((method, target, "attack returned no metrics"))

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
                "loss_features": {"att": True, "ctc": True, "cer": args.loss_cer},
                "runtime_sec": round(time.time() - t0, 1),
                "forget_detection_rate": forget_detection_rate(metrics.get("forget_only")),
                "metrics": metrics,
            }
            (tdir / "results.json").write_text(json.dumps(payload, indent=2))

            if ok:
                for eval_set in EVAL_SETS:
                    m = metrics.get(eval_set) or {}
                    shown = "  ".join(f"{k}={m[k]:.4f}" for k in ("auc", "uar", "eer")
                                      if isinstance(m.get(k), float) and not np.isnan(m[k]))
                    print(f"    {eval_set:<12} {shown or '(no finite metrics)'}")
                rate = payload["forget_detection_rate"]
                if rate is not None:
                    print(f"    -> {rate:.1%} of {target}'s forgotten utterances still "
                          f"classified as members")
                print(f"    ({payload['runtime_sec']}s)")
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
