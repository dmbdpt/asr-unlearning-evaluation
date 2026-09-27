#!/usr/bin/env python
"""Duration-matched loss-feature MIA against unlearned checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch

from mia_common import (
    DEFAULT_TEST_CSV, DEFAULT_TRAIN_CSV, FORGET_SET, LEVELS, MODEL_TAG,
    align_mia_sets_to_forget, build_cfg, build_train_eval_csvs, discover_checkpoints,
    load_attack_model, load_unlearned_model, log, run_attacks, select_gpu,
    write_duration_summary, write_summary,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- What to attack -----------------------------------------------------
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoints-root", help="tree of <method>/<subject>/last.ckpt; "
                        "each subject is attacked as its own forget speaker")
    target.add_argument("--checkpoint", help="a single checkpoint, attacked once")

    p.add_argument("--methods", nargs="+", help="tree mode: only these methods")
    p.add_argument("--subjects", nargs="+", help="tree mode: only these subjects")
    p.add_argument("--label", help="single-checkpoint mode: name for the output subdirectory")
    p.add_argument("--forget-speakers", nargs="+",
                   help="single-checkpoint mode: the speakers this checkpoint forgot. "
                        "All of them form one forget set unless --subject picks one out.")
    p.add_argument("--subject", help="single-checkpoint mode: attack this one speaker of "
                                     "--forget-speakers, with the rest merely excluded")

    # ---- Pools --------------------------------------------------------------
    p.add_argument("--exclude-forget-set", action="store_true",
                   help="drop every speaker in the run's forget set from the retain pool, "
                        "instead of counting the ones not under attack as retained data")
    p.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--test-csv", default=str(DEFAULT_TEST_CSV))
    p.add_argument("--nonmember-split", choices=("speaker", "utterance"), default="speaker")
    p.add_argument("--match-durations", action="store_true", default=True)
    p.add_argument("--no-match-durations", dest="match_durations", action="store_false")
    p.add_argument("--per-corpus-match", action="store_true", default=True)
    p.add_argument("--no-per-corpus-match", dest="per_corpus_match", action="store_false")
    p.add_argument("--duration-bins", type=int, default=20)
    p.add_argument("--class-balance", choices=("none", "downsample", "class-weight"),
                   default="class-weight")

    # ---- Attack -------------------------------------------------------------
    p.add_argument("--stages", nargs="+", choices=("pre", "post"), default=["post"],
                   help="'pre' attacks the pretrained model through the same pools, "
                        "giving the leakage upper bound alongside 'post'")
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    p.add_argument("--model-tag", default=MODEL_TAG)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--mia-train-split-ratio", type=float, default=0.5)
    p.add_argument("--loss-cer", action="store_true")
    p.add_argument("--share-features", action="store_true", default=True)
    p.add_argument("--no-share-features", dest="share_features", action="store_false")
    p.add_argument("--reuse-features", action="store_true")

    # ---- Run ----------------------------------------------------------------
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results" / "simple_mia"))
    p.add_argument("--device", help="cuda:N (default: the card with most free memory)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-n", "--dry-run", action="store_true")

    args = p.parse_args(argv)
    if args.checkpoint and not args.label:
        p.error("--checkpoint requires --label")
    if args.subject and not args.forget_speakers:
        p.error("--subject requires --forget-speakers")
    if args.subject and args.subject not in [str(s) for s in args.forget_speakers]:
        p.error(f"--subject {args.subject} must be one of --forget-speakers")
    return args


def _indices_excluding(train_raw, attacked, excluded):
    """forget_idx = the attacked speakers; retain_idx = everyone outside `excluded`."""
    attacked, excluded = {str(s) for s in attacked}, {str(s) for s in excluded}
    forget_idx, retain_idx = [], []
    for i in range(len(train_raw)):
        spk = str(train_raw.get_metadata(i, dicted=True)["speaker_id"])
        if spk in attacked:
            forget_idx.append(i)
        elif spk not in excluded:
            retain_idx.append(i)
    return forget_idx, retain_idx


def attack_one(ckpt, attacked, excluded, artifact_dir, args, cfg, device,
               train_raw, test_raw):
    """Build the duration-matched pools for one target, then attack it."""
    from src.evaluation.evaluate_mia import _prepare_forget_eval_csvs

    temp_dir = artifact_dir / "mia_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    datasets = {"train_raw": train_raw, "test": test_raw}
    if excluded:
        forget_idx, retain_idx = _indices_excluding(train_raw, attacked, excluded)
        datasets["forget_idx"], datasets["retain_idx"] = forget_idx, retain_idx
        print(f"  forget_idx={len(forget_idx)} retain_idx={len(retain_idx)} "
              f"({len(train_raw) - len(forget_idx) - len(retain_idx)} utterances excluded)")

    csv_sets = _prepare_forget_eval_csvs(
        datasets=datasets,
        temp_dir=str(temp_dir),
        forget_speakers=[str(s) for s in attacked],
        seed=args.seed,
        train_split_ratio=cfg.mia_train_split_ratio,
        nonmember_split_by=args.nonmember_split,
    )
    aligned = (align_mia_sets_to_forget(csv_sets, temp_dir, args.duration_bins,
                                        args.seed, args.per_corpus_match)
               if args.match_durations else csv_sets)
    train_csv, eval_csv, counts = build_train_eval_csvs(
        aligned, temp_dir, args.class_balance, args.seed)
    print(f"  train: {counts['n_train_members']} utt / {counts['n_train_member_subjects']} subj "
          f"(members)  vs  {counts['n_train_nonmembers']} utt / "
          f"{counts['n_train_nonmember_subjects']} subj (non-members)")
    print(f"  eval:  {counts['n_eval_members']} utt / {counts['n_eval_member_subjects']} subj "
          f"(members)  vs  {counts['n_eval_nonmembers']} utt / "
          f"{counts['n_eval_nonmember_subjects']} subj (non-members)")

    stage_results = {}
    for stage in args.stages:
        t0, model = time.time(), None
        try:
            if stage == "pre":
                from src.models.espnet import ESPnetASRWrapper
                model = ESPnetASRWrapper.from_pretrained(args.model_tag)
                model.to(device)
                model.eval()
            else:
                model = (load_unlearned_model(args.model_tag, ckpt, device)
                         if not str(ckpt).endswith(".json")
                         else load_attack_model(ckpt, args.model_tag, device))
            stage_results[stage] = run_attacks(
                model, train_csv, eval_csv, cfg, str(temp_dir / stage),
                args.levels, args.seed, args.class_balance, args.share_features)
            for level, m in stage_results[stage].items():
                if isinstance(m, dict) and "auc" in m:
                    print(f"    [{stage}/{level}] auc={m['auc']:.4f} uar={m['uar']:.4f} "
                          f"eer={m['eer']:.4f}  ({time.time()-t0:.0f}s)")
        finally:
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # A single post-only run keeps the flat shape the summary writers expect.
    results = stage_results["post"] if args.stages == ["post"] else stage_results
    return results, counts


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    for csv in (args.train_csv, args.test_csv):
        if not Path(csv).is_file():
            raise SystemExit(f"CSV not found: {csv}")

    # (checkpoint, attacked speakers, excluded speakers, artifact dir)
    jobs = []
    if args.checkpoints_root:
        root = Path(args.checkpoints_root).expanduser()
        if not root.is_dir():
            raise SystemExit(f"Checkpoint root not found: {root}")
        work = discover_checkpoints(root, set(args.methods or []), set(args.subjects or []))
        if not work:
            raise SystemExit(f"No <method>/<subject>/last.ckpt under {root} for those filters.")
        for method, subject, ckpt in work:
            excluded = FORGET_SET if args.exclude_forget_set else []
            jobs.append((ckpt, [subject], excluded, out_dir / method / subject))
    else:
        ckpt = Path(args.checkpoint).expanduser()
        if not ckpt.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        forget = [str(s) for s in (args.forget_speakers or [])]
        attacked = [args.subject] if args.subject else forget
        excluded = forget if (args.exclude_forget_set or args.subject) else []
        jobs.append((ckpt, attacked, excluded, out_dir / args.label))

    pending = [j for j in jobs if args.overwrite or not (j[3] / "results.json").exists()]
    print(f"{len(jobs)} target(s); {len(pending)} to run, {len(jobs)-len(pending)} already done")
    print(f"Stages: {', '.join(args.stages)} | levels: {', '.join(args.levels)} | "
          f"duration matching: {'on' if args.match_durations else 'OFF'}"
          f"{' (per corpus)' if args.per_corpus_match else ' (pooled)'} | "
          f"class balance: {args.class_balance} | non-member split: by {args.nonmember_split}")
    if args.dry_run:
        for ckpt, attacked, excluded, d in pending:
            note = f"  excluded={sorted(set(excluded) - set(attacked))}" if excluded else ""
            print(f"  {d.relative_to(out_dir)}  forget={attacked}{note}  <- {ckpt}")
        return 0
    if not pending:
        summary, n = write_summary(out_dir)
        dur_summary, dur_n = write_duration_summary(out_dir)
        print(f"Nothing to do. Summary over {n} row(s): {summary}")
        print(f"Duration-match summary over {dur_n} row(s): {dur_summary}")
        return 0

    gpu_index, device = select_gpu(args.device)
    print(f"Device: {device} (physical GPU {gpu_index})" if gpu_index is not None
          else f"Device: {device}")
    print(f"Train CSV: {args.train_csv}\nTest CSV:  {args.test_csv}")

    from src.data.datasets.csv_dataset import CSVDataset
    cfg = build_cfg(args)
    train_raw = CSVDataset(args.train_csv)
    test_raw = CSVDataset(args.test_csv)
    print(f"Pools: {len(train_raw)} train utterances, {len(test_raw)} test utterances")

    started, failures = time.time(), []
    for n_done, (ckpt, attacked, excluded, artifact_dir) in enumerate(pending, start=1):
        print(f"\n[{n_done}/{len(pending)}] {artifact_dir.relative_to(out_dir)}  "
              f"(forget {', '.join(attacked)})")
        t0 = time.time()
        results, counts = {}, {}
        try:
            results, counts = attack_one(ckpt, attacked, excluded, artifact_dir,
                                         args, cfg, device, train_raw, test_raw)
        except Exception:
            import traceback
            traceback.print_exc()
            failures.append(str(artifact_dir))

        runtime = time.time() - t0
        # A level that ran but came back empty is a failure too, not a null finding.
        ok = bool(results) and all(results.get(lv) for lv in args.levels)
        if not ok and str(artifact_dir) not in failures:
            failures.append(str(artifact_dir))
        # Tree mode: <out_dir>/<method>/<subject>. Single-checkpoint mode has no method.
        parts = artifact_dir.relative_to(out_dir).parts
        method = parts[0] if len(parts) > 1 else None
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "results.json").write_text(json.dumps({
            "method": method,
            "checkpoint": str(ckpt),
            "status": "ok" if ok else "failed",
            "forget_subject": attacked[0] if len(attacked) == 1 else None,
            "forget_speakers": attacked,
            "excluded_speakers": sorted(set(excluded) - set(attacked)) if excluded else [],
            "stages": args.stages,
            "levels": args.levels,
            "class_balance": args.class_balance,
            "match_durations": args.match_durations,
            "duration_bins": args.duration_bins,
            "nonmember_split": args.nonmember_split,
            "per_corpus_match": args.per_corpus_match,
            "model_tag": args.model_tag,
            "seed": args.seed,
            "runtime_sec": runtime,
            "splits": counts,
            # write_summary (mia_common.py) reads this key as "metrics", matching
            # run_informed_mia.py's schema and the pre-refactor batch_simple_mia.py.
            "metrics": results,
        }, indent=2, default=str))
        print(f"  done in {runtime/60:.1f} min")

    summary, n = write_summary(out_dir)
    dur_summary, dur_n = write_duration_summary(out_dir)
    print(f"\nSummary over {n} row(s): {summary}")
    print(f"Duration-match summary over {dur_n} row(s): {dur_summary}")
    print(f"Total: {(time.time()-started)/60:.1f} min")
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
