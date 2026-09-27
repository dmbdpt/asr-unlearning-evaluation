#!/usr/bin/env python
"""Join the reference baselines onto each attack's post-unlearning summary."""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# attack -> (post summary, {prefix: baseline summary}, join keys beyond the subject)
PAIRS = {
    "simple": ("results/simple_mia/simple_mia_summary.csv",
               {"pre": "results/pre_simple_mia/simple_mia_summary.csv",
                "retrain": "results/retrain_simple_mia/simple_mia_summary.csv"}, ["level"]),
    "informed": ("results/informed_mia/informed_mia_summary.csv",
                 {"pre": "results/pre_informed_mia/informed_mia_summary.csv",
                  "retrain": "results/retrain_informed_mia/informed_mia_summary.csv"},
                 ["eval_set"]),
    # The hybrid variant gets the hybrid baselines: a shadow pool that includes retain...
    "informed_hybrid": ("results/informed_mia_hybrid/informed_mia_summary.csv",
                        {"pre": "results/pre_informed_mia_hybrid/informed_mia_summary.csv",
                         "retrain": "results/retrain_informed_mia_hybrid/informed_mia_summary.csv"},
                        ["eval_set"]),
}
SUBJECT = "forget_subject"

# The per-attack split column, normalised into one "split" field in the combined CSV.
SPLIT_COLUMN = {"simple": "level", "informed": "eval_set",
                "informed_hybrid": "eval_set", "ez": None}
STAGES = ("post", "pre", "retrain")
# Default view: utterance-level, full eval set.
DEFAULT_SPLIT = {
    "simple": ("level", ("utterance", "utterance:test-clean", "utterance:test-other")),
    "informed": ("eval_set", ("mixed", "mixed:test-clean", "mixed:test-other")),
    "informed_hybrid": ("eval_set", ("mixed", "mixed:test-clean", "mixed:test-other")),
    "ez": None,
}
# Metrics listed first, grouped so the three stages of a metric sit side by side.
LEAD_METRICS = ["auc", "uar", "eer", "forget_detection_rate",
                "member_score_mean", "nonmember_score_mean",
                "tpr@maxfpr=10%", "tpr@maxfpr=1%", "tpr@maxfpr=0.1%",
                "fpr@maxfpr=10%", "fpr@maxfpr=1%", "fpr@maxfpr=0.1%",
                "tpr@fpr=10%", "tpr@fpr=1%", "tpr@fpr=0.1%",
                "fpr@fpr=10%", "fpr@fpr=1%", "fpr@fpr=0.1%"]


def _as_tuple(keep):
    return keep if isinstance(keep, tuple) else (keep,)


def build_combined(frames, out_path):
    """Fold every attack into one table: one row per (attack, method, subject, split)."""
    combined = []
    for attack, df in frames.items():
        df = df.copy()
        split_col = SPLIT_COLUMN[attack]
        df.insert(0, "attack", attack)
        df["split"] = df[split_col] if split_col else "utterance"
        if split_col:
            df = df.drop(columns=[split_col])
        combined.append(df)
    out = pd.concat(combined, ignore_index=True, sort=False)

    ident = ["attack", "method", SUBJECT, "split"]
    status = [f"{st}_status" for st in STAGES if f"{st}_status" in out.columns]
    metrics = [f"{st}_{m}" for m in LEAD_METRICS for st in STAGES
               if f"{st}_{m}" in out.columns]
    rest = [c for c in out.columns if c not in ident + status + metrics]
    out = out[ident + status + metrics + sorted(rest)]
    out = out.sort_values(["attack", "method", "split", SUBJECT])
    out.to_csv(out_path, index=False)
    return out


def merge_one(post_path, baselines, extra_keys, out_path, level_filter=None):
    post = pd.read_csv(post_path)
    if level_filter:
        col, keep = level_filter
        post = post[post[col].isin(_as_tuple(keep))]
    post[SUBJECT] = post[SUBJECT].astype(str)
    keys = [SUBJECT] + list(extra_keys)
    missing = [k for k in keys if k not in post.columns]
    if missing:
        raise SystemExit(f"{out_path}: join key(s) absent from post: {missing}")

    # "method" names the unlearning method on the post side and is a constant tag on each...
    merged = post.rename(columns={c: f"post_{c}" for c in post.columns if c not in keys})

    unmatched = {}
    for prefix, path in baselines.items():
        path = Path(path)
        if not path.is_file():
            print(f"{'':<16} baseline '{prefix}' missing ({path}) — skipped")
            continue
        base = pd.read_csv(path)
        if level_filter:
            col, keep = level_filter
            base = base[base[col].isin(_as_tuple(keep))]
        base[SUBJECT] = base[SUBJECT].astype(str)
        absent = [k for k in keys if k not in base.columns]
        if absent:
            raise SystemExit(f"{path}: join key(s) absent: {absent}")
        dup = base.duplicated(subset=keys).sum()
        if dup:
            raise SystemExit(f"{path}: {dup} duplicate baseline row(s) for {keys}")
        base = base.rename(columns={c: f"{prefix}_{c}" for c in base.columns if c not in keys})
        merged = merged.merge(base, on=keys, how="left", validate="many_to_one")
        # Probe a column that is never NaN by design; "*_auc" is legitimately NaN on single-class...
        probe = f"{prefix}_status"
        unmatched[prefix] = int(merged[probe].isna().sum()) if probe in merged else 0

    lead = ["post_method"] + keys
    ordered = lead + [c for c in merged.columns if c not in lead]
    merged = merged[ordered].rename(columns={"post_method": "method"})
    merged.to_csv(out_path, index=False)
    return merged, unmatched


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out-dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--variant", default="",
                   help="suffix appended to every input results directory and to the "
                        "output filenames, e.g. --variant v2 reads results/simple_mia_v2/ "
                        "and writes results/mia_all_results_v2.csv. Lets a rerun under "
                        "changed split rules sit beside the previous one instead of "
                        "overwriting it")
    p.add_argument("--all-splits", action="store_true",
                   help="keep every split, including simple's speaker-level pass and "
                        "informed's single-class forget_only eval set; by default only "
                        "the utterance-level full eval set is kept")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    tag = f"_{args.variant}" if args.variant else ""

    def variant_path(rel):
        """results/<dir>/<file> -> results/<dir><tag>/<file>."""
        parts = Path(rel).parts
        return REPO_ROOT.joinpath(*parts[:-2], parts[-2] + tag, parts[-1])

    frames = {}
    for name, (post_p, baselines, extra) in PAIRS.items():
        post_p = variant_path(post_p)
        if not post_p.is_file():
            print(f"{name:<16} SKIPPED (missing {post_p})")
            continue
        baselines = {k: variant_path(v) for k, v in baselines.items()}
        out = out_dir / f"{name}_mia_baselines{tag}.csv"
        lvl = None if args.all_splits else DEFAULT_SPLIT.get(name)
        merged, unmatched = merge_one(post_p, baselines, extra, out, level_filter=lvl)
        note = "".join(f"   ({n} {p}_ row(s) unjoined)" for p, n in unmatched.items() if n)
        have = " + ".join(p for p in unmatched)
        print(f"{name:<16} {len(merged):>4} rows x {len(merged.columns):>3} cols "
              f"[post + {have}] -> {out.relative_to(REPO_ROOT)}{note}")
        frames[name] = merged
    if frames:
        combined_path = out_dir / f"mia_all_results{tag}.csv"
        combined = build_combined(frames, combined_path)
        print(f"\ncombined      {len(combined):>4} rows x {len(combined.columns):>3} cols -> "
              f"{combined_path.relative_to(REPO_ROOT)}")
        print(f"  attacks: {', '.join(sorted(combined.attack.unique()))}")
        print(f"  splits : {', '.join(sorted(combined.split.unique()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
