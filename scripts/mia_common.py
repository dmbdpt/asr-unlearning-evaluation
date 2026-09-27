"""Shared machinery for the batch MIA, EMD and decision-boundary scripts."""
from __future__ import annotations
import argparse
import gc
import json
import logging
import os
import subprocess
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
from omegaconf import OmegaConf
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("espnet", "espnet2", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("mia")
torch.multiprocessing.set_sharing_strategy('file_system')
MODEL_TAG = "asapp/e_branchformer_librispeech"
DATA_DIR = Path(os.environ.get("LEAF_DATA_DIR", REPO_ROOT / "data"))
DEFAULT_TRAIN_CSV = DATA_DIR / "ls_train_all.csv"
DEFAULT_TEST_CSV = DATA_DIR / "ls_test_all.csv"
LEVELS = ("utterance", "speaker")
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
SPLIT_COLUMNS = ["n_train_members", "n_train_nonmembers",
                 "n_eval_members", "n_eval_nonmembers",
                 "n_train_member_subjects", "n_train_nonmember_subjects",
                 "n_eval_member_subjects", "n_eval_nonmember_subjects"]
METRIC_COLUMNS += ["n_members", "n_nonmembers"]

# The ten speakers unlearned in the paper, from LibriSpeech train-clean-100.
FORGET_SET = ["103", "1034", "1040", "1069", "1081", "1088", "1098", "1116", "118", "1183"]


# ---------------------------------------------------------------------------
# Checkpoint and model loading
# ---------------------------------------------------------------------------

SPEC_FILENAMES = ("last.ckpt", "espnet_model.json")
_NATIVE_CACHE: dict[str, object] = {}
def find_model_spec(subject_dir):
    """The attackable artifact inside a <method>/<subject>/ directory, or None."""
    for name in SPEC_FILENAMES:
        candidate = Path(subject_dir) / name
        if candidate.is_file():
            return candidate
    return None
def is_native(spec) -> bool:
    return Path(spec).name == "espnet_model.json"
def _load_native(spec, device):
    """Build a wrapper from an ESPnet train config + weights + its own bpemodel."""
    key = f"{Path(spec).resolve()}|{device}"
    if key in _NATIVE_CACHE:
        return _NATIVE_CACHE[key]

    from espnet2.bin.asr_inference import Speech2Text

    from src.models.espnet import ESPnetASRWrapper

    meta = json.loads(Path(spec).read_text())
    for field in ("asr_train_config", "asr_model_file", "bpemodel"):
        if field not in meta:
            raise ValueError(f"{spec}: missing '{field}'")
        if not Path(meta[field]).is_file():
            raise FileNotFoundError(f"{spec}: {field} -> {meta[field]} not found")

    speech2text = Speech2Text(asr_train_config=meta["asr_train_config"],
                              asr_model_file=meta["asr_model_file"],
                              bpemodel=meta["bpemodel"], device=device)
    model = ESPnetASRWrapper(speech2text)
    model.model.eval()
    # One instance per spec: a retrain tree points every subject at the same model, and...
    _NATIVE_CACHE[key] = model
    return model
def _load_lightning(spec, model_tag, device, base_model=None):
    """Apply a Lightning checkpoint's weights to the pretrained wrapper."""
    from src.models.espnet import ESPnetASRWrapper

    model = base_model if base_model is not None else ESPnetASRWrapper.from_pretrained(model_tag)
    ckpt = torch.load(spec, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {spec} has no 'state_dict' key")

    filtered = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("model."):
            continue
        new_key = key[len("model."):]
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        filtered[new_key] = value
    if not filtered:
        raise ValueError(f"Checkpoint {spec} had no 'model.'-prefixed weights")

    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing keys loading {spec}: {len(incompatible.missing_keys)} "
                           f"(first few: {incompatible.missing_keys[:5]})")
    del ckpt, filtered
    return model
def load_attack_model(spec, model_tag, device, base_model=None):
    """Return an eval-mode wrapper for the model at `spec`, on `device`."""
    model = (_load_native(spec, device) if is_native(spec)
             else _load_lightning(spec, model_tag, device, base_model))
    model.to(device)
    model.eval()
    return model


def match_to_reference_distribution(df_target, df_reference, n_bins=20, seed=42,
                                    dur_col=("dur_sec", "duration"), label=""):
    """Resample df_target so its duration histogram matches df_reference's."""
    rng = np.random.default_rng(seed)

    for col in dur_col:
        if col in df_reference.columns and col in df_target.columns:
            resolved = col
            break
    else:
        raise ValueError(f"No duration column from {dur_col} present in both frames")

    ref_vals = df_reference[resolved].to_numpy()
    edges = np.unique(np.percentile(ref_vals, np.linspace(0, 100, n_bins + 1)))
    edges[-1] += 1e-9  # make the maximum inclusive

    ref_counts = np.array([((ref_vals >= lo) & (ref_vals < hi)).sum()
                           for lo, hi in zip(edges[:-1], edges[1:])], dtype=float)
    ref_props = ref_counts / ref_counts.sum()

    target_bins = [df_target[(df_target[resolved] >= lo) & (df_target[resolved] < hi)]
                   for lo, hi in zip(edges[:-1], edges[1:])]
    avail = np.array([len(tb) for tb in target_bins])

    covered = avail > 0
    dropped = float(ref_props[~covered].sum())
    props = np.where(covered, ref_props, 0.0)
    if props.sum() == 0:
        raise ValueError("No overlap between target and reference duration ranges")
    props = props / props.sum()
    if dropped > 1e-9:
        log.warning("    %s: %.1f%% of the reference duration mass has no target "
                    "utterances; renormalized over the covered bins", label, 100 * dropped)

    total_kept = int(min(avail[i] / props[i] for i in range(len(props)) if props[i] > 0))

    kept = []
    for tb, p in zip(target_bins, props):
        n = min(round(total_kept * p), len(tb)) if p > 0 else 0
        if n:
            kept.append(tb.sample(n=n, random_state=int(rng.integers(1_000_000))))
    return pd.concat(kept, ignore_index=True) if kept else df_target.iloc[0:0]
def duration_stats(df, col="duration"):
    if len(df) == 0:
        return "n=0"
    return (f"n={len(df):>5d}  mean={df[col].mean():5.2f}s  "
            f"[{df[col].min():5.2f}, {df[col].max():5.2f}]")
def duration_report(df, col="duration", n_bins=10):
    """avg/min/max/std plus a histogram of `df[col]`, as a JSON-serializable dict."""
    if len(df) == 0:
        return {"n": 0}
    vals = df[col].to_numpy()
    counts, edges = np.histogram(vals, bins=n_bins)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "std": float(vals.std()),
        "histogram": {"edges": [float(e) for e in edges],
                      "counts": [int(c) for c in counts]},
    }
def match_per_corpus(df, forget_df, n_bins, seed, label):
    """Duration-match each LibriSpeech corpus in `df` to the forget speaker separately."""
    if "corpus" not in df.columns:
        return match_to_reference_distribution(df, forget_df, n_bins=n_bins,
                                               seed=seed, label=label)
    parts = []
    for corpus, sub in df.groupby("corpus", sort=True):
        matched = match_to_reference_distribution(sub, forget_df, n_bins=n_bins,
                                                  seed=seed, label=f"{label}/{corpus}")
        log.info("    %-12s %s  ->  %s", corpus, duration_stats(sub),
                 duration_stats(matched))
        parts.append(matched)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
def align_mia_sets_to_forget(csv_sets, out_dir, n_bins=20, seed=42, per_corpus=True):
    """Align every non-forget split onto the forgotten speaker's duration histogram, then rebuild the..."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forget_df = pd.read_csv(csv_sets["forget_eval_csv"])
    log.info("  reference (forget_eval): %s", duration_stats(forget_df))

    report = {"forget_eval": duration_report(forget_df)}
    matched_paths = {}
    for key in ("retain_train_csv", "test_train_csv", "test_eval_csv"):
        df = pd.read_csv(csv_sets[key])
        if per_corpus and key.startswith("test_"):
            matched = match_per_corpus(df, forget_df, n_bins, seed, key)
        else:
            matched = match_to_reference_distribution(df, forget_df, n_bins=n_bins,
                                                      seed=seed, label=key)
        log.info("  %-18s %s  ->  %s", key, duration_stats(df), duration_stats(matched))
        path = out_dir / f"{key.replace('_csv', '')}_matched.csv"
        matched.to_csv(path, index=False)
        matched_paths[key] = str(path)

        split_name = key.replace("_csv", "")
        report[split_name] = duration_report(matched)
        if per_corpus and key.startswith("test_") and "corpus" in matched.columns:
            report[f"{split_name}_by_corpus"] = {
                str(corpus): duration_report(sub)
                for corpus, sub in matched.groupby("corpus", sort=True)
            }

    report_path = out_dir / "duration_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("  duration-match report: %s", report_path)

    return {**csv_sets, **matched_paths, "forget_eval_csv": csv_sets["forget_eval_csv"]}
def subject_ids(df):
    if "speaker_id" not in df.columns:
        return []
    return sorted(set(df["speaker_id"].astype(str)))
def compute_split_counts(retain_train, test_train, forget_eval, test_eval):
    """Utterance and unique-subject counts for each of the four MIA splits."""
    subjects = {
        "train_members": subject_ids(retain_train),
        "train_nonmembers": subject_ids(test_train),
        "eval_members": subject_ids(forget_eval),
        "eval_nonmembers": subject_ids(test_eval),
    }
    return {
        "n_train_members": int(len(retain_train)),
        "n_train_nonmembers": int(len(test_train)),
        "n_eval_members": int(len(forget_eval)),
        "n_eval_nonmembers": int(len(test_eval)),
        "n_train_member_subjects": len(subjects["train_members"]),
        "n_train_nonmember_subjects": len(subjects["train_nonmembers"]),
        "n_eval_member_subjects": len(subjects["eval_members"]),
        "n_eval_nonmember_subjects": len(subjects["eval_nonmembers"]),
        "subject_ids": subjects,
    }
def build_train_eval_csvs(aligned, out_dir, class_balance, seed):
    """Concatenate the aligned pieces into the attacker's train/eval CSVs."""
    out_dir = Path(out_dir)
    retain_train = pd.read_csv(aligned["retain_train_csv"])
    test_train = pd.read_csv(aligned["test_train_csv"])
    forget_eval = pd.read_csv(aligned["forget_eval_csv"])
    test_eval = pd.read_csv(aligned["test_eval_csv"])

    if class_balance == "downsample" and len(retain_train) and len(test_train):
        n = min(len(retain_train), len(test_train))
        if len(retain_train) != len(test_train):
            log.info("  balancing train by downsampling to %d per class "
                     "(members %d, non-members %d)", n, len(retain_train), len(test_train))
        retain_train = retain_train.sample(n=n, random_state=seed)
        test_train = test_train.sample(n=n, random_state=seed)

    train_csv = out_dir / "mia_train_matched.csv"
    eval_csv = out_dir / "mia_eval_matched.csv"
    pd.concat([retain_train, test_train], ignore_index=True) \
      .sample(frac=1, random_state=seed).to_csv(train_csv, index=False)
    pd.concat([forget_eval, test_eval], ignore_index=True) \
      .sample(frac=1, random_state=seed).to_csv(eval_csv, index=False)

    counts = compute_split_counts(retain_train, test_train, forget_eval, test_eval)
    return str(train_csv), str(eval_csv), counts
def evaluate_by_corpus(attacker, split, override, eval_csv):
    """Score `split` once and report metrics overall and per non-member corpus."""
    saved_label = attacker.label
    attacker.label = ["utt_in_set", "ID"]
    try:
        X, labels = attacker.extract_features(split, override=override)
    finally:
        attacker.label = saved_label

    y = np.asarray(labels["utt_in_set"]).astype(int)
    ids = [str(v) for v in labels["ID"]]
    probs = attacker.classifier.predict_proba(X)[:, 1]
    preds = attacker.classifier.predict(X)

    from src.membership_inference.utils import compute_binary_mia_metrics
    out = {"overall": compute_binary_mia_metrics(y, probs, preds)}

    eval_df = pd.read_csv(eval_csv)
    if "corpus" not in eval_df.columns:
        return out
    corpus_of = dict(zip(eval_df["ID"].astype(str), eval_df["corpus"].astype(str)))
    missing = sum(1 for i in ids if i not in corpus_of)
    if missing:
        log.warning("    %d/%d eval rows have no corpus label; skipping breakdown",
                    missing, len(ids))
        return out
    corpus = np.array([corpus_of[i] for i in ids])

    member = y == 1
    for name in sorted(set(corpus[~member])):
        sel = member | (corpus == name)
        if sel.sum() == member.sum():           # no non-members of this corpus
            continue
        out[name] = compute_binary_mia_metrics(y[sel], probs[sel], preds[sel])
        out[name]["n_nonmembers"] = int((~member & (corpus == name)).sum())
    out["overall"]["n_nonmembers"] = int((~member).sum())
    out["overall"]["n_members"] = int(member.sum())
    return out
def run_attacks(model, train_csv, eval_csv, cfg, temp_dir, levels, seed,
                class_balance, share_features):
    """Run the requested attack levels against the matched CSVs."""
    from src.evaluation.evaluate_mia import _build_classifier, _build_feature_extractor
    from src.membership_inference.modules.abs_attacker import Attacker
    from src.membership_inference.modules.speaker_level_attacker import SpeakerLevelAttacker

    datasets = {"train": train_csv, "dev": eval_csv, "test": eval_csv}
    results, extractor = {}, None

    for i, level in enumerate(levels):
        name = ("loss_extractor_forget_shared" if share_features
                else f"loss_extractor_forget_{level}")
        if extractor is None or not share_features:
            extractor = _build_feature_extractor(
                model=model, cfg=cfg,
                feature_dir=os.path.join(temp_dir, "features"), name=name)

        classifier = _build_classifier(cfg=cfg, seed=seed)
        if class_balance == "class-weight":
            classifier.set_params(class_weight="balanced")

        attacker_cfg = {
            "datasets": datasets,
            "classifier": classifier,
            "feature_extractors": [extractor],
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
        }
        if level == "utterance":
            attacker_cfg.update({"label": "utt_in_set", "class_label": "utt_in_set"})
            attacker = Attacker(attacker_cfg)
            eval_split = "test"
        else:
            attacker_cfg.update({"label": ["utt_in_set", "spk_in_set", "speaker_id"],
                                 "class_label": "spk_in_set"})
            attacker = SpeakerLevelAttacker(cfg=attacker_cfg)
            eval_split = cfg.eval_split

        # Only the first level may recompute features; a shared cache is reused after.
        override = bool(cfg.override) and (i == 0 or not share_features)
        log.info("  [%s] training classifier (override=%s)...", level, override)
        attacker.train_classifier(override=override)
        log.info("  [%s] evaluating on split=%s...", level, eval_split)
        if level == "utterance":
            by_corpus = evaluate_by_corpus(attacker, eval_split, override, eval_csv)
            results[level] = by_corpus["overall"]
            for name, m in by_corpus.items():
                if name != "overall":
                    results[f"{level}:{name}"] = m
        else:
            results[level] = attacker.evaluate(split=eval_split, override=override)
        log.info("  [%s] %s", level,
                 "  ".join(f"{k}={v:.4f}" for k, v in results[level].items()
                           if isinstance(v, (int, float))))
    return results
def discover_checkpoints(root, methods, subjects):
    """(method, subject, spec) for every attackable <root>/<method>/<subject>/."""
    found = []
    for method_dir in sorted(p for p in root.iterdir()
                             if p.is_dir() and not p.name.startswith(("_", "."))):
        if methods and method_dir.name not in methods:
            continue
        for subj_dir in sorted(method_dir.iterdir(),
                               key=lambda p: (not p.name.isdigit(), p.name)):
            if not subj_dir.is_dir() or (subjects and subj_dir.name not in subjects):
                continue
            spec = find_model_spec(subj_dir)
            if spec is not None:
                found.append((method_dir.name, subj_dir.name, spec))
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
def load_unlearned_model(model_tag, spec, device):
    """Eval-mode wrapper for the artifact at `spec` (Lightning ckpt or ESPnet-native)."""
    return load_attack_model(spec, model_tag, device)
def build_cfg(args):
    return OmegaConf.create({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "override": not args.reuse_features,
        "cache_save_frequency": 1,
        "mia_train_split_ratio": args.mia_train_split_ratio,
        "loss_att": True,
        "loss_ctc": True,
        "loss_cer": args.loss_cer,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "eval_split": "test",
        # Left null on purpose: a shared cache_dir would pool feature caches across checkpoints...
        "cache_dir": None,
        "random_state": args.seed,
    })
def write_summary(out_dir):
    """Rebuild the summary CSV from every results.json currently on disk."""
    rows = []
    for path in sorted(out_dir.glob("*/*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        base = {k: payload.get(k) for k in
                ("method", "forget_subject", "checkpoint", "status", "class_balance",
                 "match_durations", "nonmember_split", "per_corpus_match", "runtime_sec")}
        base.update({k: payload.get("splits", {}).get(k) for k in SPLIT_COLUMNS})
        for level, metrics in (payload.get("metrics") or {}).items():
            row = dict(base, level=level)
            row.update({k: (metrics or {}).get(k) for k in METRIC_COLUMNS})
            rows.append(row)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["level", "method", "forget_subject"])
    summary = out_dir / "simple_mia_summary.csv"
    df.to_csv(summary, index=False)
    return summary, len(rows)
def write_duration_summary(out_dir):
    """Rebuild the duration-match summary CSV from every duration_report.json on disk."""
    rows = []
    for path in sorted(out_dir.glob("*/*/mia_temp/duration_report.json")):
        method, subject = path.parts[-4], path.parts[-3]
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        def add_row(split, stats):
            if stats.get("n", 0) == 0:
                return
            rows.append({"method": method, "forget_subject": subject, "split": split,
                        "n": stats["n"], "mean_dur": stats["mean"],
                        "min_dur": stats["min"], "max_dur": stats["max"],
                        "std_dur": stats["std"]})

        for split, stats in payload.items():
            if split.endswith("_by_corpus"):
                continue
            add_row(split, stats)
            for corpus, cstats in payload.get(f"{split}_by_corpus", {}).items():
                add_row(f"{split}/{corpus}", cstats)

    if not rows:
        return None, 0
    df = pd.DataFrame(rows).sort_values(["method", "forget_subject", "split"])
    summary = out_dir / "duration_match_summary.csv"
    df.to_csv(summary, index=False)
    return summary, len(rows)
