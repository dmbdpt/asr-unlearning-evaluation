"""Aggregate the new re-evaluation results (under results/run_2026*/reeval_*/) into a single CSV..."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pandas import json_normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
# Where completed runs live.
RUNS_ROOT = Path(os.environ.get("LEAF_RUNS_ROOT", REPO_ROOT))
# Active reeval dirs: run_<ts>/reeval_<ts2>/subject_<id>/mlflow.db <ts2> is...
DEFAULT_GLOBS = [
    str(RUNS_ROOT / "reeval_*" / "subject_*" / "mlflow.db"),
    str(RUNS_ROOT / "eval_backup_*" / "reeval_*_subject_*" / "mlflow.db"),
]
DEFAULT_OUT = RUNS_ROOT / "results" / "all_reeval_merged.csv"

# Per-subject results JSONs key the "test" split by speaker_id only — no test-clean /...
LS_TEST_MANIFEST = Path(
    os.environ.get("LEAF_DATA_DIR", str(REPO_ROOT / "data"))
) / "ls_test_all.csv"


def parse_db_path(db: Path) -> tuple[Path, str, str]:
    """Return (run_folder_path, subject, inner_reeval_ts) for a reeval DB."""
    parts = db.parts
    leaf = parts[-2]               # 'subject_<id>'  OR  'reeval_<ts>_subject_<id>'
    if "_subject_" in leaf:
        # Backup layout: reeval_<ts>_subject_<id>
        inner_ts, subject = leaf.split("_subject_", 1)
    else:
        # Active layout: subject_<id>; inner ts is the parent dir
        subject = leaf.removeprefix("subject_")
        inner_ts = parts[-3]       # 'reeval_<ts>'
    run_folder_path = Path(*parts[: parts.index(parts[-4]) + 1])  # absolute path to run_<ts>
    return run_folder_path, subject, inner_ts


def reeval_sort_key(inner_ts: str) -> tuple[str, int]:
    """Newest-first sort key for a 'reeval_<YYYYmmdd_HHMMSS>[_<pid>]' dir name."""
    rest = inner_ts.removeprefix("reeval_")
    date, _, tail = rest.partition("_")
    time_part, _, pid = tail.partition("_")
    return f"{date}_{time_part}", int(pid) if pid.isdigit() else -1


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("reeval_to_csv")


def load_speaker_subset_map(path: Path = LS_TEST_MANIFEST) -> dict[str, str]:
    """speaker_id (str) -> 'testclean' | 'testother', from the LibriSpeech test manifest."""
    try:
        manifest = pd.read_csv(path, usecols=["speaker_id", "split"])
    except (OSError, ValueError) as e:
        log.warning("Could not load LibriSpeech test manifest %s: %s — "
                    "testclean/testother columns will be empty", path, e)
        return {}
    mapping: dict[str, str] = {}
    for spk, split in manifest[["speaker_id", "split"]].drop_duplicates().itertuples(index=False):
        if split == "test-clean":
            mapping[str(spk)] = "testclean"
        elif split == "test-other":
            mapping[str(spk)] = "testother"
    return mapping


SPEAKER_SUBSET = load_speaker_subset_map()


# ---------------------------------------------------------------------------
# MLflow DB helpers

def read_reeval_db(db_path: Path) -> tuple[str, str, dict] | None:
    """Return (run_uuid, artifact_uri, params) for the single FINISHED run, or None."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT run_uuid, artifact_uri, status FROM runs "
            "WHERE status = 'FINISHED' ORDER BY start_time DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        run_uuid, artifact_uri, _status = row
        cur.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_uuid,))
        params = {f"param_{k}": v for k, v in cur.fetchall()}
        cur.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_uuid,))
        tags = dict(cur.fetchall())
        conn.close()
        return run_uuid, artifact_uri, params, tags
    except Exception as e:
        log.warning("Could not read DB %s: %s", db_path, e)
        return None


# ---------------------------------------------------------------------------
# Artifact path helpers

def find_consolidated_artifact(artifact_uri: str, subject: str) -> Path | None:
    pattern = os.path.join(artifact_uri, "reeval", f"subject_{subject}_results_*.json")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def find_pre_artifact(artifact_uri: str, subject: str) -> Path | None:
    pattern = os.path.join(artifact_uri, "reeval", f"subject_{subject}_pre_results_*.json")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def _is_run_uuid(part: str) -> bool:
    """True for a 32-hex-char MLflow run id."""
    return len(part) == 32 and all(c in "0123456789abcdefABCDEF" for c in part)


def _run_uuid_from_checkpoint(checkpoint_path: str) -> str | None:
    """Extract the MLflow run UUID from a Lightning checkpoint path."""
    parts = Path(checkpoint_path).parts
    if "mlruns" in parts:
        i = parts.index("mlruns")
        if i + 2 < len(parts) and _is_run_uuid(parts[i + 2]):
            return parts[i + 2]
    for part in parts:
        if _is_run_uuid(part):
            return part
    return None


def find_original_artifacts(run_folder_path: Path, run_uuid: str) -> dict[str, Path | None]:
    """Return paths to the original experiment run's eval artifacts."""
    out: dict[str, Path | None] = {
        "post_results": None, "pre_results": None,
        "post_mia":     None, "pre_mia":     None,
    }
    base_glob = str(run_folder_path / "mlruns" / "*" / run_uuid / "artifacts" / "evaluation")
    eval_dirs = glob.glob(base_glob)
    if not eval_dirs:
        return out
    eval_dir = eval_dirs[0]

    for key, pattern in (
        ("post_results", "subject_*_post_unlearn_results_*.json"),
        ("pre_results",  "subject_*_pre_unlearn_results_*.json"),
        ("post_mia",     "subject_*_mia_post_*.json"),
        ("pre_mia",      "subject_*_mia_pre_*.json"),
    ):
        matches = sorted(glob.glob(os.path.join(eval_dir, pattern)))
        if matches:
            out[key] = Path(matches[-1])
    return out


def load_json_safe(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not load %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Merge helpers — decide whether a section has useful data

def _has_mia_data(d: object) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    for v in d.values():
        if isinstance(v, dict) and v:
            return True
        if isinstance(v, (int, float)):
            return True
    return False


def _has_split_data(d: object) -> bool:
    """True if test/train dict has at least one speaker with metrics or a loss value."""
    if not isinstance(d, dict) or not d:
        return False
    for speaker_data in d.values():
        if not isinstance(speaker_data, dict):
            continue
        metrics = speaker_data.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            return True
        if speaker_data.get("losses") is not None:
            return True
    return False


def _has_emd_data(d: object) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    return any(v is not None for v in d.values())


_SECTION_CHECKS = {
    "mia":   _has_mia_data,
    "emd":   _has_emd_data,
    "test":  _has_split_data,
    "train": _has_split_data,
}


def merge_results(*sources: dict | None) -> dict:
    """For each top-level section, use the first source (newest) that has data."""
    merged: dict = {}
    for key, has_data in _SECTION_CHECKS.items():
        for source in sources:
            if source is None:
                continue
            val = source.get(key)
            if has_data(val):
                merged[key] = val
                log.debug("  key=%s: using source index %d", key, sources.index(source))
                break
    return merged


# ---------------------------------------------------------------------------
# Original run fallback

# ---------------------------------------------------------------------------
# CSV row builders

def flatten_payload(payload: dict, prefix: str) -> dict:
    flat = json_normalize(payload, sep=".").to_dict(orient="records")[0]
    return {
        f"{prefix}_{k}": v
        for k, v in flat.items()
        if "transcript" not in k
        and "ground_truths" not in k
        and "speaker_ids" not in k
    }


def add_split_aggregates(row: dict, subject: str, data: dict, stage: str) -> None:
    """Mutates row in place with {test,retain,forget}_{stage}_*, {stage}_emd_*, and..."""
    emd = data.get("emd", {})
    if isinstance(emd, dict):
        for k, v in emd.items():
            row[f"{stage}_{k}"] = v

    raw_rows: list[dict] = []
    for split in ("test", "train"):
        split_data = data.get(split, {})
        if not isinstance(split_data, dict):
            continue
        for eval_subject, payload in split_data.items():
            if not isinstance(payload, dict):
                continue
            flat = flatten_payload(payload, prefix=stage)
            flat["split"] = split
            flat["eval_subject"] = str(eval_subject)
            if split == "test":
                flat["subset"] = SPEAKER_SUBSET.get(str(eval_subject))
            raw_rows.append(flat)

    if raw_rows:
        df = pd.DataFrame(raw_rows)
        non_metric_cols = ("split", "eval_subject", "subset")
        # Some artifact schemas carry a raw per-utterance breakdown (e.g. "per_utt_wer")
        # alongside the scalar aggregate; those are lists, not means-of, so skip them.
        metric_cols = [c for c in df.columns
                       if c.startswith(f"{stage}_") and c not in non_metric_cols
                       and not df[c].map(lambda v: isinstance(v, list)).any()]
        test_grp = df[df["split"] == "test"]
        train_grp = df[(df["split"] == "train") & (df["eval_subject"] != subject)]
        forget_grp = df[(df["split"] == "train") & (df["eval_subject"] == subject)]
        for c in metric_cols:
            row[f"test_{c}"]   = test_grp[c].mean()   if not test_grp.empty   else float("nan")
            row[f"retain_{c}"] = train_grp[c].mean()  if not train_grp.empty  else float("nan")
            row[f"forget_{c}"] = forget_grp[c].mean() if not forget_grp.empty else float("nan")

        # WER/CER/loss broken out per LibriSpeech test subset (test-clean / test-other).
        split_metric_cols = [
            c for c in metric_cols
            if c.endswith(".wer") or c.endswith(".cer") or c.endswith("_losses")
        ]
        for subset_name in ("testclean", "testother"):
            subset_grp = test_grp[test_grp["subset"] == subset_name]
            for c in split_metric_cols:
                row[f"{subset_name}_{c}"] = (
                    subset_grp[c].mean() if not subset_grp.empty else float("nan")
                )

    mia = data.get("mia", {})
    if isinstance(mia, dict):
        for level, metrics in mia.items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                row[f"{stage}_{level}_{k}"] = v


def aggregate_subject(
    run_folder: str,
    subject: str,
    params: dict,
    post_data: dict,
    pre_data: dict | None,
) -> dict:
    row: dict = {"run_folder": run_folder, "forget_subject": subject}
    row.update(params)
    add_split_aggregates(row, subject, post_data, stage="post")
    if pre_data is not None:
        add_split_aggregates(row, subject, pre_data, stage="pre")
    return row


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glob", action="append", default=None,
                        help="Glob(s) matching reeval mlflow.dbs (may be passed multiple times)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output CSV path")
    args = parser.parse_args()

    patterns = args.glob if args.glob else DEFAULT_GLOBS
    db_files: list[Path] = []
    for pat in patterns:
        db_files.extend(Path(p) for p in glob.glob(pat))
    db_files = sorted(set(db_files))
    log.info("Found %d reeval DBs (active + backups) across %d pattern(s)",
             len(db_files), len(patterns))

    # Group by (run_folder, subject, source_checkpoint_uuid).
    Key = tuple[Path, str, str]
    grouped: dict[Key, list[tuple[str, Path, tuple]]] = defaultdict(list)
    unreadable = 0
    for db in db_files:
        run_folder_path, subject, inner_ts = parse_db_path(db)
        meta = read_reeval_db(db)
        if meta is None:
            unreadable += 1
            continue
        _uuid, _uri, _params, tags = meta
        src_uuid = _run_uuid_from_checkpoint(tags.get("source_checkpoint", ""))
        if src_uuid is None:
            # No MLflow run behind this artifact.
            src_uuid = tags.get("source_checkpoint") or f"unknown:{db}"
            if src_uuid.startswith("unknown:"):
                log.warning("No source checkpoint for %s — grouping it alone", db)
        grouped[(run_folder_path, subject, src_uuid)].append((inner_ts, db, meta))

    groups: dict[Key, list[tuple[Path, tuple]]] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda x: reeval_sort_key(x[0]), reverse=True)
        groups[key] = [(db, meta) for _, db, meta in entries]

    if unreadable:
        log.warning("%d DB(s) had no FINISHED run and were skipped", unreadable)
    log.info("Grouped into %d (run_folder, subject, source_checkpoint) experiments",
             len(groups))

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for (run_folder_path, subject, src_uuid), dbs in sorted(groups.items()):
        log.info("Processing %s / subject %s / checkpoint %s (%d reeval(s))",
                 run_folder_path.name, subject, src_uuid[:12], len(dbs))

        # --- Collect params, artifacts, and original-run UUID from all reevals ---
        params: dict = {}
        post_sources: list[dict] = []
        pre_sources: list[dict] = []
        original_post: dict | None = None
        original_pre: dict | None = None
        seen_run_uuids: set[str] = set()

        for db, meta in dbs:
            _, artifact_uri, db_params, tags = meta

            if not params:
                params = db_params

            post_art = find_consolidated_artifact(artifact_uri, subject)
            if post_art is not None:
                d = load_json_safe(post_art)
                if d is not None:
                    post_sources.append(d)
                    log.debug("  post: loaded %s", post_art)
            else:
                log.debug("  post: no reeval artifact in %s", artifact_uri)

            pre_art = find_pre_artifact(artifact_uri, subject)
            if pre_art is not None:
                d = load_json_safe(pre_art)
                if d is not None:
                    pre_sources.append(d)
                    log.debug("  pre:  loaded %s", pre_art)

            # Derive original run UUID from checkpoint path stored in tags
            ckpt_path = tags.get("source_checkpoint", "")
            orig_uuid = _run_uuid_from_checkpoint(ckpt_path)
            if orig_uuid and orig_uuid not in seen_run_uuids:
                seen_run_uuids.add(orig_uuid)
                orig = find_original_artifacts(run_folder_path, orig_uuid)
                if orig["post_results"] is not None and original_post is None:
                    original_post = load_json_safe(orig["post_results"])
                    if original_post is not None:
                        # Original logs MIA in a sibling file; graft it on so merge_results sees it.
                        if orig["post_mia"] is not None:
                            mia = load_json_safe(orig["post_mia"])
                            if mia is not None:
                                original_post["mia"] = mia
                        log.info("  original post fallback: %s", orig["post_results"])
                if orig["pre_results"] is not None and original_pre is None:
                    original_pre = load_json_safe(orig["pre_results"])
                    if original_pre is not None:
                        if orig["pre_mia"] is not None:
                            mia = load_json_safe(orig["pre_mia"])
                            if mia is not None:
                                original_pre["mia"] = mia
                        log.info("  original pre  fallback: %s", orig["pre_results"])

        # --- Merge: prefer newest source for each section ---
        merged_post = merge_results(*post_sources, original_post)
        merged_pre  = merge_results(*pre_sources, original_pre)

        if not merged_post and not merged_pre:
            skipped.append((str(run_folder_path), subject))
            log.warning("  SKIP: no data found for %s / subject %s",
                        run_folder_path.name, subject)
            continue

        for section, fn in _SECTION_CHECKS.items():
            sources_with_data = [
                i for i, s in enumerate((*post_sources, original_post))
                if s is not None and fn(s.get(section))
            ]
            if sources_with_data:
                src_label = (
                    f"reeval[{sources_with_data[0]}]"
                    if sources_with_data[0] < len(post_sources)
                    else "original"
                )
                log.info("  %-6s section: %s", section, src_label)
            else:
                log.info("  %-6s section: MISSING", section)

        row = aggregate_subject(
            run_folder_path.name,
            subject,
            params,
            merged_post,
            merged_pre or None,
        )
        row["source_run_uuid"] = src_uuid
        # The mlflow params (e.g.
        db_paths = [str(db) for db, _ in dbs]
        row["reeval_db_paths"] = ";".join(db_paths)
        row["has_backup_source"] = any("eval_backup_" in p for p in db_paths)
        rows.append(row)

    if not rows:
        log.error("No rows aggregated.")
        return 1

    df = pd.DataFrame(rows)
    id_cols = ["run_folder", "forget_subject", "source_run_uuid"]
    param_cols = sorted(c for c in df.columns if c.startswith("param_"))
    other_cols = sorted(c for c in df.columns if c not in id_cols and not c.startswith("param_"))
    df = df[id_cols + param_cols + other_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %s: %d rows, %d columns", out_path, len(df), len(df.columns))

    if skipped:
        log.warning("Skipped %d entries:", len(skipped))
        for path, subj in skipped:
            log.warning("  %s / subject %s", path, subj)

    return 0


if __name__ == "__main__":
    sys.exit(main())
