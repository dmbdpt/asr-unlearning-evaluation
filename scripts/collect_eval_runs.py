"""Discover every completed unlearning run and emit the list of (checkpoint, forgotten subject)..."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_OUT_JSON = SCRIPTS_DIR / "eval_runs.json"
DEFAULT_REMAP_LOG = SCRIPTS_DIR / "path_remappings.log"
DEFAULT_SKIPPED_LOG = SCRIPTS_DIR / "skipped_runs.log"
UNLEARNER = ""

# Anything absolute pointing at an old /tmp/leaf_run_XXXX path in a saved config gets...
OLD_TMP_PREFIX = "/tmp/leaf_run_"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_eval_runs")


@dataclass
class NestedRun:
    run_folder: str
    subject: str
    subject_source: str  # "mlflow_tag" | "inferred_from_run_name" | "inferred_from_folder"
    mlflow_run_uuid: Optional[str]
    mlflow_experiment_id: Optional[str]
    unlearner: Optional[str]
    checkpoint_path: str
    checkpoint_source: str  # "1/<uuid>/checkpoints" | "mlruns/<exp>/<uuid>/artifacts/last"
    overrides: list = field(default_factory=list)
    existing_eval_dirs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers


# Never descend into these: VCS/venv/cache noise, plus MLflow's own (huge) artifact tree...
PRUNE_DIR_NAMES = {"mlruns", ".venv", ".git", "__pycache__", "node_modules", "hub"}


def find_run_folders(root: Path) -> list[Path]:
    """Walk `root` and return every directory that directly contains an MLflow sqlite db..."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in PRUNE_DIR_NAMES and not d.startswith(".")
        ]
        if "mlflow.db" in filenames:
            found.append(Path(dirpath))
            dirnames[:] = []  # a run folder doesn't nest another run folder
    return sorted(found)


def load_run_params(conn: sqlite3.Connection, run_uuid: str) -> dict[str, str]:
    """Flattened dotted-key config params MLflow logged for this run — this is the full config..."""
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_uuid,))
    return dict(cur.fetchall())


def diagnostic_prefixes(params: dict[str, str]) -> set[str]:
    """Namespaces holding unlearner telemetry rather than config, keyed by a .class param."""
    return {
        key.rsplit(".", 1)[0]
        for key in params
        if key.endswith(".class") and "." in key
    }


_SAFE_BARE_OVERRIDE_RE = re.compile(r"^[A-Za-z0-9_.\-/+@:]+$")


def hydra_override_value(v: str) -> str:
    """Render an MLflow-stringified param value as a Hydra override value."""
    if v == "None":
        return "null"
    if v.startswith("[") or v.startswith("{"):
        return v
    if _SAFE_BARE_OVERRIDE_RE.match(v):
        return v
    escaped = v.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def find_mlflow_db(run_folder: Path) -> Optional[Path]:
    """Return the first MLflow sqlite db present in the run folder."""
    candidates = [run_folder / "mlflow.db"]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1024:
            return c
    return None


def remap_absolute_path(value: str, run_folder: Path, remap_log) -> str:
    """If value is an absolute path that no longer exists, attempt to remap it underneath run_folder."""
    if not isinstance(value, str) or not value.startswith(OLD_TMP_PREFIX):
        return value
    if os.path.exists(value):
        return value
    # Strip the /tmp/leaf_run_<XXXX>/ prefix and re-root under the run folder.
    parts = value.split("/", 3)  # ['', 'tmp', 'leaf_run_XXXX', 'rest']
    if len(parts) < 4:
        return value
    remapped = str(run_folder / parts[3])
    if os.path.exists(remapped):
        remap_log.write(f"{run_folder.name}\t{value}\t->\t{remapped}\n")
        return remapped
    return value


LOCAL_CSV_SUFFIX = "_local.csv"


def delocalize_csv_path(value: str, run_folder: Path, remap_log) -> str:
    """train_csv/test_csv values ending in `_local.csv` reference audio manifests built against a..."""
    if not isinstance(value, str) or not value.endswith(LOCAL_CSV_SUFFIX):
        return value
    candidate = value[: -len(LOCAL_CSV_SUFFIX)] + ".csv"
    if os.path.exists(candidate):
        remap_log.write(f"{run_folder.name}\t{value}\t->\t{candidate}\n")
        return candidate
    return value


def find_checkpoint(run_folder: Path, run_uuid: str, exp_id: Optional[str]) -> Optional[tuple[str, str]]:
    """Return (path, source) for the unlearned checkpoint, or None."""
    # Preferred: Lightning's own checkpoint location.
    p1 = run_folder / "1" / run_uuid / "checkpoints" / "last.ckpt"
    if p1.exists():
        return str(p1), "lightning"
    # Fallback: MLflow-logged copy.
    if exp_id is not None:
        p2 = run_folder / "mlruns" / exp_id / run_uuid / "artifacts" / "last" / "last.ckpt"
        if p2.exists():
            return str(p2), "mlflow_artifact"
    # Last-ditch: scan recursively.
    matches = list(run_folder.glob(f"**/{run_uuid}/**/last.ckpt"))
    if matches:
        return str(matches[0]), "scan"
    return None


def find_existing_eval_dirs(run_folder: Path, run_uuid: str) -> list[str]:
    """Anything that already looks like an eval artifact for this nested run."""
    out: list[str] = []
    for sub in ("artifacts/evaluation", "artifacts/comparison"):
        p = run_folder / "mlruns" / "*" / run_uuid / sub
        out.extend(str(x) for x in run_folder.glob(f"mlruns/*/{run_uuid}/{sub}"))
    return out


# ---------------------------------------------------------------------------
# Core walker


def collect_for_run(
    run_folder: Path, remap_log, skipped_log
) -> list[NestedRun]:
    found: list[NestedRun] = []

    db_path = find_mlflow_db(run_folder)
    if db_path is None:
        skipped_log.write(f"{run_folder.name}\twhole_run\tno_mlflow_db\n")
        log.warning("%s: no mlflow DB found", run_folder.name)
        return found

    # Query SQLite for nested runs (FINISHED + has subject tag).
    nested_meta: list[tuple[str, str, str]] = []  # (run_uuid, subject, exp_id)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.run_uuid, t_subj.value AS subject, r.experiment_id
            FROM runs r
            JOIN tags t_subj ON t_subj.run_uuid = r.run_uuid AND t_subj.key = 'subject'
            WHERE r.status = 'FINISHED'
            """
        )
        nested_meta = [(uid, subj, str(exp)) for (uid, subj, exp) in cur.fetchall()]

        if not nested_meta:
            skipped_log.write(f"{run_folder.name}\twhole_run\tno_finished_subject_runs\n")
            return found

        for run_uuid, subject, exp_id in nested_meta:
            ckpt = find_checkpoint(run_folder, run_uuid, exp_id)
            if ckpt is None:
                skipped_log.write(f"{run_folder.name}\t{run_uuid}\tcheckpoint not found\n")
                log.warning("%s/%s: checkpoint not found", run_folder.name, run_uuid)
                continue
            ckpt_path, ckpt_source = ckpt

            params = load_run_params(conn, run_uuid)
            unlearner = params.get("unlearning.unlearner")

            # Swap `_local.csv` manifests (built against a local LibriSpeech mount) for their...
            for key in ("data_preparation.dataset.train_csv", "data_preparation.dataset.test_csv"):
                if key in params:
                    params[key] = delocalize_csv_path(params[key], run_folder, remap_log)

            # Remap any stale absolute /tmp/leaf_run_<...> paths.
            for key in ("train_csv", "test_csv", "root", "cache_dir"):
                val = params.get(f"data_preparation.dataset.{key}")
                if val is not None:
                    remap_absolute_path(str(val), run_folder, remap_log)

            skip_prefixes = diagnostic_prefixes(params)

            # Synthesize `++key=value` overrides from every MLflow param.
            overrides = [
                f"++{k}={hydra_override_value(v)}"
                for k, v in sorted(params.items())
                if k.split(".", 1)[0] not in skip_prefixes
            ]

            found.append(
                NestedRun(
                    run_folder=str(run_folder),
                    subject=str(subject),
                    subject_source="mlflow_tag",
                    mlflow_run_uuid=run_uuid,
                    mlflow_experiment_id=str(exp_id),
                    unlearner=unlearner,
                    checkpoint_path=ckpt_path,
                    checkpoint_source=ckpt_source,
                    overrides=overrides,
                    existing_eval_dirs=find_existing_eval_dirs(run_folder, run_uuid),
                )
            )
    finally:
        conn.close()
    return found


def print_summary(entries: list[NestedRun]) -> None:
    if not entries:
        print("No runs to evaluate.")
        return
    by_folder: dict[str, list[NestedRun]] = {}
    for e in entries:
        by_folder.setdefault(e.run_folder, []).append(e)
    print(f"\n{'run_folder':<70} {'unlearner':<14} {'subject':<10} {'ckpt_src':<18}")
    print("-" * 115)
    for folder, lst in sorted(by_folder.items()):
        for e in lst:
            print(
                f"{os.path.basename(folder):<70} "
                f"{(e.unlearner or '?'):<14} "
                f"{e.subject:<10} "
                f"{e.checkpoint_source:<18}"
            )
    print(f"\nTotal nested runs: {len(entries)}  (across {len(by_folder)} run folders)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT), help="Directory to scan for run folders")
    parser.add_argument("--out", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--remap-log", default=str(DEFAULT_REMAP_LOG))
    parser.add_argument("--skipped-log", default=str(DEFAULT_SKIPPED_LOG))
    args = parser.parse_args()

    run_folders = find_run_folders(Path(args.root))
    log.info("Scanning %d run folders under %s", len(run_folders), args.root)

    all_entries: list[NestedRun] = []
    with open(args.remap_log, "w") as remap_log, open(args.skipped_log, "w") as skipped_log:
        remap_log.write("# run_folder\told_path\t->\tnew_path\n")
        skipped_log.write("# run_folder\trun_uuid_or_whole_run\treason\n")
        for rf in run_folders:
            entries = collect_for_run(rf, remap_log, skipped_log)
            if UNLEARNER:
                entries = [e for e in entries if e.unlearner == UNLEARNER]
            all_entries.extend(entries)

    print_summary(all_entries)

    payload = {"runs": [asdict(e) for e in all_entries]}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %d entries to %s", len(all_entries), args.out)
    log.info("Remaps   -> %s", args.remap_log)
    log.info("Skipped  -> %s", args.skipped_log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
