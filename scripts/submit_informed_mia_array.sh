#!/bin/bash
# Submit a SLURM job array that runs scripts/run_informed_mia.py — the leave-one-out...
set -euo pipefail

# Repo root, derived from this script's own location. Override with LEAF_REPO.
REPO="${LEAF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${LEAF_VENV:-$REPO/.venv}"

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-$HOME/unlearned_ckps}"
OUT_DIR="${OUT_DIR:-$REPO/results/informed_mia}"
METHODS="${METHODS:-}"                  # space/comma separated; empty = every method dir
SUBJECTS="${SUBJECTS:-}"                # space/comma separated TARGETS; empty = every subject
# Longer than simple_mia's default: a method with n subjects loads ~2n checkpoints here...
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
PARTITION="${PARTITION:-hlt_phd}"
# Nodes with only Pascal GPUs (GTX 1080 Ti, Titan X) — incompatible with our env.
GPU_EXCLUDE_NODES="${GPU_EXCLUDE_NODES:-g01,g03,g04}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-8}"
LOG_DIR="${LOG_DIR:-$REPO/slurm_logs}"
JOB_NAME="${JOB_NAME:-informed_mia}"

# ---- run_informed_mia.py knobs (forwarded verbatim; see its --help) ------
TRAIN_CSV="${TRAIN_CSV:-${LEAF_DATA_DIR:-$REPO/data}/ls_train_all.csv}"
TEST_CSV="${TEST_CSV:-${LEAF_DATA_DIR:-$REPO/data}/ls_test_all.csv}"
MIN_SHADOWS="${MIN_SHADOWS:-2}"
MATCH_DISTRIBUTIONS="${MATCH_DISTRIBUTIONS:-1}"  # 0 = --no-match-distributions
ALIGN_N_BINS="${ALIGN_N_BINS:-20}"
NONMEMBER_SPLIT="${NONMEMBER_SPLIT:-speaker}"
PER_CORPUS_MATCH="${PER_CORPUS_MATCH:-1}"        # 0 = --no-per-corpus-match
HYBRID_RETAIN="${HYBRID_RETAIN:-0}"
N_ESTIMATORS="${N_ESTIMATORS:-100}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
LOSS_CER="${LOSS_CER:-0}"                        # 1 = --loss-cer
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_SAVE_FREQUENCY="${CACHE_SAVE_FREQUENCY:-100}"
SEED="${SEED:-42}"
REUSE_FEATURES="${REUSE_FEATURES:-0}"            # 1 = --reuse-features

# ---------------------------------------------------------------------------
# Method discovery: every <CHECKPOINTS_ROOT>/<method>/ dir, filtered by METHODS if given.
discover_methods() {
    python3 - "$CHECKPOINTS_ROOT" "$METHODS" <<'PY'
import sys
from pathlib import Path
root, wanted = Path(sys.argv[1]), sys.argv[2].replace(",", " ").split()
found = [d.name for d in sorted(root.iterdir())
         if d.is_dir() and not d.name.startswith(("_", "."))]
if wanted:
    missing = [m for m in wanted if m not in found]
    if missing:
        sys.exit(f"No method dir for: {' '.join(missing)}")
    found = [m for m in found if m in wanted]
for m in found:
    print(m)
PY
}

# ---------------------------------------------------------------------------
# Build the run_informed_mia.py argv shared by task mode and the dry-run plan.
build_args() {
    local method="$1"
    ARGS=(
        --checkpoints-root "$CHECKPOINTS_ROOT"
        --out-dir "$OUT_DIR"
        --methods "$method"
        --device cuda:0
        --train-csv "$TRAIN_CSV"
        --test-csv "$TEST_CSV"
        --min-shadows "$MIN_SHADOWS"
        --align-n-bins "$ALIGN_N_BINS"
        --nonmember-split "$NONMEMBER_SPLIT"
        --hybrid-retain "$HYBRID_RETAIN"
        --n-estimators "$N_ESTIMATORS"
        --class-weight "$CLASS_WEIGHT"
        --train-split-ratio "$TRAIN_SPLIT_RATIO"
        --batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --cache-save-frequency "$CACHE_SAVE_FREQUENCY"
        --seed "$SEED"
    )
    if [[ -n "$SUBJECTS" ]]; then ARGS+=(--subjects $SUBJECTS); fi
    if [[ "$MATCH_DISTRIBUTIONS" == "0" ]]; then ARGS+=(--no-match-distributions); fi
    if [[ "$PER_CORPUS_MATCH" == "0" ]]; then ARGS+=(--no-per-corpus-match); fi
    if [[ "$LOSS_CER" == "1" ]]; then ARGS+=(--loss-cer); fi
    if [[ "$REUSE_FEATURES" == "1" ]]; then ARGS+=(--reuse-features); fi
    if [[ "${FORCE:-}" == "1" ]]; then ARGS+=(--overwrite); fi
}

# ---------------------------------------------------------------------------
# Task mode: launched by SLURM.
if [[ "${TASK_MODE:-0}" == "1" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set in task mode}"
    source "$VENV/bin/activate"
    export CUDA_VISIBLE_DEVICES="0"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    mapfile -t methods < <(discover_methods)
    method="${methods[$SLURM_ARRAY_TASK_ID]}"
    echo "[task] array_id=$SLURM_ARRAY_TASK_ID method=$method"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

    build_args "$method"
    cd "$REPO"
    python3 scripts/run_informed_mia.py "${ARGS[@]}"
    exit $?
fi

# ---------------------------------------------------------------------------
# Driver: discover methods, optionally dry-run, otherwise sbatch.
if [[ ! -d "$CHECKPOINTS_ROOT" ]]; then
    echo "Missing checkpoint root: $CHECKPOINTS_ROOT" >&2
    exit 2
fi

mapfile -t all_methods < <(discover_methods)
n=${#all_methods[@]}
if (( n == 0 )); then
    echo "No method dirs under $CHECKPOINTS_ROOT for those filters." >&2
    exit 2
fi
last=$(( n - 1 ))

mkdir -p "$LOG_DIR"

echo "Checkpoint root : $CHECKPOINTS_ROOT"
echo "Methods (${n})   : ${all_methods[*]}"
echo "Target filter   : ${SUBJECTS:-<all>} (every subject still counts as a shadow)"
echo "Out dir         : $OUT_DIR"
echo "Min shadows     : $MIN_SHADOWS   Hybrid retain: $HYBRID_RETAIN   Class weight: $CLASS_WEIGHT"
echo "Distribution match : $([[ "$MATCH_DISTRIBUTIONS" == "1" ]] && echo on || echo OFF)$([[ "$PER_CORPUS_MATCH" == "1" ]] && echo ' (per corpus)' || echo ' (pooled)')"
echo "Partition       : $PARTITION   Exclude: $GPU_EXCLUDE_NODES   Mem: $MEM   CPUs: $CPUS"
echo "Time / task     : $TIME_LIMIT"
echo "Log dir         : $LOG_DIR"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 — printing per-task plan, not submitting."
    for i in "${!all_methods[@]}"; do
        build_args "${all_methods[$i]}"
        echo "  array_task $i  method=${all_methods[$i]}"
        echo "    run_informed_mia.py ${ARGS[*]}"
    done
    echo
    echo "Would run: sbatch --array=0-$last ... $0 (TASK_MODE=1)"
    exit 0
fi

sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --gres="gpu:1" \
    --exclude="${GPU_EXCLUDE_NODES}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --array="0-$last" \
    --output="$LOG_DIR/${JOB_NAME}-%A_%a.out" \
    --error="$LOG_DIR/${JOB_NAME}-%A_%a.err" \
    --export=ALL,TASK_MODE=1,CHECKPOINTS_ROOT="$CHECKPOINTS_ROOT",OUT_DIR="$OUT_DIR",METHODS="$METHODS",SUBJECTS="$SUBJECTS",TRAIN_CSV="$TRAIN_CSV",TEST_CSV="$TEST_CSV",MIN_SHADOWS="$MIN_SHADOWS",MATCH_DISTRIBUTIONS="$MATCH_DISTRIBUTIONS",ALIGN_N_BINS="$ALIGN_N_BINS",NONMEMBER_SPLIT="$NONMEMBER_SPLIT",PER_CORPUS_MATCH="$PER_CORPUS_MATCH",HYBRID_RETAIN="$HYBRID_RETAIN",N_ESTIMATORS="$N_ESTIMATORS",CLASS_WEIGHT="$CLASS_WEIGHT",LOSS_CER="$LOSS_CER",TRAIN_SPLIT_RATIO="$TRAIN_SPLIT_RATIO",BATCH_SIZE="$BATCH_SIZE",NUM_WORKERS="$NUM_WORKERS",CACHE_SAVE_FREQUENCY="$CACHE_SAVE_FREQUENCY",SEED="$SEED",REUSE_FEATURES="$REUSE_FEATURES",FORCE="${FORCE:-}" \
    "$0"
