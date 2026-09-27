#!/bin/bash
# ---------------------------------------------------------------------------
# Unified experiment launcher for the LeaF machine-unlearning framework.
# ---------------------------------------------------------------------------

set -euo pipefail

usage() {
    cat <<'USAGE'
Unified experiment launcher for the LeaF machine-unlearning framework.

Usage: bash scripts/run.sh [OPTIONS] [HYDRA_OVERRIDES...]

  -c, --config   CONFIG   Hydra config name under config/ (default: config)
  -m, --method   METHOD   Unlearning method (e.g. scrub, bad_teacher)
  -g, --gpus     IDS      Comma-separated GPU ids (default: 0)
  -n, --nproc    N        torchrun processes (default: 1)
  -p, --port     PORT     Master port (default: 4337)
  -r, --results  DIR      Results base dir (default: $LEAF_RESULTS_DIR, else <repo>/results)
  -h, --help              Print this help

Examples:
  bash scripts/run.sh --method scrub
  bash scripts/run.sh --method bad_teacher --gpus 5,6,7 --nproc 3
  bash scripts/run.sh --method scrub unlearning.unlearner_config.lr=1e-5
  bash scripts/run.sh --config config data_preparation=local unlearning=cfk

For the per-method launcher with tuned hyperparameters, see
scripts/run_unlearning.sh and scripts/README.md.
USAGE
}

# ---- Defaults ---------------------------------------------------------------
CONFIG="config"
METHOD=""
GPUS="0"
NPROC="1"
PORT="4337"
REPO="${LEAF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_BASE="${LEAF_RESULTS_DIR:-$REPO/results}"
VENV="${LEAF_VENV:-$REPO/.venv}/bin/activate"

# ---- Parse flags ------------------------------------------------------------
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)   CONFIG="$2";  shift 2 ;;
        -m|--method)   METHOD="$2";  shift 2 ;;
        -g|--gpus)     GPUS="$2";    shift 2 ;;
        -n|--nproc)    NPROC="$2";   shift 2 ;;
        -p|--port)     PORT="$2";    shift 2 ;;
        -r|--results)  RESULTS_BASE="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *)             HYDRA_OVERRIDES+=("$1"); shift ;;
    esac
done

if [[ -n "$METHOD" ]]; then
    HYDRA_OVERRIDES=("++unlearning.unlearner=${METHOD}" "${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}")
fi

# ---- Environment ------------------------------------------------------------
# shellcheck disable=SC1090
source "$VENV"

echo "GPUs: $GPUS"
nvidia-smi 2>/dev/null || true

# ---- Scratch workspace ------------------------------------------------------
WORKDIR=$(mktemp -d /tmp/leaf_run_XXXXXX)
echo "Working directory: $WORKDIR"

save_and_cleanup() {
    local exit_code=$?
    RUN_ID=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${RESULTS_BASE}/run_${RUN_ID}"
    mkdir -p "$RESULTS_DIR"
    echo "Saving results to $RESULTS_DIR ..."
    rsync -av --exclude='.cache' "$WORKDIR"/ "$RESULTS_DIR"/
    echo "Cleaning up $WORKDIR ..."
    rm -rf "$WORKDIR"
    exit $exit_code
}
trap save_and_cleanup EXIT

cd "$WORKDIR"
echo "Copying project files ..."
rsync -av \
    "$REPO"/src \
    "$REPO"/.cache \
    "$REPO"/config \
    "$REPO"/run_experiment.py \
    "$WORKDIR"/

# ---- Launch -----------------------------------------------------------------
echo "Launching with config=${CONFIG}, nproc=${NPROC}, overrides: ${HYDRA_OVERRIDES[*]+"${HYDRA_OVERRIDES[*]}"}"

CUDA_VISIBLE_DEVICES="$GPUS" \
OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$PORT" \
    run_experiment.py \
    --config-name "$CONFIG" \
    "${HYDRA_OVERRIDES[@]+"${HYDRA_OVERRIDES[@]}"}"

echo "Run complete."
