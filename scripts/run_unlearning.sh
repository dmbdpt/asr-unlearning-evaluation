#!/bin/bash
# Unlearning + evaluation for one method.
set -euo pipefail

METHOD="${1:?usage: run_unlearning.sh <method> [hydra overrides...]}"
shift

REPO="${LEAF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$REPO/config/unlearning/${METHOD}.yaml" ]] || {
    echo "error: no such method '${METHOD}' (expected $REPO/config/unlearning/${METHOD}.yaml)" >&2
    echo "available:" >&2
    ls -1 "$REPO/config/unlearning" | sed 's/\.yaml$//; s/^/  /' >&2
    exit 1
}

MODE="${MODE:-per_subject}"
N_SUBJECTS="${N_SUBJECTS:-10}"
NUM_FORGET="${NUM_FORGET:-10}"
GPUS="${GPUS:-0,1,2}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
PORT="${PORT:-$((10000 + RANDOM % 20000))}"
MIA="${MIA:-false}"
SKIP_PRE="${SKIP_PRE:-true}"
SKIP_POST="${SKIP_POST:-true}"

# Per-subject mode caps the subject count with ++limit; the multi-speaker modes instead...
MODE_ARGS=()
if [[ "$MODE" == "per_subject" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${METHOD}_${N_SUBJECTS}spk}"
    MODE_ARGS=( "++clustering.enabled=false" "++limit=${N_SUBJECTS}" )
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${METHOD}_${MODE}_${NUM_FORGET}spk}"
    MODE_ARGS=(
        "unlearning.mode=${MODE}"
        "++clustering.distance_config.threshold_type=sequential"
        "++clustering.distance_config.threshold_value=${NUM_FORGET}"
    )
fi

echo "Starting ${METHOD} (${MODE}) on GPUs ${GPUS} -> experiment ${EXPERIMENT_NAME}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" \
OMP_NUM_THREADS="$NPROC" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
  torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$PORT" \
  run_experiment.py \
  --config-name config \
  data_preparation=default \
  "unlearning=${METHOD}" \
  "++skip_pre=${SKIP_PRE}" \
  "++skip_post=${SKIP_POST}" \
  "++evaluation.mia=${MIA}" \
  "${MODE_ARGS[@]}" \
  "artifacts.experiment_name=${EXPERIMENT_NAME}" \
  "$@"

echo "Run complete."
