#!/bin/bash
# Bayesian (Optuna) hyperparameter search for one unlearner.
set -euo pipefail

METHOD="${1:?usage: run_hpsearch.sh <method> [extra args...]}"
shift

REPO="${LEAF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$REPO/config/unlearning/${METHOD}.yaml" ]] || {
    echo "error: no such method '${METHOD}' (expected $REPO/config/unlearning/${METHOD}.yaml)" >&2
    exit 1
}

N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-${METHOD}_emd_optimization}"
OUTPUT_DIR="${OUTPUT_DIR:-hyperparam_results}"
FORGET_SPEAKERS="${FORGET_SPEAKERS:-163}"
MODEL_PATH="${MODEL_PATH:-}"
GPUS="${GPUS:-0,1,2}"
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"
PORT="${PORT:-$((10000 + RANDOM % 20000))}"

MODEL_PATH_ARGS=()
[[ -n "$MODEL_PATH" ]] && MODEL_PATH_ARGS=( --model_path "$MODEL_PATH" )

echo "Starting hyperparameter search — ${METHOD} (${N_TRIALS} trials) on GPUs ${GPUS}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPUS" \
OMP_NUM_THREADS="$NPROC" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
  torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$PORT" \
  scripts/hyperparameter_search.py \
  --method "$METHOD" \
  --config config/config.yaml \
  --n_trials "$N_TRIALS" \
  --study_name "$STUDY_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --forget_speakers $FORGET_SPEAKERS \
  "${MODEL_PATH_ARGS[@]}" \
  "$@"

echo "Search complete."
