#!/bin/bash
# Submit a SLURM job array that re-evaluates every (run_folder, forgotten subject) pair...
set -euo pipefail

# Repo root, derived from this script's own location. Override with LEAF_REPO.
REPO="${LEAF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${LEAF_VENV:-$REPO/.venv}"
EVAL_JSON="${EVAL_JSON:-$REPO/scripts/eval_runs.json}"
EVALS_PER_TASK="${EVALS_PER_TASK:-1}"   # keep at 1: each eval uses all 3 GPUs
TIME_LIMIT="${TIME_LIMIT:-15:00:00}"
PARTITION="${PARTITION:-hlt_phd}"
# Nodes with only Pascal GPUs (GTX 1080 Ti, Titan X) — incompatible with our env.
GPU_EXCLUDE_NODES="${GPU_EXCLUDE_NODES:-g01,g03,g04}"
MEM="${MEM:-32G}"
LOG_DIR="${LOG_DIR:-$REPO/slurm_logs}"
JOB_NAME="${JOB_NAME:-eval_array}"

# ---------------------------------------------------------------------------
# Helper invoked once per (run_folder, subject) — runs a single evaluation pinned to one...
run_one_eval() {
    local idx="$1"
    local local_gpu="$2"

    # Pull this eval's fields out of the JSON via python (jq isn't installed everywhere on...
    local fields_json
    fields_json=$(python3 - "$EVAL_JSON" "$idx" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
e = data["runs"][int(sys.argv[2])]
print(json.dumps({
    "run_folder":       e["run_folder"],
    "subject":          e["subject"],
    "checkpoint_path":  e["checkpoint_path"],
    "unlearner":        e.get("unlearner") or "",
    "overrides":        e.get("overrides") or [],
    "subject_idx":      -1,
}))
PY
)
    local run_folder subject ckpt unlearner
    run_folder=$(echo "$fields_json"   | python3 -c "import json,sys; print(json.load(sys.stdin)['run_folder'])")
    subject=$(echo "$fields_json"      | python3 -c "import json,sys; print(json.load(sys.stdin)['subject'])")
    ckpt=$(echo "$fields_json"         | python3 -c "import json,sys; print(json.load(sys.stdin)['checkpoint_path'])")
    unlearner=$(echo "$fields_json"    | python3 -c "import json,sys; print(json.load(sys.stdin)['unlearner'])")

    # Replay the run's full MLflow-param set as "++key=value" overrides against the live...
    local config_path="$REPO/config" config_name="config"
    local -a replay_overrides=()
    while IFS= read -r -d '' o; do
        replay_overrides+=("$o")
    done < <(echo "$fields_json" | python3 -c "
import json, sys
for o in json.load(sys.stdin)['overrides']:
    sys.stdout.write(o + '\0')
")

    local ts="${REEVAL_TS:-$(date +%Y%m%d_%H%M%S)_$$}"
    local out_dir="$run_folder/reeval_$ts/subject_$subject"

    # ----- Skip if a previous re-eval already covers this checkpoint -------
    if [[ -z "${FORCE:-}" ]] && python3 - "$run_folder" "$subject" "$ckpt" <<'PY' >/dev/null 2>&1
import os, sys, glob
run_folder, subject, ckpt = sys.argv[1:4]
ckpt_mtime = os.path.getmtime(ckpt)
# If a reeval_* directory already contains results newer than the checkpoint for this...
for d in sorted(glob.glob(os.path.join(run_folder, "reeval_*", f"subject_{subject}")), reverse=True):
    js = os.path.join(d, "results_serialized.json")
    if os.path.exists(js) and os.path.getmtime(js) >= ckpt_mtime:
        sys.exit(0)
sys.exit(1)
PY
    then
        echo "[task] idx=$idx subject=$subject: existing eval is newer than checkpoint — skipping (set FORCE=1 to override)"
        return 0
    fi

    # reeval_$ts is unique per submission, so nothing can collide here.
    mkdir -p "$out_dir"

    # ----- Per-eval scratch cache_dir: keeps concurrent evals off one cache_index.json.
    local task_cache
    task_cache=$(mktemp -d -t reeval_XXXX -p /tmp)
    mkdir -p "$task_cache"
    if [[ -d "$REPO/.cache/dataset"  ]]; then ln -sf "$REPO/.cache/dataset"  "$task_cache/dataset";  fi
    if [[ -d "$REPO/.cache/features" ]]; then ln -sf "$REPO/.cache/features" "$task_cache/features"; fi
    if [[ -f "$REPO/.cache/cache_index.json" ]]; then
        cp "$REPO/.cache/cache_index.json" "$task_cache/cache_index.json"
    fi

    local tracking_db="$out_dir/mlflow.db"
    local log_file="$out_dir/eval.log"

    echo "[task] idx=$idx gpu=$local_gpu subject=$subject ckpt=$ckpt"
    echo "[task]   out=$out_dir"

    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="0,1,2" \
        torchrun --nproc_per_node=3 --standalone scripts/eval_from_run.py \
            --config-path "$config_path" \
            --config-name "$config_name" \
            "${replay_overrides[@]}" \
            ++artifacts.tracking_uri="'sqlite:///$tracking_db?timeout=600'" \
            ++artifacts.cache_dir="$task_cache" \
            ++evaluation.mia_config.cache_dir="$out_dir/mia_features" \
            ++evaluation.mia_config.random_state=null \
            ++unlearning.unlearner_config.test_loss_mean=null \
            ++training.model_config.beam_size=5 \
            ++training.model_config.lm_weight=0 \
            ++evaluation.batch_size=8 \
            ++evaluation.mia_config.batch_size=8 \
            ++evaluation.metrics=true \
            ++evaluation.losses=true \
            ++evaluation.mia=true \
            ++evaluation.downsample_to_test=false \
            ++eval_from_run.checkpoint_path="$ckpt" \
            ++eval_from_run.target_subject="$subject" \
            ++eval_from_run.output_dir="$out_dir" \
            ++eval_from_run.skip_pre=true \
            > "$log_file" 2>&1
    )
    local rc=$?

    # Mark success for idempotency (touch a sentinel newer than the ckpt).
    if [[ $rc -eq 0 ]]; then
        touch "$out_dir/results_serialized.json"
        echo "[task] idx=$idx subject=$subject DONE"
    else
        echo "[task] idx=$idx subject=$subject FAILED (rc=$rc) — see $log_file"
    fi
    # Clean up the symlink cache; leave failures around for inspection.
    if [[ $rc -eq 0 ]]; then
        rm -rf "$task_cache"
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# Task mode: launched by SLURM.
if [[ "${TASK_MODE:-0}" == "1" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set in task mode}"
    source "$VENV/bin/activate"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_P2P_DISABLE=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    # 3 torchrun ranks share this task's CPU allocation (9 == --cpus-per-task below).
    export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / 3 ))

    total=$(python3 -c "import json; print(len(json.load(open('$EVAL_JSON'))['runs']))")
    base=$(( SLURM_ARRAY_TASK_ID * EVALS_PER_TASK ))

    echo "[task] array_id=$SLURM_ARRAY_TASK_ID base=$base total=$total gpus=$CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

    pids=()
    for ((j = 0; j < EVALS_PER_TASK; j++)); do
        idx=$(( base + j ))
        if (( idx >= total )); then break; fi
        run_one_eval "$idx" "$j" &
        pids+=("$!")
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=$(( fail + 1 )); done
    if (( fail > 0 )); then
        echo "[task] $fail/${#pids[@]} evals failed"
        exit 1
    fi
    echo "[task] array_id=$SLURM_ARRAY_TASK_ID: all ${#pids[@]} evals succeeded"
    exit 0
fi

# ---------------------------------------------------------------------------
# Driver: compute array size, optionally dry-run, otherwise sbatch.
if [[ ! -f "$EVAL_JSON" ]]; then
    echo "Missing $EVAL_JSON — run scripts/collect_eval_runs.py first." >&2
    exit 2
fi

n=$(python3 -c "import json; print(len(json.load(open('$EVAL_JSON'))['runs']))")
tasks=$(( (n + EVALS_PER_TASK - 1) / EVALS_PER_TASK ))
last=$(( tasks - 1 ))
# PID suffix keeps two submissions launched in the same second from sharing a reeval_<ts>...
ts=$(date +%Y%m%d_%H%M%S)_$$

mkdir -p "$LOG_DIR"

echo "JSON           : $EVAL_JSON"
echo "Total evals    : $n"
echo "Evals per task : $EVALS_PER_TASK"
echo "Array size     : $tasks (0..$last)"
echo "Partition      : $PARTITION   Exclude: $GPU_EXCLUDE_NODES   Evals/GPU: $EVALS_PER_TASK"
echo "Time / task    : $TIME_LIMIT     Mem: $MEM"
echo "Log dir        : $LOG_DIR"
echo "Timestamp tag  : $ts"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 — printing per-task plan, not submitting."
    python3 - "$EVAL_JSON" "$EVALS_PER_TASK" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))["runs"]
per = int(sys.argv[2])
for i in range(0, len(data), per):
    grp = data[i:i + per]
    print(f"  array_task {i//per:>3}:")
    for j, e in enumerate(grp):
        print(f"    gpu{j}  subject={e['subject']:<6} "
              f"unlearner={(e.get('unlearner') or '?'):<14} "
              f"folder={e['run_folder']}")
PY
    echo
    echo "Would run: sbatch --array=0-$last ... $0 (TASK_MODE=1)"
    exit 0
fi

# Submit the array. Note: we pass TASK_MODE/REEVAL_TS through --export.
sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --gres="gpu:3" \
    --exclude="${GPU_EXCLUDE_NODES}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=9 \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --array="0-$last" \
    --output="$LOG_DIR/${JOB_NAME}-%A_%a.out" \
    --error="$LOG_DIR/${JOB_NAME}-%A_%a.err" \
    --export=ALL,TASK_MODE=1,EVAL_JSON="$EVAL_JSON",EVALS_PER_TASK="$EVALS_PER_TASK",REEVAL_TS="$ts",FORCE="${FORCE:-}" \
    "$0"
