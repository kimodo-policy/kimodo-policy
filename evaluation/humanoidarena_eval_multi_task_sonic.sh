#!/usr/bin/env bash
set -Eeuo pipefail

# Run one checkpoint sequentially on a user-selected set of HumanoidArena
# SONIC tasks.  The task groups are:
#   hoi -> doubledesk, football, pp_box
#   hsi -> boxing, open_door, sit_sofa, vision_navi
#   all -> hoi followed by hsi
#
# The actual simulator/model-server work remains in
# humanoidarena_eval_signal_task_sonic.sh.  This wrapper only provides a stable
# order, per-task result directories, logs, and failure/timeout handling.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${KIMODO_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
CHECKPOINT="${KIMODO_CHECKPOINT:-}"
GPU_LIST="${KIMODO_GPUS:-0}"
SEED_LIST="${KIMODO_EVAL_SEEDS:-}"
REPEATS_PER_SEED="${KIMODO_REPEATS_PER_SEED:-}"
RESULTS_DIR="${KIMODO_HOI_RESULTS_DIR:-}"
DTYPE="${KIMODO_DTYPE:-fp32}"
DIFFUSION_STEPS="${KIMODO_DIFFUSION_STEPS:-10}"
EXECUTION_FRAMES="${KIMODO_EXECUTION_FRAMES:-0}"
RTC="${KIMODO_RTC:-1}"
RTC_OVERLAP_FRAMES="${KIMODO_RTC_OVERLAP_FRAMES:-12}"
RTC_FROZEN_FRAMES="${KIMODO_RTC_FROZEN_FRAMES:-1}"
RTC_RAMP_POWER="${KIMODO_RTC_RAMP_POWER:-1.0}"
PERSISTENT_SIM="${KIMODO_PERSISTENT_SIM:-}"
DETERMINISTIC_EVAL="${KIMODO_DETERMINISTIC_EVAL:-0}"
RECORD_VIDEO_EVERY_N="${KIMODO_RECORD_VIDEO_EVERY_N:-0}"
PORT_BASE="${KIMODO_HOI_PORT_BASE:-18080}"
SERVER_READY_TIMEOUT="${KIMODO_SERVER_READY_TIMEOUT:-600}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-10800}"
MAX_STEPS="${KIMODO_MAX_STEPS:-}"
TEXT_CACHE="${KIMODO_TEXT_EMBEDDING_CACHE:-}"
SERVER_PYTHON="${KIMODO_SERVER_PYTHON:-${SERVER_PYTHON:-}}"
SIM_ENV="${KIMODO_SIM_ENV:-${SIM_ENV:-}}"
CONDA_ROOT="${KIMODO_CONDA_ROOT:-${CONDA_ROOT:-}}"
HUMANOIDARENA_ROOT="${KIMODO_HUMANOIDARENA_ROOT:-${HUMANOIDARENA_ROOT:-}}"
DRY_RUN=0

EVAL_SCRIPT="${SCRIPT_DIR}/humanoidarena_eval_signal_task_sonic.sh"
TASK_REQUESTS=()
TASKS=()
HOI_TASKS=(doubledesk football pp_box)
HSI_TASKS=(boxing open_door sit_sofa vision_navi)

usage() {
  cat <<'EOF'
Run one checkpoint sequentially on user-selected HumanoidArena SONIC tasks.

Usage:
  humanoidarena_eval_multi_task_sonic.sh --task TASK [TASK ...] --checkpoint PATH [options]
  humanoidarena_eval_multi_task_sonic.sh --task hoi --checkpoint PATH [options]

Required:
  --task TASK [TASK ...]   Tasks/groups in the exact execution order. Supported:
                          hoi, hsi, all, doubledesk, football, pp_box,
                          boxing, open_door, sit_sofa, vision_navi
  --checkpoint PATH       Checkpoint directory containing training_state.pt and config.json

Paths and scheduling:
  --project PATH          Kimodo-Policy project root (default: repository root)
  --gpus LIST             Comma/space-separated GPU indices (default: 0)
  --seeds LIST            Override official task seeds, e.g. 0,1,2
  --repeats N             Override episodes per seed
  --results-dir PATH      Root directory for the selected task results
  --port-base N           First model-server port (default: 18080)
  --timeout N             Timeout for each task in seconds (default: 10800)
  --max-steps N           Override the official task maximum simulation steps
  --server-python PATH    Python executable used by the model server
  --sim-env PATH          Isaac simulator environment directory
  --conda-root PATH       Conda installation used to activate --sim-env

Evaluation options (forwarded to the signal-task evaluator):
  --dtype fp32|bf16
  --diffusion-steps N
  --execution-frames N
  --rtc 0|1
  --rtc-overlap-frames N
  --rtc-frozen-frames N
  --rtc-ramp-power X
  --persistent-sim 0|1
  --deterministic-eval 0|1
  --record-video-every-n N
  --server-ready-timeout N
  --text-cache PATH

Other:
  --dry-run               Validate and print all selected launch plans only
  -h, --help              Show this help

Example:
  KIMODO_SIM_ENV=/path/to/unitree_sim_env \
  KIMODO_SERVER_PYTHON=/path/to/model-server-python \
  bash evaluation/humanoidarena_eval_multi_task_sonic.sh \
    --task hoi \
    --checkpoint log/experiments/my_run/checkpoint_500000 \
    --gpus 0,1,2 --seeds 0,1,2
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      shift
      [[ $# -gt 0 && "${1:0:1}" != "-" ]] || {
        echo "--task requires at least one task or task group" >&2
        exit 2
      }
      while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do
        TASK_REQUESTS+=("$1")
        shift
      done
      ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --project) need_value "$@"; PROJECT_ROOT="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPU_LIST="$2"; shift 2 ;;
    --seeds) need_value "$@"; SEED_LIST="$2"; shift 2 ;;
    --repeats) need_value "$@"; REPEATS_PER_SEED="$2"; shift 2 ;;
    --results-dir) need_value "$@"; RESULTS_DIR="$2"; shift 2 ;;
    --dtype) need_value "$@"; DTYPE="$2"; shift 2 ;;
    --diffusion-steps) need_value "$@"; DIFFUSION_STEPS="$2"; shift 2 ;;
    --execution-frames) need_value "$@"; EXECUTION_FRAMES="$2"; shift 2 ;;
    --rtc) need_value "$@"; RTC="$2"; shift 2 ;;
    --rtc-overlap-frames) need_value "$@"; RTC_OVERLAP_FRAMES="$2"; shift 2 ;;
    --rtc-frozen-frames) need_value "$@"; RTC_FROZEN_FRAMES="$2"; shift 2 ;;
    --rtc-ramp-power) need_value "$@"; RTC_RAMP_POWER="$2"; shift 2 ;;
    --persistent-sim) need_value "$@"; PERSISTENT_SIM="$2"; shift 2 ;;
    --deterministic-eval) need_value "$@"; DETERMINISTIC_EVAL="$2"; shift 2 ;;
    --record-video-every-n) need_value "$@"; RECORD_VIDEO_EVERY_N="$2"; shift 2 ;;
    --port-base) need_value "$@"; PORT_BASE="$2"; shift 2 ;;
    --timeout) need_value "$@"; EVAL_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-steps) need_value "$@"; MAX_STEPS="$2"; shift 2 ;;
    --server-ready-timeout) need_value "$@"; SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    --text-cache) need_value "$@"; TEXT_CACHE="$2"; shift 2 ;;
    --server-python) need_value "$@"; SERVER_PYTHON="$2"; shift 2 ;;
    --sim-env) need_value "$@"; SIM_ENV="$2"; shift 2 ;;
    --conda-root) need_value "$@"; CONDA_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -z "$CHECKPOINT" ]]; then
        CHECKPOINT="$1"
        shift
      else
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

[[ ${#TASK_REQUESTS[@]} -gt 0 ]] || {
  echo "At least one --task is required (for example: --task hoi)." >&2
  usage >&2
  exit 2
}
[[ -n "$CHECKPOINT" ]] || { echo "A checkpoint is required." >&2; usage >&2; exit 2; }
[[ -x "$EVAL_SCRIPT" ]] || { echo "Evaluation script is missing or not executable: $EVAL_SCRIPT" >&2; exit 3; }
if [[ "$PROJECT_ROOT" != /* ]]; then
  PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}/${PROJECT_ROOT}"
fi
PROJECT_ROOT="$(readlink -m "$PROJECT_ROOT")"
HUMANOIDARENA_ROOT="${HUMANOIDARENA_ROOT:-${PROJECT_ROOT}/HumanoidArena}"
if [[ "$CHECKPOINT" != /* ]]; then
  if [[ -e "${PROJECT_ROOT}/${CHECKPOINT}" ]]; then
    CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
  fi
fi
CHECKPOINT="$(readlink -m "$CHECKPOINT")"
[[ -f "${CHECKPOINT}/training_state.pt" ]] || { echo "Checkpoint is missing training_state.pt: $CHECKPOINT" >&2; exit 3; }
[[ -f "${CHECKPOINT}/config.json" ]] || { echo "Checkpoint is missing config.json: $CHECKPOINT" >&2; exit 3; }

append_requested_task() {
  local raw="${1,,}"
  case "$raw" in
    hoi) TASKS+=("${HOI_TASKS[@]}") ;;
    hsi) TASKS+=("${HSI_TASKS[@]}") ;;
    all) TASKS+=("${HOI_TASKS[@]}" "${HSI_TASKS[@]}") ;;
    doubledesk|double_desk|double-desk|doubledesk_sonic|double_desk_sonic)
      TASKS+=(doubledesk) ;;
    football|football_sonic|football_single|football_single_sonic)
      TASKS+=(football) ;;
    pp_box|pp-box|ppbox|pickplace_box|pick_place_box|pp_box_sonic)
      TASKS+=(pp_box) ;;
    boxing|boxing_sonic) TASKS+=(boxing) ;;
    open_door|open-door|opendoor|open_door_sonic) TASKS+=(open_door) ;;
    sit_sofa|sit-sofa|sitsofa|sit_sofa_sonic) TASKS+=(sit_sofa) ;;
    vision_navi|vision-navi|navigation|vision_navi_sonic)
      TASKS+=(vision_navi) ;;
    *)
      echo "Unsupported task or task group: $1" >&2
      echo "Supported groups: hoi, hsi, all" >&2
      exit 2
      ;;
  esac
}

for ((task_index = 0; task_index < ${#TASK_REQUESTS[@]}; task_index++)); do
  raw_task="${TASK_REQUESTS[$task_index]}"
  # Accept the natural spellings `double desk` and `pp box` in addition to
  # doubledesk and pp_box.  The parser preserves all tokens after --task so
  # their order remains exactly the order requested by the user.
  if [[ "${raw_task,,}" == double && $((task_index + 1)) -lt ${#TASK_REQUESTS[@]} && "${TASK_REQUESTS[$((task_index + 1))],,}" == desk ]]; then
    append_requested_task doubledesk
    task_index=$((task_index + 1))
  elif [[ "${raw_task,,}" == pp && $((task_index + 1)) -lt ${#TASK_REQUESTS[@]} && "${TASK_REQUESTS[$((task_index + 1))],,}" == box ]]; then
    append_requested_task pp_box
    task_index=$((task_index + 1))
  else
    append_requested_task "$raw_task"
  fi
done

[[ ${#TASKS[@]} -gt 0 ]] || { echo "No tasks selected." >&2; exit 2; }

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required." >&2
  exit 3
fi
[[ "$DTYPE" == fp32 || "$DTYPE" == bf16 ]] || { echo "--dtype must be fp32 or bf16" >&2; exit 2; }
for value in "$DIFFUSION_STEPS" "$EXECUTION_FRAMES" "$RTC_OVERLAP_FRAMES" "$RTC_FROZEN_FRAMES" "$RECORD_VIDEO_EVERY_N" "$PORT_BASE" "$EVAL_TIMEOUT_SECONDS" "$SERVER_READY_TIMEOUT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected non-negative integer, got: $value" >&2; exit 2; }
done
if [[ -n "$MAX_STEPS" && ! "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "--max-steps must be a non-negative integer" >&2
  exit 2
fi
[[ "$RTC" == 0 || "$RTC" == 1 ]] || { echo "--rtc must be 0 or 1" >&2; exit 2; }
[[ "$DETERMINISTIC_EVAL" == 0 || "$DETERMINISTIC_EVAL" == 1 ]] || { echo "--deterministic-eval must be 0 or 1" >&2; exit 2; }
if [[ -n "$PERSISTENT_SIM" && "$PERSISTENT_SIM" != 0 && "$PERSISTENT_SIM" != 1 ]]; then
  echo "--persistent-sim must be 0 or 1" >&2
  exit 2
fi

CHECKPOINT_NAME="$(basename "$CHECKPOINT")"
TASK_LABEL="$(printf '%s+' "${TASKS[@]}")"
TASK_LABEL="${TASK_LABEL%+}"
if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="${PROJECT_ROOT}/eval_results/multi_task_sonic_${CHECKPOINT_NAME}_${TASK_LABEL}_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="${PROJECT_ROOT}/${RESULTS_DIR}"
fi
RESULTS_DIR="$(readlink -m "$RESULTS_DIR")"
LOG_DIR="${RESULTS_DIR}/logs"
STATUS_FILE="${RESULTS_DIR}/status.tsv"
mkdir -p "$LOG_DIR"

exec 9>"${RESULTS_DIR}/.multi_task.lock"
if ! flock -n 9; then
  echo "Another HOI task evaluation is already running for: $RESULTS_DIR" >&2
  exit 4
fi

write_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "${3:-}" | tee -a "$STATUS_FILE"
}

common_args=(
  --project "$PROJECT_ROOT"
  --checkpoint "$CHECKPOINT"
  --gpus "$GPU_LIST"
  --dtype "$DTYPE"
  --diffusion-steps "$DIFFUSION_STEPS"
  --execution-frames "$EXECUTION_FRAMES"
  --rtc "$RTC"
  --rtc-overlap-frames "$RTC_OVERLAP_FRAMES"
  --rtc-frozen-frames "$RTC_FROZEN_FRAMES"
  --rtc-ramp-power "$RTC_RAMP_POWER"
  --record-video-every-n "$RECORD_VIDEO_EVERY_N"
  --port-base "$PORT_BASE"
  --server-ready-timeout "$SERVER_READY_TIMEOUT"
)
[[ -n "$SEED_LIST" ]] && common_args+=(--seeds "$SEED_LIST")
[[ -n "$REPEATS_PER_SEED" ]] && common_args+=(--repeats "$REPEATS_PER_SEED")
[[ -n "$MAX_STEPS" ]] && common_args+=(--max-steps "$MAX_STEPS")
[[ -n "$PERSISTENT_SIM" ]] && common_args+=(--persistent-sim "$PERSISTENT_SIM")
[[ -n "$TEXT_CACHE" ]] && common_args+=(--text-cache "$TEXT_CACHE")
[[ "$DETERMINISTIC_EVAL" == 1 ]] && common_args+=(--deterministic-eval 1)

write_status "STARTED" "all" "tasks=${TASKS[*]}; checkpoint=$CHECKPOINT; results=$RESULTS_DIR"
for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  task_result_dir="${RESULTS_DIR}/${task}"
  task_log="${LOG_DIR}/$(printf '%02d' "$((index + 1))")_${task}.log"
  task_args=("${common_args[@]}" --task "$task" --results-dir "$task_result_dir")
  [[ "$DRY_RUN" == 1 ]] && task_args+=(--dry-run)

  write_status "STARTED" "$task" "log=$task_log"
  set +e
  timeout --signal=TERM --kill-after=120s "$EVAL_TIMEOUT_SECONDS" \
    env \
      KIMODO_HUMANOIDARENA_ROOT="$HUMANOIDARENA_ROOT" \
      KIMODO_SERVER_PYTHON="$SERVER_PYTHON" \
      KIMODO_SIM_ENV="$SIM_ENV" \
      KIMODO_CONDA_ROOT="$CONDA_ROOT" \
      bash "$EVAL_SCRIPT" "${task_args[@]}" 2>&1 | tee "$task_log"
  task_status=${PIPESTATUS[0]}
  set -e

  if (( task_status == 124 )); then
    write_status "TIMEOUT" "$task" "limit=${EVAL_TIMEOUT_SECONDS}s; log=$task_log"
    exit 124
  elif (( task_status != 0 )); then
    write_status "FAILED" "$task" "exit=$task_status; log=$task_log"
    exit "$task_status"
  fi

  if [[ "$DRY_RUN" == 0 && ! -f "${task_result_dir}/aggregate_summary.json" ]]; then
    write_status "INVALID_RESULT" "$task" "aggregate_summary.json missing"
    exit 5
  fi
  write_status "COMPLETED" "$task" "result=$task_result_dir"
done

write_status "COMPLETED" "all" "tasks=${TASKS[*]}"
echo "HOI evaluation complete: $RESULTS_DIR"
