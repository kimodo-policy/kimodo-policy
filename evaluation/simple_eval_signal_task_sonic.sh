#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TASK="${TASK:-G1WholebodyCloseDoorTeleop-v0}"
CHECKPOINT="${CHECKPOINT:-}"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-0}"
# The official SIMPLE protocol has 10 fixed episodes for each of levels 0, 1
# and 2.  This script launches one process per level and aggregates the three
# level summaries into a single task/checkpoint result.
EPISODES="${EPISODES:-10}"
LEVELS="${LEVELS:-0,1,2}"
RESULTS_DIR="${RESULTS_DIR:-}"
TEXT_CACHE="${TEXT_CACHE:-${PROJECT_ROOT}/data/cache/Simple}"
SIMPLE_DATA_DIR="${SIMPLE_DATA_DIR:-}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"
PYTHON_BIN="${KIMODO_PYTHON:-${PROJECT_ROOT}/SIMPLE/.venv/bin/python}"
MODEL_SITE_PACKAGES="${KIMODO_MODEL_SITE_PACKAGES:-}"
ISAAC_CACHE_ROOT="${ISAAC_CACHE_ROOT:-}"
# Keep the evaluator defaults aligned with evaluation/simple_eval.md.
DTYPE="${DTYPE:-fp32}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"
# Execute the complete 50-frame prediction horizon before replanning.  The
# model is trained to produce a 50-frame chunk, so the default evaluator must
# keep prediction and execution horizons aligned.
EXECUTION_FRAMES="${EXECUTION_FRAMES:-50}"
RTC="${RTC:-0}"
RTC_OVERLAP_FRAMES="${RTC_OVERLAP_FRAMES:-0}"
RTC_FROZEN_FRAMES="${RTC_FROZEN_FRAMES:-0}"
RTC_RAMP_POWER="${RTC_RAMP_POWER:-1.0}"
MAX_NAVIGATION_SPEED="${MAX_NAVIGATION_SPEED:-1.5}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"
MAX_STEPS="${MAX_STEPS:-auto}"
RENDER_HZ="${RENDER_HZ:-50}"
SAVE_VIDEO=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Evaluate a Kimodo checkpoint on a SIMPLE G1 task.

Usage:
  simple_eval_signal_task_sonic.sh [options]

Core options:
  --task NAME               SIMPLE task name, with or without simple/ prefix
  --checkpoint PATH         checkpoint directory (config.json + training_state.pt)
  --gpus LIST               comma/space-separated CUDA indices (default: 0)
  --seeds LIST              comma/space-separated episode seeds (default: 0)
  --episodes N              episodes per level (default: 10; official protocol)
  --levels LIST             official levels to run (default: 0,1,2)
  --results-dir PATH        output root (default: eval_results/simple/...)
  --text-cache PATH         data/cache/Simple directory
  --simple-data-dir PATH    SIMPLE data root containing robot/scene assets
  --eval-data-root PATH     root containing <task>/{level,dr-level}-{0,1,2}
  --python PATH             Python environment containing Kimodo and SIMPLE

Inference/simulation options:
  --dtype fp32|bf16         model dtype (default: fp32)
  --diffusion-steps N       DDIM sampling steps (default: 10)
  --execution-frames N      model frames executed per replan; 0 means full chunk
  --rtc 0|1                 real-time chunking (default: 0)
  --rtc-overlap-frames N    RTC overlap in model frames (default: 0)
  --rtc-frozen-frames N     RTC frozen prefix (default: 0)
  --rtc-ramp-power X        RTC ramp exponent (default: 1.0)
  --max-navigation-speed X  fail fast above this planar speed in m/s (default: 1.5)
  --sim-mode NAME           mujoco or mujoco_isaac (default: mujoco_isaac)
  --max-steps N|auto        TimeLimit per episode; auto uses task metadata (default)
  --no-save-video           disable MP4 recording
  --dry-run                 validate and print the launch plan only
  -h, --help                show this help

Example:
  bash evaluation/simple_eval_signal_task_sonic.sh \
    --task G1WholebodyCloseDoorTeleop-v0 \
    --checkpoint /path/to/checkpoint_100000 --gpus 0,1,2 --seeds 0 --episodes 10
EOF
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || { echo "Missing value for $1" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) need_value "$@"; TASK="$2"; shift 2 ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPUS="$2"; shift 2 ;;
    --seeds) need_value "$@"; SEEDS="$2"; shift 2 ;;
    --episodes|--repeats) need_value "$@"; EPISODES="$2"; shift 2 ;;
    --levels) need_value "$@"; LEVELS="$2"; shift 2 ;;
    --results-dir) need_value "$@"; RESULTS_DIR="$2"; shift 2 ;;
    --text-cache) need_value "$@"; TEXT_CACHE="$2"; shift 2 ;;
    --simple-data-dir) need_value "$@"; SIMPLE_DATA_DIR="$2"; shift 2 ;;
    --eval-data-root) need_value "$@"; EVAL_DATA_ROOT="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --dtype) need_value "$@"; DTYPE="$2"; shift 2 ;;
    --diffusion-steps) need_value "$@"; DIFFUSION_STEPS="$2"; shift 2 ;;
    --execution-frames) need_value "$@"; EXECUTION_FRAMES="$2"; shift 2 ;;
    --rtc) need_value "$@"; RTC="$2"; shift 2 ;;
    --rtc-overlap-frames) need_value "$@"; RTC_OVERLAP_FRAMES="$2"; shift 2 ;;
    --rtc-frozen-frames) need_value "$@"; RTC_FROZEN_FRAMES="$2"; shift 2 ;;
    --rtc-ramp-power) need_value "$@"; RTC_RAMP_POWER="$2"; shift 2 ;;
    --max-navigation-speed) need_value "$@"; MAX_NAVIGATION_SPEED="$2"; shift 2 ;;
    --sim-mode) need_value "$@"; SIM_MODE="$2"; shift 2 ;;
    --max-steps) need_value "$@"; MAX_STEPS="$2"; shift 2 ;;
    --no-save-video) SAVE_VIDEO=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || { echo 'Pass --checkpoint /path/to/checkpoint_STEP (see evaluation/simple_eval.md).' >&2; exit 2; }
[[ "$TASK" == simple/* ]] || TASK="simple/${TASK}"
CHECKPOINT="$(readlink -m "$CHECKPOINT")"
TEXT_CACHE="$(readlink -m "$TEXT_CACHE")"
TASK_NAME="${TASK#simple/}"
if [[ -n "$SIMPLE_DATA_DIR" ]]; then
  SIMPLE_DATA_DIR="$(readlink -m "$SIMPLE_DATA_DIR")"
  [[ -d "$SIMPLE_DATA_DIR" ]] || {
    echo "SIMPLE data directory is missing: $SIMPLE_DATA_DIR" >&2
    exit 3
  }
  export SIMPLE_DATA_DIR
fi

if [[ -n "$MODEL_SITE_PACKAGES" ]]; then
  MODEL_SITE_PACKAGES="$(readlink -m "$MODEL_SITE_PACKAGES")"
  [[ -d "$MODEL_SITE_PACKAGES" ]] || {
    echo "Model site-packages directory is missing: $MODEL_SITE_PACKAGES" >&2
    exit 3
  }
  export KIMODO_MODEL_SITE_PACKAGES="$MODEL_SITE_PACKAGES"
else
  echo "Warning: set KIMODO_MODEL_SITE_PACKAGES if the active SIMPLE environment lacks transformers or safetensors" >&2
fi

if [[ -z "$EVAL_DATA_ROOT" ]]; then
  [[ -n "$SIMPLE_DATA_DIR" ]] || {
    echo "--eval-data-root is required when --simple-data-dir is not set" >&2
    exit 3
  }
  EVAL_DATA_ROOT="$SIMPLE_DATA_DIR/simple-eval"
fi
EVAL_DATA_ROOT="$(readlink -m "$EVAL_DATA_ROOT")"
[[ -d "$EVAL_DATA_ROOT" ]] || {
  echo "SIMPLE eval data root is missing: $EVAL_DATA_ROOT" >&2
  exit 3
}

# Isaac Sim keeps process-global locks under ~/.cache/ov by default.  Running
# the three official levels concurrently therefore needs a private cache per
# GPU, just like SIMPLE's replay workers.  Keep this on the data disk and allow
# callers to override it for another filesystem.
if [[ -z "$ISAAC_CACHE_ROOT" ]]; then
  ISAAC_CACHE_ROOT="${SIMPLE_DATA_DIR:-${PROJECT_ROOT}}/isaac-cache/eval-workers/$TASK_NAME"
fi
ISAAC_CACHE_ROOT="$(readlink -m "$ISAAC_CACHE_ROOT")"
mkdir -p "$ISAAC_CACHE_ROOT"

[[ -d "$CHECKPOINT" ]] || { echo "Checkpoint directory is missing: $CHECKPOINT" >&2; exit 3; }
[[ -f "$CHECKPOINT/config.json" ]] || { echo "Missing checkpoint config.json: $CHECKPOINT" >&2; exit 3; }
[[ -f "$CHECKPOINT/training_state.pt" ]] || { echo "Missing checkpoint training_state.pt: $CHECKPOINT" >&2; exit 3; }
[[ -d "$TEXT_CACHE" ]] || { echo "Simple text cache directory is missing: $TEXT_CACHE" >&2; exit 3; }
[[ -x "$PYTHON_BIN" ]] || {
  echo "Python executable is missing: $PYTHON_BIN (set KIMODO_PYTHON or --python)" >&2
  exit 3
}

for value_name in EPISODES DIFFUSION_STEPS EXECUTION_FRAMES RTC_OVERLAP_FRAMES RTC_FROZEN_FRAMES RENDER_HZ; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "$value_name must be a non-negative integer, got $value" >&2; exit 2; }
done
if [[ "$MAX_STEPS" != auto ]]; then
  [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "MAX_STEPS must be a non-negative integer or auto, got $MAX_STEPS" >&2; exit 2; }
  [[ "$MAX_STEPS" -gt 0 ]] || { echo "--max-steps must be greater than zero" >&2; exit 2; }
fi
[[ "$EPISODES" -gt 0 ]] || { echo "--episodes must be greater than zero" >&2; exit 2; }
[[ "$DIFFUSION_STEPS" -gt 0 ]] || { echo "--diffusion-steps must be greater than zero" >&2; exit 2; }
[[ "$RTC" == 0 || "$RTC" == 1 ]] || { echo "--rtc must be 0 or 1" >&2; exit 2; }
(( RTC_FROZEN_FRAMES <= RTC_OVERLAP_FRAMES )) || { echo "--rtc-frozen-frames cannot exceed --rtc-overlap-frames" >&2; exit 2; }
[[ "$DTYPE" == fp32 || "$DTYPE" == bf16 ]] || { echo "--dtype must be fp32 or bf16" >&2; exit 2; }
"$PYTHON_BIN" - "$RTC_RAMP_POWER" <<'PY_VALIDATE'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("--rtc-ramp-power must be finite and positive")
PY_VALIDATE
"$PYTHON_BIN" - "$MAX_NAVIGATION_SPEED" <<'PY_VALIDATE_SPEED'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("--max-navigation-speed must be finite and positive")
PY_VALIDATE_SPEED

normalize_list() {
  local value="$1"
  value="${value#[}"
  value="${value%]}"
  value="${value//,/ }"
  echo "$value"
}
GPUS="$(normalize_list "$GPUS")"
SEEDS="$(normalize_list "$SEEDS")"
LEVELS="$(normalize_list "$LEVELS")"
read -r -a GPU_ARRAY <<< "$GPUS"
read -r -a SEED_ARRAY <<< "$SEEDS"
read -r -a LEVEL_ARRAY <<< "$LEVELS"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
[[ ${#SEED_ARRAY[@]} -gt 0 ]] || { echo "No seeds specified" >&2; exit 2; }
[[ ${#LEVEL_ARRAY[@]} -gt 0 ]] || { echo "No levels specified" >&2; exit 2; }
for gpu in "${GPU_ARRAY[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 2; }; done
for seed in "${SEED_ARRAY[@]}"; do [[ "$seed" =~ ^-?[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }; done
for level in "${LEVEL_ARRAY[@]}"; do [[ "$level" =~ ^[012]$ ]] || { echo "Invalid SIMPLE level: $level (expected 0, 1 or 2)" >&2; exit 2; }; done

resolve_level_dir() {
  local level="$1"
  local candidate
  for candidate in \
    "$EVAL_DATA_ROOT/$TASK_NAME/level-$level" \
    "$EVAL_DATA_ROOT/$TASK_NAME/dr-level-$level" \
    "$EVAL_DATA_ROOT/simple/$TASK_NAME/level-$level" \
    "$EVAL_DATA_ROOT/simple/$TASK_NAME/dr-level-$level"; do
    if [[ -f "$candidate/meta/episodes.jsonl" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "Missing eval split for $TASK_NAME level-$level under $EVAL_DATA_ROOT" >&2
  return 1
}

declare -A LEVEL_DIRS
for level in "${LEVEL_ARRAY[@]}"; do
  LEVEL_DIRS["$level"]="$(resolve_level_dir "$level")"
done

if [[ -z "$RESULTS_DIR" ]]; then
  checkpoint_name="$(basename "$CHECKPOINT")"
  RESULTS_DIR="${PROJECT_ROOT}/eval_results/simple/${TASK_NAME}_${checkpoint_name}_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_DIR="$(readlink -m "$RESULTS_DIR")"

cat <<EOF
SIMPLE Kimodo evaluation plan
  task:             $TASK
  checkpoint:       $CHECKPOINT
  python:           $PYTHON_BIN
  GPUs:             ${GPU_ARRAY[*]}
  seeds:            ${SEED_ARRAY[*]}
  episodes/seed:    $EPISODES
  levels:           ${LEVEL_ARRAY[*]}
  dtype:            $DTYPE
  diffusion steps:  $DIFFUSION_STEPS
  execution frames: $EXECUTION_FRAMES
  RTC:              $RTC (overlap=$RTC_OVERLAP_FRAMES frozen=$RTC_FROZEN_FRAMES power=$RTC_RAMP_POWER)
  max nav speed:    $MAX_NAVIGATION_SPEED m/s
  sim mode:         $SIM_MODE
  max steps:        $MAX_STEPS
  simple data:      ${SIMPLE_DATA_DIR:-package default}
  eval data root:   $EVAL_DATA_ROOT
  save video:       $SAVE_VIDEO
  results:          $RESULTS_DIR
EOF

if [[ "$DRY_RUN" == 1 ]]; then
  echo "Dry run passed; no evaluation was launched."
  exit 0
fi

mkdir -p "$RESULTS_DIR"
{
  printf 'task=%s\ncheckpoint=%s\ndtype=%s\ndiffusion_steps=%s\nexecution_frames=%s\nrtc=%s\nsim_mode=%s\nmax_steps=%s\nmax_navigation_speed=%s\nepisodes_per_level=%s\nlevels=%s\nsimple_data_dir=%s\neval_data_root=%s\nmodel_site_packages=%s\nstarted_at=%s\n' \
    "$TASK" "$CHECKPOINT" "$DTYPE" "$DIFFUSION_STEPS" "$EXECUTION_FRAMES" "$RTC" "$SIM_MODE" "$MAX_STEPS" "$MAX_NAVIGATION_SPEED" "$EPISODES" "${LEVEL_ARRAY[*]}" "${SIMPLE_DATA_DIR:-}" "$EVAL_DATA_ROOT" "$MODEL_SITE_PACKAGES" "$(date -Iseconds)"
  for level in "${LEVEL_ARRAY[@]}"; do
    printf 'level_%s_data_dir=%s\n' "$level" "${LEVEL_DIRS[$level]}"
  done
} \
  > "$RESULTS_DIR/run_config.txt"

run_one() {
  local level="$1" gpu="$2" seed="$3"
  local level_dir="$RESULTS_DIR/level_${level}"
  local worker_cache="$ISAAC_CACHE_ROOT/gpu_${gpu}"
  local torch_extensions_root="${SIMPLE_DATA_DIR:-${PROJECT_ROOT}}/torch-extensions"
  # Reuse the shared CuRobo extension cache when it is already populated.  A
  # fresh per-GPU cache is only needed for the first compile; otherwise every
  # Isaac smoke test would rebuild the same NFS-hosted CUDA extensions.
  if [[ -d "$torch_extensions_root/kinematics_fused_cu" ]]; then
    torch_extensions_root="$torch_extensions_root"
  else
    torch_extensions_root="$torch_extensions_root/gpu_${gpu}"
  fi
  mkdir -p "$level_dir"
  mkdir -p "$worker_cache/portable" "$worker_cache/warp" "$worker_cache/xdg" "$worker_cache/tmp"
  local video_flag="--no-save-video"
  [[ "$SAVE_VIDEO" == 1 ]] && video_flag="--save-video"
  local -a max_steps_args=()
  [[ "$MAX_STEPS" == auto ]] || max_steps_args=(--max-episode-steps "$MAX_STEPS")
  local model_device="cuda:0"
  local cuda_arch="${TORCH_CUDA_ARCH_LIST:-}"
  if [[ -z "$cuda_arch" ]]; then
    cuda_arch="$(env -u CUDA_VISIBLE_DEVICES "$PYTHON_BIN" -c 'import sys, torch; print("%d.%d" % torch.cuda.get_device_capability(int(sys.argv[1])))' "$gpu")"
  fi
  local -a env_args=(
    "MUJOCO_GL=egl"
    "OMNI_KIT_ACCEPT_EULA=Y"
    "PATH=${PROJECT_ROOT}/SIMPLE/.venv/bin:${PATH}"
    "SIMPLE_ISAAC_PORTABLE_ROOT=${worker_cache}/portable"
    "XDG_CACHE_HOME=${worker_cache}/xdg"
    "WARP_CACHE_PATH=${worker_cache}/warp"
    "TMPDIR=${worker_cache}/tmp"
    "TORCH_CUDA_ARCH_LIST=$cuda_arch"
    "TORCH_EXTENSIONS_DIR=${torch_extensions_root}"
    "KIMODO_MODEL_SITE_PACKAGES=${MODEL_SITE_PACKAGES}"
    "PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/SIMPLE/src:$PROJECT_ROOT/SIMPLE/third_party:$PROJECT_ROOT/SIMPLE/third_party/curobo/src:$PROJECT_ROOT/SIMPLE/third_party/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
  )
  if [[ "$SIM_MODE" == *isaac* ]]; then
    # Isaac Sim must see the physical card index.  Masking it with
    # CUDA_VISIBLE_DEVICES makes Kit attach to an invalid device on some
    # multi-GPU hosts, so use its explicit GPU selectors instead.  Kimodo is
    # pointed at the same physical card rather than the remapped cuda:0.
    env_args+=(
      "SIMPLE_ISAAC_GPU=$gpu"
      "SIMPLE_ISAAC_ACTIVE_GPU=$gpu"
      "SIMPLE_ISAAC_PHYSICS_GPU=$gpu"
      "SIMPLE_ISAAC_MAX_GPU_COUNT=1"
      "SIMPLE_ISAAC_NO_CUDA_MASK=1"
      "SIMPLE_ISAAC_ALLOW_ZERO_GPU_COUNT=1"
    )
    model_device="cuda:$gpu"
    env -u CUDA_VISIBLE_DEVICES "${env_args[@]}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/simple_server.py" \
      --eval --task "$TASK" --checkpoint "$CHECKPOINT" \
      --text-embedding-cache "$TEXT_CACHE" --device "$model_device" \
      --dtype "$DTYPE" --diffusion-steps "$DIFFUSION_STEPS" \
      --execution-frames "$EXECUTION_FRAMES" --rtc "$RTC" \
      --rtc-overlap-frames "$RTC_OVERLAP_FRAMES" --rtc-frozen-frames "$RTC_FROZEN_FRAMES" \
      --rtc-ramp-power "$RTC_RAMP_POWER" --max-navigation-speed "$MAX_NAVIGATION_SPEED" --sim-mode "$SIM_MODE" --headless \
      "${max_steps_args[@]}" --num-episodes "$EPISODES" --seeds "$seed" \
      --eval-data-dir "${LEVEL_DIRS[$level]}" --level "$level" \
      --results-dir "$level_dir" "$video_flag" \
      > "$level_dir/pipeline.log" 2>&1
  else
    env_args+=("CUDA_VISIBLE_DEVICES=$gpu")
    env "${env_args[@]}" \
      "$PYTHON_BIN" "$SCRIPT_DIR/simple_server.py" \
      --eval --task "$TASK" --checkpoint "$CHECKPOINT" \
      --text-embedding-cache "$TEXT_CACHE" --device "$model_device" \
      --dtype "$DTYPE" --diffusion-steps "$DIFFUSION_STEPS" \
      --execution-frames "$EXECUTION_FRAMES" --rtc "$RTC" \
      --rtc-overlap-frames "$RTC_OVERLAP_FRAMES" --rtc-frozen-frames "$RTC_FROZEN_FRAMES" \
      --rtc-ramp-power "$RTC_RAMP_POWER" --max-navigation-speed "$MAX_NAVIGATION_SPEED" --sim-mode "$SIM_MODE" --headless \
      "${max_steps_args[@]}" --num-episodes "$EPISODES" --seeds "$seed" \
      --eval-data-dir "${LEVEL_DIRS[$level]}" --level "$level" \
      --results-dir "$level_dir" "$video_flag" \
      > "$level_dir/pipeline.log" 2>&1
  fi
}

failures=0
eval_seed="${SEED_ARRAY[0]}"
for ((offset=0; offset<${#LEVEL_ARRAY[@]}; offset+=${#GPU_ARRAY[@]})); do
  pids=()
  labels=()
  for ((slot=0; slot<${#GPU_ARRAY[@]}; slot++)); do
    index=$((offset + slot))
    (( index < ${#LEVEL_ARRAY[@]} )) || break
    level="${LEVEL_ARRAY[$index]}"
    gpu="${GPU_ARRAY[$slot]}"
    echo "Launching level=$level seed=$eval_seed on GPU=$gpu"
    run_one "$level" "$gpu" "$eval_seed" &
    pids+=("$!")
    labels+=("level=$level,gpu=$gpu")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "Completed ${labels[$index]}"
    else
      failed_level="${LEVEL_ARRAY[$((offset + index))]}"
      echo "Failed ${labels[$index]} (see ${RESULTS_DIR}/level_${failed_level}/pipeline.log)" >&2
      failures=$((failures + 1))
    fi
  done
done

"$PYTHON_BIN" - "$RESULTS_DIR" <<'PY_AGGREGATE'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
items = []
for path in sorted(root.glob("level_*/summary.json")):
    data = json.loads(path.read_text())
    level_name = path.parent.name.removeprefix("level_")
    items.append({"level": int(level_name), "episodes": data.get("episodes", 0), "successes": data.get("successes", 0), "success_rate": data.get("success_rate", 0.0), "eval_data_dir": data.get("eval_data_dir")})
episodes = sum(int(item["episodes"]) for item in items)
successes = sum(int(item["successes"]) for item in items)
summary = {"episodes": episodes, "successes": successes, "success_rate": successes / episodes if episodes else 0.0, "per_level": items}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY_AGGREGATE

if (( failures > 0 )); then
  echo "Evaluation finished with $failures failed level job(s): $RESULTS_DIR" >&2
  exit 1
fi
date -Iseconds > "$RESULTS_DIR/.done"
echo "Evaluation complete: $RESULTS_DIR"
