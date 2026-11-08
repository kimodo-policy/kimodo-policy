#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Train the 4-layer ControlNet single-task experiment.

Usage:
  bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh TASK [BACKEND]

BACKEND defaults to sonic. TASK accepts the dataset task name or a short alias:
  doubledesk, football, grap_cup, pp_box, boxing, open_door, sit_sofa, vision_navi
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK_INPUT="${1:-${KIMODO_ARENA_TASK:-}}"
BACKEND="${2:-${KIMODO_ARENA_BACKEND:-sonic}}"
if [[ -z "${TASK_INPUT}" ]]; then
  usage >&2
  exit 2
fi

case "${TASK_INPUT,,}" in
  doubledesk|double_desk|double-desk|hoi_double_desk) ARENA_TASK="HOI_double_desk" ;;
  football|hoi_football) ARENA_TASK="HOI_football" ;;
  grap_cup|grap-cup|grapcup|grab_cup|grab-cup|grabcup|hoi_grap_cup) ARENA_TASK="HOI_grap_cup" ;;
  pp_box|pp-box|pickplace_box|pickplace-box|hoi_pp_box) ARENA_TASK="HOI_pp_box" ;;
  boxing|hsi_boxing) ARENA_TASK="HSI_boxing" ;;
  open_door|open-door|opendoor|hsi_open_door) ARENA_TASK="HSI_open_door" ;;
  sit_sofa|sit-sofa|sitsofa|hsi_sit_sofa) ARENA_TASK="HSI_sit_sofa" ;;
  vision_navi|vision-navi|vision_navigation|vision-navigation|hsi_vision_navi) ARENA_TASK="HSI_vision_navi" ;;
  *)
    echo "Unsupported HumanoidArena task: ${TASK_INPUT}" >&2
    usage >&2
    exit 2
    ;;
esac

BACKEND="${BACKEND,,}"
if [[ "${BACKEND}" != "sonic" && "${BACKEND}" != "twist2" ]]; then
  echo "BACKEND must be sonic or twist2, got: ${BACKEND}" >&2
  exit 2
fi

if [[ -n "${KIMODO_ENV:-}" ]]; then
  ACCELERATE_BIN="${KIMODO_ENV}/bin/accelerate"
else
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate was not found; activate kimodo-policy or set KIMODO_ENV=/path/to/env" >&2
  exit 1
fi

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi

GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

RUN_NAME="humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_${ARENA_TASK}_${BACKEND}"
export KIMODO_ARENA_TASK="${ARENA_TASK}"
export KIMODO_ARENA_BACKEND="${BACKEND}"
export KIMODO_SINGLE_TASK_SAVE_ROOT="${KIMODO_SINGLE_TASK_SAVE_ROOT:-log/HumanoidArena_Single_Task/${RUN_NAME}/}"
export KIMODO_SINGLE_TASK_RUN_NAME="${KIMODO_SINGLE_TASK_RUN_NAME:-${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "HumanoidArena task: ${ARENA_TASK}"
echo "Backend: ${BACKEND}"
echo "GPUs: ${GPU_LIST}"
echo "Config: ${CONFIG_PATH}"
echo "Save root: ${KIMODO_SINGLE_TASK_SAVE_ROOT}"

cd "${PROJECT_ROOT}"
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29658}" \
  train.py \
  --config "${CONFIG_PATH}"

# conda activate kimodo-policy
# export KIMODO_ENV="$CONDA_PREFIX"
# KIMODO_GPUS=0,1,2,3 \
# HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
# bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh doubledesk sonic
