#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Fine-tune the 4-layer ControlNet single-task experiment.

Usage:
  bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh TASK [BACKEND] [CHECKPOINT]
  bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh CHECKPOINT TASK [BACKEND]

BACKEND defaults to sonic. TASK accepts the dataset task name or a short alias:
  doubledesk, football, grap_cup, pp_box, boxing, open_door, sit_sofa, vision_navi

The initialization checkpoint can be supplied as the third argument (or first
argument in the alternate form) or with KIMODO_INIT_CHECKPOINT. It must contain
training_state.pt and config.json.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${1:-}" && -d "${1}" ]]; then
  # Also accept the positional order used by the multi-task FT launcher.
  INIT_CHECKPOINT="${1}"
  TASK_INPUT="${2:-${KIMODO_ARENA_TASK:-}}"
  BACKEND="${3:-${KIMODO_ARENA_BACKEND:-sonic}}"
else
  TASK_INPUT="${1:-${KIMODO_ARENA_TASK:-}}"
  BACKEND="${2:-${KIMODO_ARENA_BACKEND:-sonic}}"
  INIT_CHECKPOINT="${3:-${KIMODO_INIT_CHECKPOINT:-}}"
fi

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

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.yaml}"
GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PWD}/${CONFIG_PATH}"
fi

if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "Usage: bash $0 TASK [BACKEND] /path/to/pretrain/checkpoint_STEP" >&2
  echo "Or set KIMODO_INIT_CHECKPOINT=/path/to/pretrain/checkpoint_STEP" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}/training_state.pt" || ! -f "${INIT_CHECKPOINT}/config.json" ]]; then
  echo "Initialization checkpoint is incomplete: ${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${INIT_CHECKPOINT}" != /* ]]; then
  INIT_CHECKPOINT="$(cd -- "${INIT_CHECKPOINT}" && pwd)"
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

RUN_NAME="ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_${ARENA_TASK}_${BACKEND}"
export KIMODO_ARENA_TASK="${ARENA_TASK}"
export KIMODO_ARENA_BACKEND="${BACKEND}"
export KIMODO_FT_SAVE_ROOT="${KIMODO_FT_SAVE_ROOT:-log/HumanoidArena_Single_Task/${RUN_NAME}/}"
export KIMODO_FT_RUN_NAME="${KIMODO_FT_RUN_NAME:-${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "HumanoidArena task: ${ARENA_TASK}"
echo "Backend: ${BACKEND}"
echo "GPUs: ${GPU_LIST}"
echo "Config: ${CONFIG_PATH}"
echo "Initialization checkpoint: ${INIT_CHECKPOINT}"
echo "Save root: ${KIMODO_FT_SAVE_ROOT}"

cd "${PROJECT_ROOT}"
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29664}" \
  train.py \
  --config "${CONFIG_PATH}" \
  --init-checkpoint "${INIT_CHECKPOINT}"

# conda activate kimodo-policy
# export KIMODO_ENV="$CONDA_PREFIX"
# KIMODO_GPUS=0,1,2,3 \
# HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
# bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
#   doubledesk sonic /path/to/pretrain/checkpoint_STEP
