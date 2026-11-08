#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Fine-tune the 4-layer ControlNet continuous-hand model on one SIMPLE task.

Usage:
  bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh TASK [CHECKPOINT]
  bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh CHECKPOINT TASK

TASK defaults to KIMODO_SIMPLE_TASK. CHECKPOINT is the initialization checkpoint
(config.json + training_state.pt), supplied as the second argument or with
KIMODO_INIT_CHECKPOINT.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Accept both TASK CHECKPOINT and CHECKPOINT TASK, matching the HumanoidArena
# fine-tuning launchers. A checkpoint is unambiguously a directory containing
# the two files below, while a SIMPLE task is a directory below SIMPLE_ROOT.
SIMPLE_ROOT="${KIMODO_SIMPLE_ROOT:-datasets/Simple}"
if [[ "${SIMPLE_ROOT}" != /* ]]; then
  SIMPLE_ROOT="${PROJECT_ROOT}/${SIMPLE_ROOT}"
fi
if [[ -n "${1:-}" && -d "${1}" && -f "${1}/training_state.pt" && -f "${1}/config.json" ]]; then
  INIT_CHECKPOINT="${1}"
  TASK="${2:-${KIMODO_SIMPLE_TASK:-}}"
else
  TASK="${1:-${KIMODO_SIMPLE_TASK:-}}"
  INIT_CHECKPOINT="${2:-${KIMODO_INIT_CHECKPOINT:-}}"
fi

if [[ -z "${TASK}" || "${TASK}" == */* ]]; then
  echo "TASK must be one SIMPLE task directory name, got: ${TASK}" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "${SIMPLE_ROOT}" ]]; then
  echo "SIMPLE dataset root does not exist or is not a directory: ${SIMPLE_ROOT}" >&2
  echo "Set KIMODO_SIMPLE_ROOT to the downloaded SIMPLE dataset root." >&2
  exit 2
fi
if [[ ! -d "${SIMPLE_ROOT}/${TASK}" ]]; then
  echo "SIMPLE task directory does not exist: ${SIMPLE_ROOT}/${TASK}" >&2
  exit 2
fi
if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "An initialization checkpoint is required." >&2
  echo "Usage: bash $0 TASK /path/to/checkpoint_STEP" >&2
  echo "Or set KIMODO_INIT_CHECKPOINT=/path/to/checkpoint_STEP" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}/training_state.pt" || ! -f "${INIT_CHECKPOINT}/config.json" ]]; then
  echo "Initialization checkpoint is incomplete: ${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${INIT_CHECKPOINT}" != /* ]]; then
  INIT_CHECKPOINT="$(cd -- "${INIT_CHECKPOINT}" && pwd)"
fi

if [[ -n "${KIMODO_ENV:-}" ]]; then
  ACCELERATE_BIN="${KIMODO_ENV}/bin/accelerate"
else
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate was not found; activate the training environment or set KIMODO_ENV=/path/to/env" >&2
  exit 1
fi

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PWD}/${CONFIG_PATH}"
fi

GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "KIMODO_GPUS must be comma-separated GPU indices" >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

RUN_NAME="ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_${TASK}"
export KIMODO_SIMPLE_TASK="${TASK}"
export KIMODO_SIMPLE_ROOT="${SIMPLE_ROOT}"
export KIMODO_SIMPLE_SAVE_ROOT="${KIMODO_SIMPLE_SAVE_ROOT:-log/Simple_Single_Task/${RUN_NAME}/}"
export KIMODO_SIMPLE_RUN_NAME="${KIMODO_SIMPLE_RUN_NAME:-${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "SIMPLE task: ${TASK}"
echo "SIMPLE root: ${SIMPLE_ROOT}"
echo "GPUs: ${GPU_LIST}"
echo "Config: ${CONFIG_PATH}"
echo "Initialization checkpoint: ${INIT_CHECKPOINT}"
echo "Save root: ${KIMODO_SIMPLE_SAVE_ROOT}"

cd "${PROJECT_ROOT}"
MULTI_GPU_ARGS=()
(( NUM_PROCESSES <= 1 )) || MULTI_GPU_ARGS=(--multi_gpu)
"${ACCELERATE_BIN}" launch \
  "${MULTI_GPU_ARGS[@]}" \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29665}" \
  train.py \
  --config "${CONFIG_PATH}" \
  --init-checkpoint "${INIT_CHECKPOINT}"

# conda activate kimodo-env
# export KIMODO_ENV="$CONDA_PREFIX"
# KIMODO_GPUS=0,1,2,3 \
# KIMODO_SIMPLE_ROOT=/path/to/SIMPLE \
# bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh \
#   G1WholebodyXMovePickTeleop-v0 /path/to/pretrain/checkpoint_STEP
