#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${KIMODO_ENV:-}" ]]; then
  ACCELERATE_BIN="${KIMODO_ENV}/bin/accelerate"
else
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate was not found; activate kimodo-policy or set KIMODO_ENV=/path/to/env" >&2
  exit 1
fi

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PWD}/${CONFIG_PATH}"
fi

# Real-world training is initialized from a compatible Kimodo checkpoint.
# Accept either the first positional argument or KIMODO_INIT_CHECKPOINT.
INIT_CHECKPOINT="${1:-${KIMODO_INIT_CHECKPOINT:-}}"
if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "Usage: bash $0 /path/to/pretrain/checkpoint_STEP [dataset_name_or_glob]" >&2
  echo "Or set KIMODO_INIT_CHECKPOINT=/path/to/pretrain/checkpoint_STEP" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}/training_state.pt" || ! -f "${INIT_CHECKPOINT}/config.json" ]]; then
  echo "Initialization checkpoint is incomplete: ${INIT_CHECKPOINT}" >&2
  echo "Expected training_state.pt and config.json in that directory." >&2
  exit 2
fi
if [[ "${INIT_CHECKPOINT}" != /* ]]; then
  INIT_CHECKPOINT="$(cd -- "${INIT_CHECKPOINT}" && pwd)"
fi

# The second positional argument overrides REAL_WORLD_DATASET.  The default
# wildcard lets one run across every compatible dataset below REAL_WORLD_ROOT.
DATASET_NAME="${2:-${REAL_WORLD_DATASET:-*}}"
if [[ -z "${DATASET_NAME}" ]]; then
  echo "REAL_WORLD_DATASET must be a non-empty folder name or glob" >&2
  exit 2
fi
export REAL_WORLD_DATASET="${DATASET_NAME}"
if [[ "${DATASET_NAME}" == "*" ]]; then
  DATASET_TAG="all"
else
  DATASET_TAG="${DATASET_NAME//[^[:alnum:]_.-]/_}"
  DATASET_TAG="${DATASET_TAG:-dataset}"
fi
export KIMODO_FT_SAVE_ROOT="${KIMODO_FT_SAVE_ROOT:-log/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_${DATASET_TAG}/}"

GPU_LIST="${KIMODO_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
NUM_PROCESSES="${#GPU_IDS[@]}"
if (( NUM_PROCESSES < 1 )); then
  echo "KIMODO_GPUS must contain at least one GPU index" >&2
  exit 2
fi

CHECKPOINT_FLAG="--init-checkpoint"
if [[ "${KIMODO_RESUME:-0}" == "1" ]]; then
  CHECKPOINT_FLAG="--resume"
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Real-world dataset: ${DATASET_NAME}"
echo "Save root: ${KIMODO_FT_SAVE_ROOT}"

cd "${PROJECT_ROOT}"
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29653}" \
  train.py \
  --config "${CONFIG_PATH}" \
  "${CHECKPOINT_FLAG}" "${INIT_CHECKPOINT}"

# Example:
# conda activate kimodo-policy
# export KIMODO_ENV="$CONDA_PREFIX"
# export REAL_WORLD_ROOT=real-world
# export REAL_WORLD_DATASET=01_SitSofa
# KIMODO_GPUS=4,5,6,7 \
# KIMODO_INIT_CHECKPOINT=/path/to/pretrain/checkpoint_STEP \
# bash scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.sh
