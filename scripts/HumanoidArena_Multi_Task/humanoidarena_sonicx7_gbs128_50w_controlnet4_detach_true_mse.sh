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

CONFIG_PATH="${KIMODO_CONFIG:-${SCRIPT_DIR}/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml}"
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

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
"${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${KIMODO_MASTER_PORT:-29659}" \
  train.py \
  --config "${CONFIG_PATH}"

# conda activate kimodo-policy
# export KIMODO_ENV="$CONDA_PREFIX"
# KIMODO_GPUS=0,1,2,3 \
# HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
# bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh
