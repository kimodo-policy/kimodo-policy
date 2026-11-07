#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
unset PYTHONPATH LD_LIBRARY_PATH
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
export ROBOT_INTERFACE="${ROBOT_INTERFACE:-eno1}"
export CAMERA_HOST="${CAMERA_HOST:-}"
[[ -d "$WBC_ROOT" ]] || { echo 'Install the workstation runtime bundle first.' >&2; exit 2; }
cd "$WBC_ROOT"
