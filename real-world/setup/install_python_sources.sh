#!/usr/bin/env bash
# Conda dependencies are assumed prepared; install relocated editable sources only.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
: "${TELEOP_PYTHON:?Set TELEOP_PYTHON}"
: "${COLLECTION_PYTHON:?Set COLLECTION_PYTHON}"
"$TELEOP_PYTHON" -m pip install --no-deps --no-build-isolation -e "$WBC_ROOT/gear_sonic" -e "$GMR_ROOT"
export PATH="$(dirname "$TELEOP_PYTHON"):$PATH"
export CMAKE_PREFIX_PATH="$("$TELEOP_PYTHON" -m pybind11 --cmakedir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
"$TELEOP_PYTHON" -m pip install --no-deps --no-build-isolation -e "$WBC_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
"$COLLECTION_PYTHON" -m pip install --no-deps --no-build-isolation -e "$WBC_ROOT/gear_sonic"
"$TELEOP_PYTHON" "$REAL_WORLD_ROOT/tools/check_environment.py" teleop
"$COLLECTION_PYTHON" "$REAL_WORLD_ROOT/tools/check_environment.py" collection
