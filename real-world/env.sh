#!/usr/bin/env bash
# Source this file in every terminal. local.env belongs to this machine only.
export REAL_WORLD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${KIMODO_WORKSTATION_INSTALLING:-0}" != 1 ]]; then
  if [[ -f "$REAL_WORLD_ROOT/.state/workstation.env" ]]; then source "$REAL_WORLD_ROOT/.state/workstation.env"; fi
  if [[ -f "$REAL_WORLD_ROOT/local.env" ]]; then source "$REAL_WORLD_ROOT/local.env"; fi
fi
export COLLECTION_ROOT="${COLLECTION_ROOT:-$REAL_WORLD_ROOT/runtime}"
export WBC_ROOT="${WBC_ROOT:-$REAL_WORLD_ROOT/external/GR00T-WholeBodyControl}"
export GMR_ROOT="${GMR_ROOT:-$REAL_WORLD_ROOT/external/GMR}"
export SONIC_NATIVE_ROOT="${SONIC_NATIVE_ROOT:-$COLLECTION_ROOT/native}"
export KIMODO_PROJECT_ROOT="${KIMODO_PROJECT_ROOT:-$REAL_WORLD_ROOT/..}"
export KIMODO_DEPLOY_CONFIG="${KIMODO_DEPLOY_CONFIG:-$REAL_WORLD_ROOT/deploy/deploy_config.json}"
export KIMODO_TEXT_CACHE="${KIMODO_TEXT_CACHE:-$REAL_WORLD_ROOT/assets/text-cache}"
export PYTHONNOUSERSITE=1
