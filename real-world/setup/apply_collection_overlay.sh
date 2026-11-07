#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
[[ -f "$WBC_ROOT/gear_sonic_deploy/CMakeLists.txt" ]] || { echo 'Missing full SONIC runtime source' >&2; exit 2; }
# Only apply to a fresh bundle extraction; never use this against a working checkout.
cp -a "$REAL_WORLD_ROOT/collection/overlay/GR00T-WholeBodyControl/." "$WBC_ROOT/"
echo 'Reviewed collection overlay installed.'
