#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
if pgrep -f '[g]1_deploy_onnx_ref|[p]ico_manager_thread_server.py' >/dev/null; then
 echo 'Existing SONIC or PICO manager detected; finish its session first.' >&2; exit 2
fi
DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${KIMODO_DIAGNOSTICS_ROOT:-$DEPLOY_DIR/logs}/$(date +%Y%m%d-%H%M%S)-sonic-$$"
mkdir -p "$RUN_DIR"
echo "SONIC diagnostics: $RUN_DIR/console.log"
# Keep stdin on the user's terminal so keyboard emergency controls still work.
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
: "${COLLECTION_ROOT:?Set COLLECTION_ROOT to the configured workstation root}"
exec "$REAL_WORLD_ROOT/collection/bin/sonic-control" "$@"
