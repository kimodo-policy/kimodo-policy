#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
: "${KIMODO_SERVER_SSH:?Set KIMODO_SERVER_SSH to your SSH host alias or user@host}"
# Use the same config/environment port resolution as server and bridge.
port="$(python3 - "$REAL_WORLD_ROOT/deploy" <<'PYPORT'
import sys
sys.path.insert(0, sys.argv[1])
from deploy_cli import load_config
print(load_config()['server']['port'])
PYPORT
)"
exec ssh -4 -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
 -L "127.0.0.1:${port}:127.0.0.1:${port}" "$KIMODO_SERVER_SSH"
