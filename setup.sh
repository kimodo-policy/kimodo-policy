#!/usr/bin/env bash
# Install the release environments without changing NVIDIA drivers or CUDA.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$PROJECT_ROOT/.setup"
TARGET=all
DRY_RUN=0
CHECK_ONLY=0
SYSTEM_DEPS=0
DOWNLOAD_RESOURCES=0
ACCEPT_EULA=0
CONDA_BIN="${CONDA_EXE:-}"
STAGE=arguments

usage() {
  cat <<'EOF'
Usage: bash setup.sh [options]

Default: install kimodo-policy, unitree_sim_env, and SIMPLE/.venv.
Requires Linux x86_64, Conda, and an NVIDIA GPU/driver. SIMPLE also needs
a CUDA toolkit with nvcc (set CUDA_HOME if necessary).

  --only all|training|arena|simple  Select environments (default: all).
                                  arena/simple also install kimodo-policy.
  --conda PATH                    Path to the conda executable.
  --system-deps                   Install Ubuntu system packages using apt.
  --accept-isaac-eula              Confirm acceptance of NVIDIA Isaac Sim EULA.
  --download-resources            Download Arena SONIC + complete SIMPLE evaluation assets.
  --dry-run                       Print the plan; do not install or write files.
  --check                         Check existing installations; do not install.
  -h, --help                      Show this help.

After installation: source .setup/env.sh
Model checkpoints, Arena scene assets, and benchmark datasets are separate
from environment installation. Task text embeddings are bundled in Git. See README.md.
EOF
}
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[setup] %s\n' "$*"; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
while (( $# )); do
  case "$1" in
    --only) need_value "$@"; TARGET="$2"; shift 2 ;;
    --conda) need_value "$@"; CONDA_BIN="$2"; shift 2 ;;
    --system-deps) SYSTEM_DEPS=1; shift ;;
    --accept-isaac-eula) ACCEPT_EULA=1; shift ;;
    --download-resources) DOWNLOAD_RESOURCES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done
case "$TARGET" in all|training|arena|simple) ;; *) die "Invalid --only: $TARGET" ;; esac
(( DRY_RUN + CHECK_ONLY <= 1 )) || die '--dry-run and --check are mutually exclusive'
if (( CHECK_ONLY && (SYSTEM_DEPS || DOWNLOAD_RESOURCES) )); then
  die '--check cannot be combined with installation/download options'
fi
WANT_ARENA=0; WANT_SIMPLE=0
[[ "$TARGET" == all || "$TARGET" == arena ]] && WANT_ARENA=1
[[ "$TARGET" == all || "$TARGET" == simple ]] && WANT_SIMPLE=1
plan() {
  log "Repository: $PROJECT_ROOT"
  echo '  kimodo-policy: Python 3.10, torch 2.11.0/cu126, model dependencies'
  (( ! WANT_ARENA )) || echo '  unitree_sim_env: Python 3.11, Isaac Sim 5.0.0, Isaac Lab release/2.2.0'
  (( ! WANT_SIMPLE )) || echo '  SIMPLE/.venv: Python 3.10, locked uv dependencies, Isaac Sim 4.5.0, cuRobo'
  (( ! SYSTEM_DEPS )) || echo '  Ubuntu system packages (sudo required unless running as root)'
  (( ! DOWNLOAD_RESOURCES )) || echo '  Download selected benchmark runtime resources (not training datasets)'
  echo '  Verify imports; write .setup/env.sh only after successful installation'
}
plan
(( ! DRY_RUN )) || exit 0

STAGE=preflight
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'This installer targets Linux x86_64.'
# Prevent inherited Python/uv settings from installing into an unrelated environment.
unset PYTHONHOME PYTHONPATH UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
if [[ -z "$CONDA_BIN" ]]; then CONDA_BIN="$(type -P conda || true)"; fi
if [[ -z "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
[[ -x "$CONDA_BIN" ]] || die 'Install Conda first, or pass --conda /path/to/bin/conda.'
CONDA_BASE="$("$CONDA_BIN" info --base)"
[[ -x "$CONDA_BASE/bin/python" ]] || die "Missing Conda base Python: $CONDA_BASE/bin/python"
command -v nvidia-smi >/dev/null || die 'NVIDIA driver not detected (nvidia-smi is missing).'
nvidia-smi -L >/dev/null || die 'Cannot access an NVIDIA GPU.'
if (( WANT_SIMPLE && ! CHECK_ONLY )); then
  "$CONDA_BASE/bin/python" -c 'import platform; name, version = platform.libc_ver(); assert name == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 34)' || die 'A fresh SIMPLE environment needs glibc >= 2.34 (Ubuntu 22.04 or newer recommended). Existing environments can be inspected with --check.'
  # SIMPLE pins torch/cu128. Prefer a matching toolkit over a system CUDA 13.
  if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -z "${CUDA_HOME:-}" ]] && ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
  command -v nvcc >/dev/null || die 'SIMPLE needs nvcc. Install a CUDA toolkit and set CUDA_HOME, then rerun.'
  nvcc --version | grep -Eq 'release 12\.' || die 'SIMPLE uses PyTorch CUDA 12.8. Set CUDA_HOME to a CUDA 12.x toolkit (12.8 recommended).'
  if [[ -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  fi
  # Recent Conda CUDA toolkits keep headers below targets/ instead of include/.
  if [[ ! -f "$CUDA_HOME/include/cuda_runtime.h" && -d "$CUDA_HOME/targets/x86_64-linux/include" ]]; then
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  fi
fi
if (( (WANT_ARENA || WANT_SIMPLE) && ! ACCEPT_EULA )); then
  die 'Review https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html and rerun with --accept-isaac-eula.'
fi
(( ! ACCEPT_EULA )) || export OMNI_KIT_ACCEPT_EULA=YES
[[ -f "$PROJECT_ROOT/requirements.txt" ]] || die 'Missing training requirements.'
(( ! WANT_ARENA )) || [[ -f "$PROJECT_ROOT/HumanoidArena/isaaclab_twist2_g1/requirements.txt" ]] || die 'Missing vendored HumanoidArena.'
(( ! WANT_SIMPLE )) || [[ -f "$PROJECT_ROOT/SIMPLE/uv.lock" ]] || die 'Missing vendored SIMPLE/uv.lock.'

if (( ! CHECK_ONLY )); then
  mkdir -p "$STATE_DIR"
  LOG_FILE="$STATE_DIR/install-$(date +%Y%m%d-%H%M%S)-$$.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  trap 'rc=$?; printf "\n[setup] Failed during %s (line %s, exit %s). Log: %s\n" "$STAGE" "$LINENO" "$rc" "$LOG_FILE" >&2; exit "$rc"' ERR
  if (( SYSTEM_DEPS )); then
    STAGE=system-packages
    command -v apt-get >/dev/null || die '--system-deps requires Ubuntu/Debian apt-get.'
    SUDO=(); (( EUID == 0 )) || SUDO=(sudo)
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y git git-lfs curl wget unzip cmake build-essential python3-dev ffmpeg gstreamer1.0-libav cyclonedds-dev libegl1 libgl1 libvulkan1 libglib2.0-0 libsm6 libice6 libxrender1
  fi
  for cmd in git cmake curl; do command -v "$cmd" >/dev/null || die "Missing $cmd; rerun with --system-deps."; done
  git lfs version >/dev/null || die 'Missing Git LFS; rerun with --system-deps.'
  git -C "$PROJECT_ROOT" lfs install --local
  # Papers/media are not runtime dependencies; SIMPLE controllers are regular Git files.
  if (( WANT_ARENA )); then
    git -C "$PROJECT_ROOT" lfs pull --include='HumanoidArena/**' --exclude=''
  fi
fi

find_env() {
  "$CONDA_BIN" env list --json | "$CONDA_BASE/bin/python" -c '
import json, os, sys
name=sys.argv[1]
paths=[p for p in json.load(sys.stdin)["envs"] if os.path.basename(p)==name]
if len(paths)>1: raise SystemExit("Ambiguous Conda environment name: "+name)
print(paths[0] if paths else "")' "$1"
}
ensure_env() {
  local name="$1" version="$2" prefix
  prefix="$(find_env "$name")"
  if [[ -z "$prefix" ]]; then
    (( ! CHECK_ONLY )) || die "Missing environment: $name"
    "$CONDA_BIN" create -y -n "$name" "python=$version" >&2
    prefix="$(find_env "$name")"
  fi
  [[ -x "$prefix/bin/python" ]] || die "Missing Python in $name"
  "$prefix/bin/python" -c 'import sys; assert sys.argv[1] == "%s.%s" % sys.version_info[:2], "Existing environment has incompatible Python"' "$version"
  printf '%s\n' "$prefix"
}
STAGE=training-environment
KIMODO_ENV="$(ensure_env kimodo-policy 3.10)"
if (( ! CHECK_ONLY )); then
  log 'Installing Kimodo-Policy dependencies'
  "$KIMODO_ENV/bin/python" -m pip install --upgrade pip
  "$KIMODO_ENV/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
  "$KIMODO_ENV/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
fi
cd "$PROJECT_ROOT"
"$KIMODO_ENV/bin/python" -c 'import train; import evaluation.humanoidarena_server; import torch; assert torch.cuda.is_available(); print("Kimodo training/server imports OK")'
export KIMODO_MODEL_SITE_PACKAGES
KIMODO_MODEL_SITE_PACKAGES="$("$KIMODO_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

KIMODO_SIM_ENV="$(find_env unitree_sim_env)"
if (( WANT_ARENA )); then
  STAGE=arena-environment
  KIMODO_SIM_ENV="$(ensure_env unitree_sim_env 3.11)"
  if (( ! CHECK_ONLY )); then
    "$KIMODO_SIM_ENV/bin/python" -m pip install --upgrade pip
    "$KIMODO_SIM_ENV/bin/python" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    "$KIMODO_SIM_ENV/bin/python" -m pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com
    LAB_DIR="$STATE_DIR/IsaacLab"
    if [[ ! -e "$LAB_DIR" ]]; then
      git clone --branch release/2.2.0 https://github.com/isaac-sim/IsaacLab.git "$LAB_DIR"
    fi
    [[ -d "$LAB_DIR/.git" ]] || die "$LAB_DIR exists but is not an IsaacLab Git checkout."
    [[ "$(git -C "$LAB_DIR" rev-parse HEAD)" == "$(git -C "$LAB_DIR" rev-parse origin/release/2.2.0)" ]] || die 'Unexpected IsaacLab revision; inspect .setup/IsaacLab before rerunning.'
    [[ -z "$(git -C "$LAB_DIR" status --porcelain --untracked-files=no)" ]] || die 'IsaacLab has local edits; inspect .setup/IsaacLab before rerunning.'
    (
      # Isaac Lab's installer selects its Python through CONDA_PREFIX/PATH.
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "$KIMODO_SIM_ENV"
      cd "$LAB_DIR"
      ./isaaclab.sh --install
    )
    "$KIMODO_SIM_ENV/bin/python" -m pip install -r "$PROJECT_ROOT/HumanoidArena/isaaclab_twist2_g1/requirements.txt"
  fi
  "$KIMODO_SIM_ENV/bin/python" -c 'from importlib.metadata import version; assert version("isaacsim") in ("5.0.0", "5.0.0.0"); import isaaclab; print("Arena packages OK")'
fi

if (( WANT_SIMPLE )); then
  STAGE=simple-environment
  "$KIMODO_ENV/bin/python" -c 'import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libvulkan.so.1")' || die 'Missing EGL/Vulkan system libraries. Rerun with --system-deps (and expose NVIDIA graphics libraries when using a container).'
  if (( ! CHECK_ONLY )); then
    # Isolate uv from an activated caller environment. Keep all vendored sources.
    UV_BIN="$(type -P uv || true)"
    if [[ -z "$UV_BIN" ]]; then
      "$KIMODO_ENV/bin/python" -m pip install uv
      UV_BIN="$KIMODO_ENV/bin/uv"
    fi
    export PATH="$(dirname "$UV_BIN"):$PATH"
    export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/SIMPLE/.venv"
    (
      cd "$PROJECT_ROOT/SIMPLE"
      UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 "$UV_BIN" sync --locked --python "$KIMODO_ENV/bin/python" --all-groups --index-strategy unsafe-best-match
      "$UV_BIN" pip install --python .venv/bin/python 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
      if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        TORCH_CUDA_ARCH_LIST="$(.venv/bin/python -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')"
        [[ -n "$TORCH_CUDA_ARCH_LIST" ]] || die 'Cannot detect CUDA architectures. Set TORCH_CUDA_ARCH_LIST.'
        export TORCH_CUDA_ARCH_LIST
      fi
      bash scripts/install_curobo.sh
    )
  fi
  "$PROJECT_ROOT/SIMPLE/.venv/bin/python" -c 'from importlib.metadata import version; assert version("isaacsim") in ("4.5.0", "4.5.0.0"); import simple; from evaluation.simple_server import _add_model_dependency_paths; _add_model_dependency_paths(); import model.kimodo_policy; import curobo.curobolib.kinematics_fused_cu; print("SIMPLE/model/CUDA extension imports OK")'
fi

if (( DOWNLOAD_RESOURCES )); then
  STAGE=runtime-resources
  if (( WANT_ARENA )); then
    "$KIMODO_ENV/bin/hf" download nvidia/GEAR-SONIC model_encoder.onnx model_decoder.onnx observation_config.yaml --local-dir "$PROJECT_ROOT/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
  fi
  if (( WANT_SIMPLE )); then
    "$KIMODO_ENV/bin/python" "$PROJECT_ROOT/utils/prepare_simple.py" --assets
  fi
fi
if (( CHECK_ONLY )); then log 'Requested environment checks passed (no simulation launched).'; exit 0; fi

STAGE=runtime-paths
# Shell-escape every path, including spaces; publish only after all steps succeed.
{
  echo '# Generated by setup.sh. Source this file from Bash.'
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export KIMODO_ENV=%q\n' "$KIMODO_ENV"
  printf 'export KIMODO_SERVER_PYTHON=%q\n' "$KIMODO_ENV/bin/python"
  printf 'export KIMODO_MODEL_SITE_PACKAGES=%q\n' "$KIMODO_MODEL_SITE_PACKAGES"
  if [[ -n "$KIMODO_SIM_ENV" ]]; then printf 'export KIMODO_SIM_ENV=%q\n' "$KIMODO_SIM_ENV"; fi
  printf 'export KIMODO_HUMANOIDARENA_ROOT=%q\n' "$PROJECT_ROOT/HumanoidArena"
  printf 'export SONIC_POLICY_ROOT=%q\n' "$PROJECT_ROOT/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
  printf 'export TEXT_EMBEDDING_CACHE=%q\n' "$PROJECT_ROOT/data/cache/HumanoidArena"
  printf 'export KIMODO_PYTHON=%q\n' "$PROJECT_ROOT/SIMPLE/.venv/bin/python"
  printf 'export SIMPLE_DATA_DIR=%q\n' "$PROJECT_ROOT/SIMPLE/data"
  printf 'export EVAL_DATA_ROOT=%q\n' "$PROJECT_ROOT/SIMPLE/data/simple-eval"
  if [[ -n "${CUDA_HOME:-}" ]]; then
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    echo 'export PATH="$CUDA_HOME/bin:$PATH"'
    if [[ -n "${CPATH:-}" ]]; then printf 'export CPATH=%q\n' "$CPATH"; fi
  fi
  echo 'export MUJOCO_GL=egl'
  (( ! ACCEPT_EULA )) || echo 'export OMNI_KIT_ACCEPT_EULA=YES'
} > "$STATE_DIR/env.sh.tmp"
mv "$STATE_DIR/env.sh.tmp" "$STATE_DIR/env.sh"
log "Installation and import checks completed. Log: $LOG_FILE"
printf 'Next: source %q\n' "$STATE_DIR/env.sh"
echo 'Task text embeddings are bundled in Git. Prepare checkpoints, scene assets, and task datasets before evaluation.'
