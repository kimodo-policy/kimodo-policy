# 1. Installation and Directory Layout

Clone this release on both hosts. The commands below prepare the workstation environments; use the root training setup on the model host. The model host needs the complete top-level model source; the workstation uses this real-world directory and the runtime archive.

## 1.1 Clone and configure both hosts

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kimodo-policy/kimodo-policy.git "$HOME/kimodo-policy"
cd "$HOME/kimodo-policy/real-world"
cp local.env.example local.env
cp deploy/deploy_config.example.json deploy/deploy_config.json
nano local.env
```

Use a fresh clone when migrating from the old SONIC submodule layout. Do not overwrite existing machine configuration with these copy commands. `GIT_LFS_SKIP_SMUDGE=1` avoids downloading unrelated benchmark assets during clone; the real-world runtime is installed separately. On a workstation using automatic setup, remove the placeholder interpreter assignments from `local.env` so the generated paths take effect.

Edit `local.env`:

- Model host: set `KIMODO_PROJECT_ROOT` and `KIMODO_SERVER_PYTHON`.
- Workstation: set `ROBOT_INTERFACE`, `PICO_INTERFACE`, `CAMERA_HOST`, and `KIMODO_SERVER_SSH`. Automatic setup supplies `TELEOP_PYTHON` and `COLLECTION_PYTHON`; override them only when using separately prepared environments.
- Use absolute interpreter paths. Find one with `conda run -n YOUR_ENV python -c 'import sys; print(sys.executable)'`.
- Remove or comment out unused role-specific placeholders. Do not write `conda activate ...` as an interpreter path.

The default paths are `external/GR00T-WholeBodyControl`, `external/GMR`, and `runtime/native`, all beneath this directory. They replace the former SONIC submodule location.

Start each subsequent terminal with:

```bash
cd "$HOME/kimodo-policy/real-world"  # Adjust if you cloned elsewhere
source env.sh
```

## 1.2 Workstation: install environments and runtime

Use a fresh checkout on Linux x86_64 with Conda and an NVIDIA GPU driver already available. The native bundle was exported from the Ubuntu 20.04 workstation documented in the [version record](00-versions-assets.md). Workstation installation is separate from the root README's training and simulation setup.

From the repository root:

```bash
bash setup.sh --only workstation --system-deps --download-resources
source real-world/env.sh
```

This creates two local Conda prefixes, `real-world/.envs/kimodo-teleop` and `real-world/.envs/kimodo-collection`, using Python 3.10 and CPU PyTorch 2.6.0. SONIC C++ still uses the workstation GPU and bundled native runtime. Pinned pip dependencies are recorded in `environments/teleop.requirements.txt` and `environments/collection.requirements.txt`; LeRobot uses the original collection integration's fixed Git commit.

The command installs Ubuntu build libraries when `--system-deps` is selected, downloads the runtime, checks its archive and all extracted file hashes, applies the collection overlay, installs the editable sources, builds SONIC, and checks imports and dynamic libraries. It does not launch a camera, PICO service, model server, or robot controller. It does not replace the system CUDA installation, glibc, or NVIDIA driver.

Logs and generated paths are saved under `real-world/.state/`. `env.sh` loads the generated workstation paths first, then applies any `local.env` overrides. Existing machine configuration is preserved. On reruns, source, asset and native-library hashes are verified before reuse; the PICO SDK extension is rebuilt locally and checked by import rather than compared with its original binary hash. An unknown or modified source/runtime directory is not overwritten. Use a fresh checkout when upgrading the bundled runtime or collection overlay.

Preview or check an installation:

```bash
bash setup.sh --only workstation --dry-run
bash setup.sh --only workstation --check
```

Use `--conda /path/to/bin/conda` if Conda is not detected. Configure robot/PICO interfaces, camera address and SSH settings in `real-world/local.env`. On the workstation, omit the placeholder interpreter assignments from `local.env.example` to retain the generated paths. The installer creates `deploy/deploy_config.json` from its example only if the file is absent.

### Manual download or offline runtime installation

Download the bundle and its published verification metadata from [Hugging Face](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/assets/real_world_assets), or use:

```bash
hf download YunhengWang/kimodo-policy --repo-type dataset \
  --include 'assets/real_world_assets/**' --local-dir .setup/workstation-download
bash setup.sh --only workstation --system-deps \
  --runtime-archive "$PWD/.setup/workstation-download/assets/real_world_assets/workstation-runtime.tar.gz"
```

The runtime is approximately 3.81 GB compressed. Reserve space for the extracted resources, both Conda environments, build products and generated TensorRT engines. The archive contains no Conda environments, task datasets, task policies or cached TensorRT engines. Python/Conda package installation still requires network access unless those packages are already cached.

The [runtime manifest](../manifests/workstation-runtime.json) is bundled with this code release. The installer verifies it automatically; [download metadata](../manifests/workstation-download.json) records the release URL, size and archive hash. For manual inspection:

```bash
sha256sum .setup/workstation-download/assets/real_world_assets/workstation-runtime.tar.gz
```

Compare the result with `archive_sha256` in the manifest. Device pairing, network connectivity, task-pose calibration and on-site acceptance remain separate steps in [chapter 2](02-network-camera.md) and the [deployment guide](../deploy/start.md).

## 1.3 Model host: validate sources and assets

If the model environment is not installed, run the root installer for **training only** (not its default, which also installs simulators):

```bash
cd "$KIMODO_PROJECT_ROOT"
bash setup.sh --only training
source .setup/env.sh
conda activate kimodo-policy
python -m pip install pyzmq==27.2.0 opencv-python-headless==5.0.0.93
python -c 'import sys; print(sys.executable)'
cd real-world
nano local.env  # Set KIMODO_SERVER_PYTHON to the printed path
source env.sh
```

The two additional packages match the real-world environment checker and are not installed by the root training requirements. Skip installation if your model environment already exists. The `--only training` mode does not create workstation environments. The model and training interfaces are already integrated; no source overlay is needed.

```bash
"$KIMODO_SERVER_PYTHON" tools/check_environment.py server
```

Follow the [download and verification guide](released-assets.md) to obtain the eight evaluated task checkpoints and training datasets. Install the base models under `$KIMODO_PROJECT_ROOT/checkpoints/`. In `deploy/deploy_config.json`, set each `tasks.<task>.checkpoint` to an **absolute checkpoint directory**. Explicit paths avoid ambiguity between training runs.

```bash
nano deploy/deploy_config.json  # Set tasks.boxing.checkpoint to the downloaded absolute path
"$KIMODO_SERVER_PYTHON" tools/preflight_server.py boxing
```

This checks checkpoint metadata, required assets, task pose, and text-cache resolution. It does not construct the model, allocate a GPU, or prove that all weight tensors can load. The subsequent server startup performs actual model loading.

## 1.4 Ports and configuration

- Camera: 5555; SONIC state: 5557; reference actions: 5556.
- Model HTTP: 18080 by default, bound to loopback and forwarded through SSH.
- Set identical `server.port` values in both hosts' JSON files, or set the same `KIMODO_PORT` in both `local.env` files. The latter overrides server, bridge, and tunnel ports. A one-off `--port` only affects that launcher invocation.
- Interpreter and camera variables in `local.env` override JSON examples.
- The default text cache is `assets/text-cache`. For newly trained instructions, point the server to its training-generated cache as described in chapter 3.
