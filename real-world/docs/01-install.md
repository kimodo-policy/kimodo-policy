# 1. Installation and Directory Layout

The workstation requires the compatible teleoperation and collection Conda environments listed in chapter 0. Clone this release on both hosts. The model host needs the complete top-level model source; the workstation uses this real-world directory and the runtime archive.

## 1.1 Clone and configure both hosts

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kimodo-policy/kimodo-policy.git "$HOME/kimodo-policy"
cd "$HOME/kimodo-policy/real-world"
cp local.env.example local.env
cp deploy/deploy_config.example.json deploy/deploy_config.json
nano local.env
```

Use a fresh clone when migrating from the old SONIC submodule layout. Do not overwrite existing machine configuration with these copy commands. `GIT_LFS_SKIP_SMUDGE=1` avoids downloading unrelated benchmark assets during clone; the real-world runtime is installed separately.

Edit `local.env`:

- Model host: set `KIMODO_PROJECT_ROOT` and `KIMODO_SERVER_PYTHON`.
- Workstation: set `TELEOP_PYTHON`, `COLLECTION_PYTHON`, `ROBOT_INTERFACE`, `PICO_INTERFACE`, `CAMERA_HOST`, and `KIMODO_SERVER_SSH`.
- Use absolute interpreter paths. Find one with `conda run -n YOUR_ENV python -c 'import sys; print(sys.executable)'`.
- Remove or comment out unused role-specific placeholders. Do not write `conda activate ...` as an interpreter path.

The default paths are `external/GR00T-WholeBodyControl`, `external/GMR`, and `runtime/native`, all beneath this directory. They replace the former SONIC submodule location.

Start each subsequent terminal with:

```bash
cd "$HOME/kimodo-policy/real-world"  # Adjust if you cloned elsewhere
source env.sh
```

## 1.2 Workstation: install runtime assets

On the Ubuntu 20.04 workstation, install build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libeigen3-dev libgtest-dev \
  libzmq3-dev zlib1g-dev libyaml-cpp-dev libusb-1.0-0 libgl1 libglib2.0-0 \
  libmpfr6 libmpc3 libgmp10 rsync git
```

Observed system package versions are recorded in `manifests/system-dependencies.json`. This apt command uses available distribution versions; it is not a bit-for-bit system rebuild.

After obtaining the matching archive:

```bash
python3 tools/install_runtime_bundle.py /path/to/workstation-runtime.tar.gz \
  --manifest manifests/workstation-runtime.json
bash setup/apply_collection_overlay.sh
bash setup/install_python_sources.sh
bash collection/bin/build-sonic
bash collection/bin/check-local
```

The installer checks the archive and every extracted file, confines extraction to a staging directory, and refuses to overwrite existing SONIC/GMR/native directories. Apply the collection overlay only to a fresh extraction.

The source installer binds SONIC, GMR, and the PICO SDK to their new paths. It does not solve Conda dependencies. Your environments must already include pip, setuptools, wheel, cmake, pybind11, and the recorded runtime dependencies; `--no-deps` does not remove that requirement.

```bash
"$TELEOP_PYTHON" tools/check_environment.py teleop
"$COLLECTION_PYTHON" tools/check_environment.py collection
```

`build-sonic` builds `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`; it does not control the robot. `check-local` checks imports, GMR joint mapping, and dynamic libraries without connecting to hardware.

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

The two additional packages match the real-world environment checker and are not installed by the root training requirements. Skip installation if your model environment already exists. This installer does not create the workstation teleoperation/collection environments. The model and training interfaces are already integrated; no source overlay is needed.

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
