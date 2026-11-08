# Validation Record

Updated: 2026-11-08. All checks described here avoid physical robot control.

## Prior package validation

- Python AST, shell syntax, JSON, relative Markdown links, eight task-pose mappings, and package SHA256 checks passed.
- 22 deployment tests passed, covering hand filtering, lifecycle/HOLD, result freshness, HTTP/image transport, and SONIC lookahead.
- Six installation/path tests passed, including paths with spaces, extraction, overwrite refusal, traversal/hash rejection, and shared custom HTTP port handling.
- The original model host passed CPU metadata/asset/text-cache preflight for the eight released tasks. Seven unchanged tasks were checked on 2026-11-07; `teleop-long`, its original pose and checkpoint 30000 were checked on 2026-11-08; see `manifests/server-preflight-observed.json`.
- One preprocessing-equivalence test passed on that host without constructing the model or allocating a GPU.
- The 3,814,025,719-byte runtime archive was fully extracted with the installer into a temporary macOS directory. All 8,545 files passed SHA256 verification. Temporary files were removed afterward. This validates extraction/integrity, not Linux dynamic loading or compilation.

## Reproduce offline checks

From the real-world directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/verify_package.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_release_installation.py -v
# Use an interpreter with numpy, scipy, Pillow, and OpenCV:
cd deploy
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  test_hand_stability test_deployment_lifecycle test_deployment_regressions \
  test_image_wire test_sonic_lookahead -v
# In the full model environment, after sourcing env.sh:
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_preprocessing_equivalence -v
```

## Still requiring validation

Real-checkpoint GPU inference and on-site PICO/G1 recording/motion acceptance have not been performed as part of packaging. The PICO headset APK and robot camera environment still need a verified version record. The workstation software installation and build were validated as described below; package pins are not a cross-platform lock with wheel hashes.

The integrated release removes the old deployment subtree/submodule entry and uses one English deployment guide. The unified model code remains in the parent repository. Repository-level compatibility checks are reported alongside the integration change.

## Integrated main-repository validation

The integration was checked against main commit `905265caaf34ef31cb7a904ae137544160d5eb42`:

- 9 parent-repository real-world HTTP/bridge/runtime tests passed.
- 22 packaged deployment tests and 6 installer/path tests passed.
- 117 CPU tests passed in an isolated temporary copy on the model host with `CUDA_VISIBLE_DEVICES` empty. These covered training initialization/resume, the eight-GPU launcher, real-world loading, startup conditioning, hand control/diffusion, G1 motion conversion, text caches, simulator server transactions, and preprocessing equivalence.
- Total: 154 passing tests across these suites. No training job, GPU model service, or robot controller was started.
- Python/shell/JSON/link/pose/hash checks passed. All 13 real-world Markdown documents are English. The model, motion, dataset, and evaluation implementation files in the parent repository were left unchanged.

For the CPU integration suite, from the repository root with the model environment active:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD/tests:$PWD/real-world/deploy:$PWD" python -m unittest \
  test_training_resume test_realworld_launcher test_realworld_loader \
  test_startup_conditioning test_hand_control test_hand_synchronized_diffusion \
  test_g1_reference test_evaluation_text_cache test_server_transaction \
  test_preprocessing_equivalence -v
```

The existing HTTP test double was updated for the session-aware reset/runtime-owned validation contract. Two stale training-test paths were updated to the current script layout and grouped-sampling configuration. A direct/importlib server import issue was fixed by resolving sibling modules after adding the deployment directory to `sys.path`.

## Real-world model and dataset release validation (2026-11-08)

- Eight checkpoint steps and task instructions match the released task catalog; `long` uses `teleop-long`.
- All learned model tensors are finite and unchanged after replacing machine-specific path metadata with portable paths.
- The 236 demonstrations contain 157,769 frames. Episode metadata, required action/state shapes, finite values, and decoded video frame counts agree.
- Checkpoint and data file sizes and SHA-256 hashes are recorded in [released-assets.json](../manifests/released-assets.json). These structural and integrity checks do not assert that every demonstration is successful or free of recording artifacts.

## Documentation follow-through review — 2026-11-08

Reviewed against main `6a0245784d9f74a84f6d499a929baa13b1b1d479`, preserving the released eight-task assets and `teleop-long` mapping. All 39 Bash code blocks passed syntax checks; the eight release checkpoint/dataset/pose mappings matched. Package syntax, relative links, and hashes passed, and `setup.sh --only training --dry-run` selected only the model environment. This documentation-only revision did not rerun GPU or physical robot tests. The README now links to an exact source/generated-directory inventory and separates deployment-only, training-only, and full collection workflows.

## Workstation installation validation (2026-11-08)

- Validated on Linux x86_64, Ubuntu 20.04.4, with an RTX 4090 and an existing NVIDIA driver. Both Python 3.10 Conda environments were created in an isolated checkout; system build dependencies were installed with apt.
- The five published Hugging Face release files passed remote size/hash checks. Anonymous HTTP access and a resumed public download passed full archive SHA-256 verification; the installer pins dataset revision `e5927e63d9cbe23df2f5054d504e5d0924cd06ab`.
- The 3,814,025,719-byte archive matched SHA-256 `f3504e6c930fc10fdfdff0e6c143b8a5c83b73ebc03e028ab14fda3ac34a44b1`. All 8,545 files passed verification before applying the collection overlay.
- Pinned teleoperation/collection requirements installed successfully. SONIC/GMR editable sources and the PICO Python SDK were built and installed; both environments passed `pip check` and the deployment-critical version checks.
- The `g1_deploy_onnx_ref` C++ target compiled and linked. GMR constructed the G1 q29 mapping (`nq == 36`); the recorder/exporter imports passed. SONIC dynamic-library checks and the isolated PC Service loader check passed.
- The completed installation passed `setup.sh --only workstation --check`; rerunning the complete installer reused the verified runtime and environments and passed the build/import/library checks again.
- Eleven workstation setup tests cover verified downloads and resumption, corrupt archive rejection, unmanaged/modified directory protection, repeated preparation after rebuilding the PICO extension, mode routing, and protection against local interpreter-path overrides during installation.

These checks validate the software installation, source build and local loading. No PICO, camera or robot service was started, and no GPU TensorRT inference or physical recording/motion acceptance is claimed.
