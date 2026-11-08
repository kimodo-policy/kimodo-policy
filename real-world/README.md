# Kimodo Policy — Real-World Robotics

PICO teleoperation → G1 recording → fine-tuning → remote policy deployment.

**Hardware:** Unitree G1 with two Dex3 hands, a robot camera, an NVIDIA workstation, and a GPU model server. PICO is needed for teleoperation only. The workstation runs SONIC; the server runs Kimodo. Isaac Sim/Lab are not required.

## Before you start

Prepare the [matching environments and assets](docs/00-versions-assets.md). The eight evaluated policies and datasets are released; [download commands](docs/released-assets.md) include checksum verification. Text embeddings and task poses are included in Git.

From the repository root, run `bash setup.sh --only workstation --system-deps --download-resources` on the workstation. This explicit mode prepares the teleoperation and collection/bridge environments, downloads the [3.81 GB runtime bundle](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/assets/real_world_assets), verifies its hashes, and builds SONIC. The model server uses `bash setup.sh --only training`. Follow the [installation guide](docs/01-install.md); the default training/simulation setup does not install workstation environments.

The matched PICO APK and robot camera environment still need a verified version record. See the [validation scope](docs/verification.md) for software checks and remaining on-site acceptance steps.

## Follow the commands in order

| Step | Guide | Where / completion check |
|---|---|---|
| 1 | [Install and configure](docs/01-install.md) | Both hosts: configure `local.env`; workstation passes `check-local`; server passes task preflight |
| 2 | [Connect G1, camera, and PICO](docs/02-network-camera.md) | Workstation + robot: correct route, fresh camera frames; PICO connected for collection |
| 3 | [Collect, validate, transfer, and train](docs/03-collection-training.md) | Workstation records valid episodes; model host fine-tunes or resumes |
| 4 | [Deploy a task](deploy/start.md) | Server → SSH tunnel → SONIC → check → run → HOLD/shutdown |

**Deployment only:** use a released checkpoint and follow steps 1, 2 (skip PICO), and 4. **Training only:** follow step 1's model-host setup, download a dataset, then section 3.7. Each operating guide identifies the host and terminal; replace placeholders before running commands.

## Files you edit

- `local.env`: machine paths, Python interpreters, network interfaces, camera address, and SSH alias.
- `deploy/deploy_config.json`: task checkpoints, diffusion steps, execution length, RTC, and bridge settings. Copy the example once; `deploy.sh run boxing` selects a task without rewriting JSON.
- `deploy/*_initial_pose.json`: recorded task-specific starting poses; check them against your robot/data before execution.

See the [complete directory map](docs/directory.md) for every tracked file and generated runtime directory. Additional references: [architecture](docs/architecture.md), [data semantics](docs/data-and-training.md), and [distribution/migration notes](docs/release-scope.md).

An on-site operator must support the robot and prepare an emergency stop before starting SONIC or `run`. Ctrl+C enters streaming HOLD; follow the [shutdown sequence](deploy/start.md#46-stop-hold-and-switch-tasks) before closing the controller.
