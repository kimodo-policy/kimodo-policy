# Kimodo Policy — Real-World Robotics

PICO teleoperation → G1 data collection → server-side fine-tuning → remote policy inference → workstation/SONIC execution.

This directory provides the real-world workflow for **Unitree G1 with two Dex3 hands**. The model runs on a remote GPU server; the workstation runs camera/state clients, the deployment bridge, and the GPU-backed SONIC controller. Isaac Sim and Isaac Lab are **not required** for this workflow.

## Requirements and release status

Prepare separate compatible Conda environments for teleoperation, collection/bridge, and the model server. See the [observed versions and asset requirements](docs/00-versions-assets.md). These are records of working installations, not a fully resolved cross-platform environment lock.

**Code alone is insufficient.** Obtain the matching `workstation-runtime.tar.gz` (approximately 3.81 GB), the base models, and your task checkpoints. The runtime archive is distributed separately from Git and verified against [its manifest](manifests/workstation-runtime.json). A public download location has not yet been established in this package. Do not assume the archive or the eight real-world checkpoints are included in a clone.

Offline tests and archive integrity checks have passed. A clean Linux rebuild and a complete hardware acceptance run remain separate validation steps. The PICO headset APK version and the robot-side camera environment still need a verified version record.

## Follow this sequence

1. [Verified environment records and required assets](docs/00-versions-assets.md)
2. [Installation and directory layout](docs/01-install.md)
3. [Networking, PICO, and robot camera](docs/02-network-camera.md)
4. [Teleoperation, recording, validation, transfer, and fine-tuning](docs/03-collection-training.md)
5. [Deployment: commands for each terminal](deploy/start.md)

Additional references: [architecture](docs/architecture.md), [data semantics](docs/data-and-training.md), [validation record](docs/verification.md), and [distribution scope](docs/release-scope.md).

## Directory layout

```text
real-world/
  local.env.example       # Copy to local.env and configure this machine
  env.sh                  # Source in each terminal; starts no services
  setup/                  # Runtime source binding and collection overlay
  collection/bin/         # PICO, SONIC, recording, and validation commands
  collection/overlay/     # Reviewed changes applied to the pinned runtime bundle
  deploy/                 # Server, bridge, task poses, and configuration example
  environments/           # Observed Python package versions and constraints
  manifests/              # Native dependencies, runtime files, and task assets
  assets/text-cache/      # Eight original task embeddings
  external/               # Installed SONIC/GMR sources; excluded from Git
  runtime/                # Installed native libraries, logs, and state; excluded from Git
  tools/                  # Installation, validation, and maintainer export tools
```

Model, dataset, and training code live in the parent repository under `model/`, `data/`, `motion/`, and `scripts/Real_World/`. The integrated release does not require copying a second training overlay over those files.

For upgrades from the old layout, use a fresh clone. The former `real-world/GR00T-WholeBodyControl` submodule and `deploy/Start.md` are replaced by the asset installation workflow and the lowercase [deploy/start.md](deploy/start.md). Preserve existing datasets, checkpoints, local configuration, and logs outside the old checkout before retiring it.

SONIC and deployment `run` commands can move the robot. An on-site operator must support the robot and have an emergency stop available. Ctrl+C enters a streaming HOLD only while the controller and communication remain operational; it is not power-loss protection.
