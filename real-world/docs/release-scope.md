# Provenance and Distribution Scope

The package is based on read-only snapshots of working model-server and workstation installations taken on 2026-11-07. It retains their deployment, hand filtering, and task-pose mechanisms, while adding portable paths, asset installation, validation, and documentation.

## Repository integration

The parent repository already contains the matching policy, real-world loader, motion codec, and FT YAML. They are not duplicated here. The real-world FT launcher additionally supports `KIMODO_RESUME=1` to invoke the existing `train.py --resume` interface.

The old SONIC submodule path is retired. Use the pinned runtime archive under `external/GR00T-WholeBodyControl`; do not silently substitute the former submodule revision. The canonical deployment guide is `deploy/start.md`.

## Distribution

Source, configuration examples, manifests, and small task text caches are distributed through Git. The eight evaluated policy checkpoints and training datasets are available through Hugging Face; see [downloads and verification](released-assets.md). Base models are distributed separately. The approximately 3.81 GB [workstation runtime archive](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/assets/real_world_assets) is distributed through the dataset repository with version metadata and a per-file SHA-256 manifest. Install it with the explicit `setup.sh --only workstation` mode; it is not part of the default training/simulation setup.

The runtime archive retains upstream licenses. Native libraries, robot resources, and models remain subject to their original licenses and redistribution conditions. No new third-party license is granted by this packaging work. Workstation native binaries target the recorded Linux x86_64 environment, not the robot's ARM64 environment.

## Validation limits

See the [validation record](verification.md). A fresh pair of Conda environments and the native build passed on the documented Ubuntu 20.04/RTX 4090 host. Software checks do not establish successful motion on a new robot or task success. Record the matched PICO APK and robot camera environment before claiming complete environment reproducibility. Users must set their own network interfaces, addresses, SSH connection, checkpoints, and scene calibration. Provided task poses describe the original datasets; they are not universal robot calibration.

`SOURCE_MANIFEST.json` records historical source hashes. `PACKAGE_MANIFEST.json` records current packaged files; runtime assets have their own manifest. Historical hashes may differ after reviewed edits or refer to interfaces now maintained in the parent repository.

KP is the existing name of owned channels. ZeroMQ transport and legacy argument compatibility remain unchanged.

## Upgrade from the original deployment directory

Use a fresh clone instead of overlaying a working robot installation. The old `real-world/GR00T-WholeBodyControl` submodule and `deploy/Start.md` have been replaced by the pinned runtime installation and `deploy/start.md`. Preserve datasets, checkpoints, local configuration, and logs before retiring an old checkout.
