# Provenance and Distribution Scope

The package is based on read-only snapshots of working model-server and workstation installations taken on 2026-11-07. It retains their deployment, hand filtering, and task-pose mechanisms, while adding portable paths, asset installation, validation, and documentation.

## Repository integration

The parent repository already contains the matching policy, real-world loader, motion codec, and FT YAML. They are not duplicated here. The real-world FT launcher additionally supports `KIMODO_RESUME=1` to invoke the existing `train.py --resume` interface.

The old SONIC submodule path is retired. Use the pinned runtime archive under `external/GR00T-WholeBodyControl`; do not silently substitute the former submodule revision. The canonical deployment guide is `deploy/start.md`.

## Distribution

Commit source, configuration examples, manifests, and small task text caches to Git. Distribute the approximately 3.81 GB runtime archive, base models, and task checkpoints separately. Maintainers must provide actual download locations; this package does not claim unpublished assets are publicly available.

The runtime archive retains upstream licenses. Native libraries, robot resources, and models remain subject to their original licenses and redistribution conditions. No new third-party license is granted by this packaging work. Workstation native binaries target the recorded Linux x86_64 environment, not the robot's ARM64 environment.

## Validation limits

See the [validation record](verification.md). Offline checks do not establish successful motion on a new robot, a clean Linux build, or task success. Record the matched PICO APK and robot camera environment before claiming complete environment reproducibility. Users must set their own network interfaces, addresses, SSH connection, checkpoints, and scene calibration. Provided task poses describe the original datasets; they are not universal robot calibration.

`SOURCE_MANIFEST.json` records historical source hashes. `PACKAGE_MANIFEST.json` records current packaged files; runtime assets have their own manifest. Historical hashes may differ after reviewed edits or refer to interfaces now maintained in the parent repository.

KP is the existing name of owned channels. ZeroMQ transport and legacy argument compatibility remain unchanged.
