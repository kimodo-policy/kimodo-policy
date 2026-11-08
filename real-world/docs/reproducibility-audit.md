# Reproducibility Status

This page supersedes the preliminary packaging audit. For current instructions, start with the [real-world README](../README.md).

The package now includes observed dependency inventories, a checksummed runtime archive manifest and installer, a unified directory/environment entry point, task embeddings and poses, and end-to-end operating instructions.

The eight evaluated checkpoints and training datasets are now available with [download commands and checksums](released-assets.md). The workstation runtime is published separately with checksums and an explicit installer mode. Remaining acceptance requirements include a verified PICO APK version, the robot-side camera environment record, and on-site hardware validation. The explicit installer, native build and local software checks passed on Ubuntu 20.04 with an RTX 4090; see the current validation record for the exact scope. See [distribution scope](release-scope.md) and [validation](verification.md). Do not treat offline packaging checks as proof of complete hardware reproducibility.
