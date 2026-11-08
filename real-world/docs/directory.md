# Real-World Directory Map

Use the [README](../README.md) for the operating sequence. This inventory separates tracked source from machine-generated assets. It is not an additional setup step.

| Directory / file | Purpose |
|---|---|
| `collection/bin/` | Operator commands: service, manager, SONIC, camera/network checks, recorder, dataset validators |
| `collection/overlay/` | Reviewed Python/C++ source applied to the runtime bundle |
| `collection/setup/` | Isolated native library/compiler environment |
| `deploy/` | Task launcher, HTTP server, asynchronous bridge, RTC, hand filters, initial poses, diagnostics, tests |
| `setup/` | Apply collection overlay and bind editable Python packages |
| `tools/` | Asset installation/export, environment/task preflight, offline verification |
| `environments/` | Observed versions for teleoperation, collection, and server |
| `manifests/` | Released dataset/checkpoint checksums, runtime checksums, task catalog, dependency records |
| `assets/text-cache/` | Eight released task embeddings |
| `docs/` | Sequential operating guides and optional reference material |
| `env.sh`, `local.env.example` | Portable paths and per-machine settings |
| `PACKAGE_MANIFEST.json`, `SOURCE_MANIFEST.json` | Current package hashes and historical provenance |

## Tracked source files

```text
real-world/
├── assets/
│   └── text-cache/
│       ├── teleop-banana__b11cb9af8794.pt
│       ├── teleop-boxing__986113d51416.pt
│       ├── teleop-football__ddee1bcebe62.pt
│       ├── teleop-long__6cd5daf2e16f.pt
│       ├── teleop-pnp__ca9a651badf1.pt
│       ├── teleop-pushing__bd82fc4820be.pt
│       ├── teleop-sitting__d43c5d3890f3.pt
│       └── teleop-transportation__dc51a0c0c5aa.pt
├── collection/
│   ├── bin/
│   │   ├── build-sonic
│   │   ├── check-camera
│   │   ├── check-local
│   │   ├── check-network
│   │   ├── collect
│   │   ├── common.sh
│   │   ├── pico-manager
│   │   ├── scan-datasets
│   │   ├── sonic-control
│   │   ├── validate-dataset
│   │   └── xrt-service
│   ├── overlay/
│   │   └── GR00T-WholeBodyControl/
│   │       ├── gear_sonic/
│   │       │   ├── data/
│   │       │   │   ├── exporter.py
│   │       │   │   ├── features_sonic_vla.py
│   │       │   │   ├── g1_reference.py
│   │       │   │   └── video_writer.py
│   │       │   └── scripts/
│   │       │       ├── check_camera_stream.py
│   │       │       ├── convert_to_humanoidarena_v3_1.py
│   │       │       ├── pico_manager_thread_server.py
│   │       │       ├── run_camera_viewer.py
│   │       │       ├── run_data_exporter.py
│   │       │       ├── run_realsense_camera_server.py
│   │       │       ├── scan_datasets.py
│   │       │       └── validate_g1_vla_dataset.py
│   │       └── gear_sonic_deploy/
│   │           └── src/
│   │               └── g1/
│   │                   └── g1_deploy_onnx_ref/
│   │                       ├── include/
│   │                       │   ├── input_interface/
│   │                       │   │   └── zmq_manager.hpp
│   │                       │   └── output_interface/
│   │                       │       ├── output_interface.hpp
│   │                       │       ├── ros2_output_handler.hpp
│   │                       │       └── zmq_output_handler.hpp
│   │                       └── src/
│   │                           └── g1_deploy_onnx_ref.cpp
│   └── setup/
│       └── native_env.sh
├── deploy/
│   ├── __init__.py
│   ├── banana_initial_pose.json
│   ├── boxing_initial_pose.json
│   ├── deploy.sh
│   ├── deploy_cli.py
│   ├── deploy_config.example.json
│   ├── deployment_lifecycle.py
│   ├── diagnostics.py
│   ├── football_initial_pose.json
│   ├── generate_initial_poses.py
│   ├── hand_stability.py
│   ├── image_wire.py
│   ├── kimodo_rtc.py
│   ├── kimodo_sonic_bridge.py
│   ├── long_initial_pose.json
│   ├── long_v2_initial_pose.json
│   ├── pnp_initial_pose.json
│   ├── pushing_initial_pose.json
│   ├── real_world_server.py
│   ├── replay_probe.py
│   ├── replay_session.py
│   ├── run_kimodo_sonic.py
│   ├── sitting_initial_pose.json
│   ├── sonic.sh
│   ├── start.md
│   ├── summarize_diagnostics.py
│   ├── test_deployment_lifecycle.py
│   ├── test_deployment_regressions.py
│   ├── test_hand_stability.py
│   ├── test_image_wire.py
│   ├── test_preprocessing_equivalence.py
│   ├── test_sonic_lookahead.py
│   ├── transportation_initial_pose.json
│   └── tunnel.sh
├── docs/
│   ├── 00-versions-assets.md
│   ├── 01-install.md
│   ├── 02-network-camera.md
│   ├── 03-collection-training.md
│   ├── architecture.md
│   ├── collection.md
│   ├── data-and-training.md
│   ├── directory.md
│   ├── release-scope.md
│   ├── released-assets.md
│   ├── reproducibility-audit.md
│   ├── setup.md
│   └── verification.md
├── environments/
│   ├── collection.observed-constraints.txt
│   ├── collection.observed.json
│   ├── server.observed-constraints.txt
│   ├── server.observed.json
│   ├── teleop.observed-constraints.txt
│   └── teleop.observed.json
├── manifests/
│   ├── cppzmq.json
│   ├── native-downloads.json
│   ├── observed-environment.json
│   ├── released-assets.json
│   ├── runtime-assets.json
│   ├── server-preflight-observed.json
│   ├── system-dependencies.json
│   ├── task-catalog.json
│   ├── text-cache.json
│   └── workstation-runtime.json
├── setup/
│   ├── apply_collection_overlay.sh
│   └── install_python_sources.sh
├── tools/
│   ├── check_environment.py
│   ├── export_workstation_bundle.py
│   ├── install_runtime_bundle.py
│   ├── preflight_server.py
│   ├── test_release_installation.py
│   └── verify_package.py
├── .gitignore
├── PACKAGE_MANIFEST.json
├── README.md
├── SOURCE_MANIFEST.json
├── env.sh
└── local.env.example
```

## Created locally; not committed

```text
real-world/
├── local.env                           # This machine's configuration
├── deploy/deploy_config.json            # Selected task checkpoints and settings
├── deploy/logs/                         # Deployment and SONIC logs
├── data/teleop-TASK/                     # Training datasets on the model host
├── external/
│   ├── GR00T-WholeBodyControl/           # Extracted SONIC + collection source
│   │   └── outputs/teleop-TIMESTAMP/    # Workstation recordings: meta/data/videos
│   └── GMR/                            # Extracted retargeting source
└── runtime/
    ├── native/                         # Private native libraries and compilers
    ├── .build/sonic/                    # CMake build files
    ├── .state/xrt/                      # XRT application state
    └── setup/logs/                      # Teleoperation service/manager logs
```

## Shared files in the parent repository

```text
kimodo-policy/
├── setup.sh, requirements.txt           # Model environment installation
├── train.py                             # Fine-tuning and full-state resume
├── scripts/Real_World/                  # FT launcher and YAML
├── data/realworld_loader.py              # Recording → model data adapter
├── data/cache/RealWorld/                 # Training-generated text caches
├── model/kimodo_policy.py                # Shared policy
├── motion/g1_reference.py                # Shared motion conversion
├── checkpoints/                         # Frozen base models
├── Real_World/                          # Downloaded released task checkpoints
└── log/Real_World/                       # New fine-tuning outputs
```

`long` uses the released `teleop-long` dataset and `long_initial_pose.json`. `long_v2_initial_pose.json` is retained as a historical alternative and is not selected by the release configuration. `docs/setup.md` and `docs/collection.md` are compatibility links; new users should follow the numbered guides.
