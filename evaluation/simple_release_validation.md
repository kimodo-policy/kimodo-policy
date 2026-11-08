# SIMPLE release validation

Validation date: 2026-11-08. These checks cover the vendored SIMPLE integration
and the frozen resources referenced by `utils/simple_release_manifest.json`.

| Check | Result |
| --- | --- |
| Frozen training data | All 794 episodes across the eight released tasks passed metadata, parquet, task-index, numeric-value and video-presence checks. |
| Current official Close Door training archive | All 99 episodes passed the same checks. |
| Published resources | The eight retained training archives and eight Git-bundled SIMPLE embeddings retain their validated hashes. The evaluation asset archives are unchanged. |
| Download and extraction | Pinned training download and offline verification of bundled text embeddings passed; all 12 evaluation resource archives passed SHA-256 verification and full extraction. |
| Scene references | All six HSSD rooms resolved their local USD and texture dependencies, including exporter-specific paths and filename case differences. |
| Model dependencies | 20 local backbone, visual-encoder and Close Door checkpoint files matched the published model files by size and hash. |
| Regression tests | The SIMPLE test selection passed (32 tests, one USD test skipped where USD was unavailable); the isolated USD/release-tool selection passed all four tests. |
| Training | One real GPU forward/backward/optimizer step completed with finite loss on Close Door data. Single-GPU and four-GPU launcher argument checks passed. |
| Evaluation | Close Door level 0 completed one two-step episode using newly extracted release assets, the released policy, RTX 4090 and CUDA 12.8. The evaluator produced episode results, videos, a summary and its completion marker. |
| Fresh dependency resolution | The locked environment resolved for Ubuntu 22.04-compatible wheels. This was a dependency-installation plan, not a complete clean installation. |

The evaluation smoke test deliberately stopped after two steps and reported
zero successes; it verifies loading, rendering and inference, not task success.
The eight-task benchmark and its full episode protocol were not rerun in this
audit. A full clean-machine installation remains to be validated separately.

The tested simulation environment already existed on Ubuntu 20.04. It was
repaired with EGL libraries and a separate CUDA 12.8 compiler without replacing
the system CUDA installation. A fresh installation of the current lockfile
requires glibc 2.34 or newer; use Ubuntu 22.04 or newer as documented.

The release contains eight Teleop task policies. MP controllers and evaluation
episodes remain in the simulator resource bundle. MP training data, text caches
and task policies are not part of this release. End-to-end MP evaluation was not
validated by this audit; adding an MP result requires a matching trained policy
and a separate evaluation.
