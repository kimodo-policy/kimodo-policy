# 0. Observed Versions and Required Assets

These versions were read from working machines on 2026-11-07. They are not minimum requirements or a claim of compatibility with every Ubuntu release.

| Component | Workstation: teleoperation / collection / bridge | Model server / training |
|---|---|---|
| OS | Ubuntu 20.04.6, x86_64 | Ubuntu 24.04.4, x86_64 |
| Observed GPU / driver | RTX 3060 Laptop, 6 GB / 570.133.07 | H20 / 595.58.03 |
| Python | 3.10.9 | 3.10.20 |
| PyTorch | 2.6.0+cpu | 2.11.0+cu126 |
| NumPy / SciPy | 1.26.4 / 1.15.3 | 2.2.6 / 1.15.3 |
| SONIC native runtime | CUDA 11.8.89, TensorRT 10.13.0.35, ORT 1.16.3, GCC 10 | Not needed |
| XRT PC Service / Python SDK | 1.0.0.0 / 1.0.2 | Not needed |

CPU PyTorch on the workstation does not mean CPU-only control: SONIC C++ still uses its local GPU runtime.

`environments/*.observed.json` records Python packages. The matching `*.observed-constraints.txt` files are version references, not verified clean Conda solves, and do not include wheel hashes. The following instructions assume compatible Conda environments already exist.

The GMR baseline is `bb1bbe40774794fceb2a7c579a3464a28e68c844` from [GMR](https://github.com/YanjieZe/GMR). The working SONIC tree had no Git metadata, so it is pinned by the archive and per-file SHA256 manifest rather than an inferred commit. The integrated model interfaces were compared with main commit `905265caaf34ef31cb7a904ae137544160d5eb42`.

## Required assets

| Asset | Installation / source | Verification |
|---|---|---|
| `workstation-runtime.tar.gz` | Obtain the archive matching this code release from the maintainer | `manifests/workstation-runtime.json`; includes SONIC/GMR, SDK, robot resources, reference motions, ONNX models, and native libraries |
| SONIC decoder, encoder, and planner | Included in the runtime archive | `manifests/runtime-assets.json` |
| Eight task text embeddings | Included in `assets/text-cache/` | `manifests/text-cache.json`; copied without re-encoding |
| Kimodo-G1-RP-v1 and DINOv3 | Obtain the base model assets and install under the model repository's `checkpoints/` | See directory structure below |
| Real-world task checkpoints | Obtain from the maintainer or fine-tune your own | Each directory needs `config.json` and `training_state.pt`; numeric/latest discovery additionally requires eight rank RNG files |
| Llama / LLM2Vec | Needed to generate missing training text caches | Meta-Llama-3-8B-Instruct, LLM2Vec-Meta-Llama-3-8B-Instruct-mntp, LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised |

The main repository links to the [Kimodo-Policy model release](https://huggingface.co/YunhengWang/kimodo-policy). This does not establish public download paths for the eight real-world checkpoints. Maintainers must provide their actual locations and hashes.

```text
checkpoints/
  Kimodo-G1-RP-v1/{config.yaml,model.safetensors,stats/...}
  dinov3-vitl16-pretrain-lvd1689m/{config.json,model.safetensors,preprocessor_config.json}
```

The runtime archive excludes Conda environments, recorded datasets, task checkpoints, and cached TensorRT engines. Build engines on the target GPU. Native dependencies remain under `runtime/native`; do not replace system glibc or CUDA. Allow at least 7 GB for extracted files plus the archive and additional build/engine space.

## Maintainer export

To create a new archive from the original, known-working workstation layout:

```bash
python3 tools/export_workstation_bundle.py \
  --source-root /path/to/existing/workstation \
  --output /path/to/release/workstation-runtime.tar.gz
```

This creates the archive and a `.tar.gz.json` manifest. Update `manifests/workstation-runtime.json` together with the distributed archive. Preserve upstream licenses and check redistribution terms for native binaries and models.

The PICO headset APK version and robot-side camera Python/SDK versions have not been verified. Do not infer APK compatibility from the PC SDK version alone.

See the [task catalog](../manifests/task-catalog.json) for dataset names, exact instructions, recorded checkpoint defaults, and embedding dimensions. All eight caches passed CPU loading and task resolution in the original model environment.
