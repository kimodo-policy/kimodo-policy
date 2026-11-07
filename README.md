<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/picture/kimodo-policy-title-dark.svg">
    <img src="asset/picture/kimodo-policy-title.svg" alt="Kimodo-Policy: From Text-to-Motion Generators to Humanoid Vision-Language-Action Policies" width="960">
  </picture>
</p>

<p align="center">
  <strong>Yunheng Wang</strong><sup>1,*</sup> &nbsp;
  <strong>Jiaxi Zhang</strong><sup>1,*</sup> &nbsp;
  <strong>Wei Lu</strong><sup>1,*</sup> &nbsp;
  Yuetong Fang<sup>1</sup> &nbsp;
  Taowen Wang<sup>1</sup> &nbsp;
  Yixiao Feng<sup>1</sup><br>
  Shuning Zhang<sup>1</sup> &nbsp;
  Junzhe Xu<sup>1,2</sup> &nbsp;
  Yichi Wang<sup>3</sup> &nbsp;
  Zizhao Yuan<sup>1</sup> &nbsp;
  Kailun Huang<sup>1</sup><br>
  Lusong Li<sup>2</sup> &nbsp;
  Mingqian Zhou<sup>4</sup> &nbsp;
  Zecui Zeng<sup>2,†</sup> &nbsp;
  Renjing Xu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>HKUST(GZ)&thinsp;·&thinsp;<sup>2</sup>JD Explore Academy&thinsp;·&thinsp;<sup>3</sup>BJUT&thinsp;·&thinsp;<sup>4</sup>Cambridge<br>
  <sub>* Equal contribution · † Co-corresponding authors</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv paper">
  <a href="https://kimodo-policy.github.io/"><img src="https://img.shields.io/badge/%F0%9F%9A%80_Project_Page-Kimodo--Policy-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/YunhengWang/kimodo-policy"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Dataset_%26_Results-Kimodo--Policy-yellow" alt="Hugging Face dataset and results"></a>
  <a href="https://huggingface.co/YunhengWang/kimodo-policy"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Checkpoints-Kimodo--Policy-yellow" alt="Hugging Face checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

**Kimodo-Policy turns pretrained text-to-motion generators into humanoid vision-language-action policies.** Given egocentric vision, language instructions, and motion history, it predicts coordinated **root → body → hand** actions for closed-loop loco-manipulation. Our core idea is to reuse a frozen motion backbone’s learned coordination and temporal structure, while visual ControlNet branches and a dedicated HandNet ground these priors in the current scene. We evaluate this approach on HumanoidArena, SIMPLE, and real-world humanoid tasks.

![Kimodo-Policy overview](asset/picture/teaser.png)

## 🗺️ Open-Source Roadmap

- [x] [HumanoidArena Training & Evaluation](#humanoidarena-training--evaluation)

- [x] SIMPLE Benchmark: [Training](scripts/Simple_Single_Task/) & [Evaluation](evaluation/simple_eval.md)

- [x] [Real-World Data Collection, Training & Deployment](real-world/README.md) — source and guides; runtime assets and checkpoints are obtained separately.

- [ ] More Benchmarks (Coming Soon)

## ⚙️ Installation & Environment Setup

Use Ubuntu 22.04 or newer on x86_64, with Conda and a compatible NVIDIA GPU driver installed. SIMPLE also requires a CUDA 12.x toolkit with `nvcc` (12.8 recommended). The setup script installs all three environments and checks their imports:

| Environment | Python | Purpose |
| --- | --- | --- |
| `kimodo-policy` | 3.10 | Training and Arena policy serving |
| `unitree_sim_env` | 3.11 | HumanoidArena: Isaac Sim 5.0.0 + Isaac Lab 2.2.0 |
| `SIMPLE/.venv` | 3.10 | SIMPLE: MuJoCo 3.3.6 + Isaac Sim 4.5.0 |

### Quick Installation

Clone the project and run the setup script. The `--system-deps` option installs Ubuntu system packages and may request your sudo password. Use `--accept-isaac-eula` after reviewing and accepting the [Isaac Sim license](https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html).

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kimodo-policy/kimodo-policy.git
cd kimodo-policy
bash scripts/setup.sh --system-deps --accept-isaac-eula
source .setup/env.sh
```

If you already cloned the repository, run the last two commands from its root. The script installs Git LFS files, Python dependencies, Isaac Lab, and cuRobo. It uses the included Arena and SIMPLE integrations and keeps their simulator environments separate, following the [Arena](https://github.com/William-wAng618/HumanoidArena/blob/main/docs/04_environment_setup.md) and [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) setup workflows.

Existing named environments are reused; missing ones are created. Rerunning reinstalls the selected dependency versions and retries the setup without deleting environments. Logs are saved under `.setup/`; the runtime path file is published only after all selected installation and import checks pass.

### Optional Setup Commands

```bash
# Preview the installation plan without making changes.
bash scripts/setup.sh --dry-run

# Install only training dependencies.
bash scripts/setup.sh --only training

# Install one benchmark and the shared Kimodo model environment.
bash scripts/setup.sh --only arena --accept-isaac-eula
bash scripts/setup.sh --only simple --accept-isaac-eula

# Also download SONIC controller files and SIMPLE base scene/robot resources.
bash scripts/setup.sh --accept-isaac-eula --download-resources

# Check existing environments without reinstalling packages.
bash scripts/setup.sh --check --accept-isaac-eula
```

Use `--conda /path/to/bin/conda` if Conda is not detected. For SIMPLE, the script prefers `/usr/local/cuda-12.8` when available; set `CUDA_HOME` to select another CUDA 12.x toolkit. The script does not install GPU drivers or CUDA toolkits.

### Use the Installed Environments

In each new terminal, enter the repository and load its environment paths:

```bash
source .setup/env.sh
conda activate kimodo-policy
```

The generated file configures the Python executables, model dependency path, and default simulator/resource paths used by our wrappers. It does not activate a simulator environment; the evaluation wrappers select the correct runtime.

### Prepare Runtime Assets

Environment installation does not include model weights or benchmark datasets. Before evaluation, prepare:

- **Kimodo weights and text caches:** download the required [released checkpoints](https://huggingface.co/YunhengWang/kimodo-policy), prepare the backbone/encoder files required by the configuration under `checkpoints/`, and place task embeddings under `data/cache/HumanoidArena/` or `data/cache/Simple/`. Training generates missing task embeddings when the LLM2Vec weights are available.

- **Arena scenes and SONIC:** follow **Evaluation Assets** under **Data Preparation** below for the official downloads, our backup asset link, and the required directory layout.

- **SIMPLE evaluation episodes:** download the target task archives from the [official evaluation dataset](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval) and extract them under `SIMPLE/data/simple-eval/`, following the [evaluation guide](https://github.com/physical-superintelligence-lab/SIMPLE/blob/main/docs/source/tutorials/eval.md).

- **Training data:** follow **Data Preparation** below.

Add `--dry-run` to either evaluation wrapper to check its configuration and required paths before launching simulation. Installation/import checks do not replace a small end-to-end evaluation.

## 🤖 HumanoidArena Training & Evaluation

This section covers training and evaluation on HumanoidArena. Complete **Installation & Environment Setup** above first. Training and policy serving use `kimodo-policy`; the Arena simulator runs separately in `unitree_sim_env`.

![HumanoidArena — seven task demonstrations](asset/readme/humanoidarena-gallery.gif)

### ⚙️ Runtime Paths

For training, activate the `kimodo-policy` environment and point the dataset variable to the local HumanoidArena dataset:

~~~bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"
test -d "$HUMANOID_ARENA_ROOT" || {
  echo "Missing HumanoidArena dataset: $HUMANOID_ARENA_ROOT" >&2
  exit 2
}
~~~

For evaluation, keep the simulator and model server in separate environments:

```bash
source .setup/env.sh
```

Before a full run, verify that the simulator Python, model-server Python, SONIC artifacts, text cache, and Isaac Lab task YAML are available. The evaluation wrappers provide dry-run validation without launching Isaac Sim.

### 📦 Data Preparation

#### • Training Data

Before training, download the dataset from the [official HumanoidArena website](https://humanoidarena.github.io/). If the official release is updated or changed, you can use our [archived dataset snapshot on Hugging Face](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/HumanoidArena_dataset_v3_1) to reproduce the experiments with the same data version.

Activate the `kimodo-policy` environment with the project dependencies installed, then run the following commands from the repository root to download our archived snapshot:

```bash
conda activate kimodo-policy
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python -m pip install huggingface_hub
hf download YunhengWang/kimodo-policy \
  --repo-type dataset \
  --include "data/HumanoidArena_dataset_v3_1/**" \
  --local-dir "$PROJECT_ROOT"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"
```

If the dataset is gated or requires authentication, run hf auth login first.

Check dataset completeness and audit the first five selected episodes before training:

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="${HUMANOID_ARENA_ROOT:-$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1}"
python utils/check_datasets.py \
  --config scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml \
  --workers 1 \
  --limit 5 \
  --fail-on-errors \
  --output "$PROJECT_ROOT/log/humanoidarena_dataset_audit.json"
```

The checker verifies selected dataset files, episode metadata, state/action shapes, finite values, frame indices, timestamps, root rotations, and hand-command ranges. It also reports motion-continuity flags in the JSON report; these flags do not remove episodes or change training data. `--fail-on-errors` returns a nonzero exit status for data-reading or schema errors. Add `--fail-on-invalid` to also fail on motion-quality flags. Remove `--limit 5` to audit every selected episode. Video decoding is not checked.

#### • Evaluation Assets

Evaluation requires simulation assets and SONIC controller files in addition to the Kimodo checkpoint and task text cache. The training dataset alone is not sufficient.

**Simulation assets.** Download the [official HumanoidArena asset package](https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view) first, following the [official setup guide](https://github.com/William-wAng618/HumanoidArena/blob/main/docs/04_environment_setup.md). If the official download is unavailable or the assets change, use our [archived HumanoidArena assets on Hugging Face](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/assets/humanoidarena_assets).

To download our backup, run from the repository root:

```bash
conda activate kimodo-policy
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
hf download YunhengWang/kimodo-policy \
  --repo-type dataset \
  --include "assets/humanoidarena_assets/**" \
  --local-dir "$PROJECT_ROOT/.setup/arena-assets"
```

The backup is downloaded under `.setup/arena-assets/assets/humanoidarena_assets/`. Extract any archives before placing the files in the simulator directory. For the official ZIP, replace the archive path below with your downloaded file:

```bash
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export ARENA_ASSET_DIR="$PROJECT_ROOT/HumanoidArena/isaaclab_twist2_g1/assets"
mkdir -p "$ARENA_ASSET_DIR"
unzip /path/to/humanoidarena_assets.zip -d "$ARENA_ASSET_DIR"
```

This extraction command assumes the archive contains `objects/` and `robots/` at its root. If it already contains an `assets/` directory, extract it into `HumanoidArena/isaaclab_twist2_g1/` instead. For an unpacked backup, copy its `objects/` and `robots/` directories into `$ARENA_ASSET_DIR/`. Both download sources must produce this layout, without an extra nested `assets/` folder:

```text
HumanoidArena/isaaclab_twist2_g1/assets/
├── objects/
└── robots/
```

**SONIC controller.** Download the three controller files from the [official NVIDIA GEAR-SONIC release](https://huggingface.co/nvidia/GEAR-SONIC). Skip this download if setup was already run with `--download-resources`.

```bash
conda activate kimodo-policy
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export SONIC_POLICY_ROOT="$PROJECT_ROOT/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir "$SONIC_POLICY_ROOT"
```

Check the installed resource paths before evaluation:

```bash
(
  set -eu
  PROJECT_ROOT="$(git rev-parse --show-toplevel)"
  ARENA_ASSET_DIR="$PROJECT_ROOT/HumanoidArena/isaaclab_twist2_g1/assets"
  SONIC_POLICY_ROOT="$PROJECT_ROOT/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
  for directory in objects robots; do
    test -d "$ARENA_ASSET_DIR/$directory" || {
      echo "Missing simulation assets: $ARENA_ASSET_DIR/$directory" >&2
      exit 1
    }
  done
  for file in model_encoder.onnx model_decoder.onnx observation_config.yaml; do
    test -s "$SONIC_POLICY_ROOT/$file" || {
      echo "Missing SONIC file: $SONIC_POLICY_ROOT/$file" >&2
      exit 1
    }
  done
  echo "Arena asset directories and SONIC files found."
)
```

These are presence checks, not a full asset-integrity or rendering test. `check_datasets.py` audits training episodes; it does not validate simulator assets. Before a full evaluation, also prepare the task checkpoint and `data/cache/HumanoidArena/`, run the evaluation wrapper with `--dry-run`, and verify a small simulation run.

### 🚀 Kimodo-Policy Single-Task Training on HumanoidArena

To help you quickly reproduce our HumanoidArena results, we provide downloadable checkpoints corresponding to the paper, along with evaluation results and visualizations. For each task, the table below includes both a model fine-tuned from 419h pretraining and a model trained from scratch. Click a checkpoint to download the weights, or a result to explore the evaluations and visualizations.

| Task | Pretrain | Kimodo-Policy checkpoint | Results&nbsp;(SR) |
| --- | --- | --- | --- |
| doubledesk | ❌ | [humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | [40.0 ± 0.0%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) |
| football | ❌ | [humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | [28.3 ± 6.2%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) |
| pp_box | ❌ | [humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | [80.0 ± 4.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) |
| boxing | ❌ | [humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | [73.3 ± 13.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) |
| open_door | ❌ | [humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | [90.0 ± 4.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) |
| sit_sofa | ❌ | [humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | [100.0 ± 0.0%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) |
| vision_navi | ❌ | [humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | [75.0 ± 10.8%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) |
| doubledesk | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | [28.3 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) |
| football | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | [26.7 ± 4.7%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) |
| pp_box | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | [78.3 ± 8.5%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) |
| boxing | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | [70.0 ± 8.2%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) |
| open_door | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | [93.3 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) |
| sit_sofa | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | [96.7 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) |
| vision_navi | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | [76.7 ± 18.9%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) |

You can also train each task from scratch or fine-tune from a pretrained checkpoint available in our [Hugging Face model repository](https://huggingface.co/YunhengWang/kimodo-policy/tree/main). The default configuration uses **4 GPUs** (`KIMODO_GPUS=0,1,2,3`). Run the following commands from the repository root, replacing the dataset and checkpoint paths with your local paths and `doubledesk` with your target task.

**Train from scratch:**

```bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
KIMODO_GPUS=0,1,2,3 \
HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic
```

**Fine-tune from a pretrained checkpoint:**

```bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
KIMODO_GPUS=0,1,2,3 \
HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic /path/to/pretrain/checkpoint_STEP
```

### 🚀 Kimodo-Policy Multi-Task Training on HumanoidArena

To help you reproduce our multi-task HumanoidArena experiments, we provide downloadable checkpoints, evaluation results, and visualizations for policies jointly trained on all seven tasks. The table lists the following variants in order:

- **Standard:** Kimodo-G1-RP-v1 motion backbone with the default policy architecture, trained without VLA policy pretraining.

- **SEED:** The same policy architecture with the Kimodo-G1-SEED-v1 motion backbone, isolating the effect of the motion prior.

- **Large:** The RP-v1 backbone with increased ControlNet capacity, raising active policy parameters from 91.7M to 158.9M, without VLA policy pretraining.

- **105h:** The standard model fine-tuned from 105 hours of VLA policy pretraining.

- **419h:** The standard model fine-tuned from 419 hours of VLA policy pretraining, achieving the best overall success rate among these variants.

`Pretrain` refers to VLA policy pretraining; all variants retain a pretrained, frozen motion backbone. HOI and HSI report mean success rates for human–object and human–scene interaction tasks, respectively. Click a checkpoint to download the weights, or an overall result to explore the evaluations and visualizations.

| Task | Pretrain | Kimodo-Policy checkpoint | HOI Avg. | HSI Avg. | Overall Results |
| --- | --- | --- | --- | --- | --- |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 51.7% | 79.6% | [67.6%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w_SEED](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED) | 52.8% | 72.9% | [64.3%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED) |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w_large](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large) | 45.6% | 85.8% | [68.6%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large) |
| multi-task | ✅ 105h | [ft_105h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_105h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 48.9% | 82.5% | [68.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/ft_105h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |
| multi-task | ✅ 419h | [ft_419h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 56.1% | 84.2% | [72.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |

You can also train a single policy jointly on all seven HumanoidArena tasks from scratch or fine-tune from a pretrained checkpoint available in our [Hugging Face model repository](https://huggingface.co/YunhengWang/kimodo-policy/tree/main). The default configuration uses **4 GPUs** (`KIMODO_GPUS=0,1,2,3`). Run the following commands from the repository root, replacing the dataset and checkpoint paths with your local paths.

**Train from scratch:**

```bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
KIMODO_GPUS=0,1,2,3 \
HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh
```

**Fine-tune from a pretrained checkpoint:**

```bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
KIMODO_GPUS=0,1,2,3 \
HUMANOID_ARENA_ROOT=/path/to/HumanoidArena_dataset_v3_1 \
bash scripts/HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh \
  /path/to/pretrain/checkpoint_STEP
```

### 🧪 Evaluation

We evaluate Kimodo-Policy on HumanoidArena with SONIC. The default evaluation setup uses **3 GPUs** (`--gpus 0,1,2`). The Isaac Sim/Isaac Lab simulator runs in `unitree_sim_env`, while the Kimodo inference server runs in `kimodo-policy`. Install both environments as described in **Installation & Environment Setup**, then configure **Runtime Paths** above.

#### • SONIC evaluation

Run the command below from the repository root to evaluate `doubledesk`. Replace the task name and checkpoint path with your own, and adjust `--gpus` if you use different GPU IDs. The example uses seeds `0,1,2`, disables RTC and persistent simulator reuse, and enables deterministic simulator startup.

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_STEP \
  --gpus 0,1,2 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_doubledesk"
```

To evaluate multiple tasks, use `evaluation/humanoidarena_eval_multi_task_sonic.sh`. See [HumanoidArena Evaluation](evaluation/humanoidarena_eval.md) for task selection, video recording, and result aggregation.

Use a separate `--results-dir` for each evaluation run. Summary statistics, per-episode results, and optional videos are saved there. The checkpoint must be a Kimodo-Policy training checkpoint; SONIC simulator assets are configured separately.

> [!NOTE]
>
> HumanoidArena is built on Isaac Sim. Small differences in visual rendering may lead to slight variations in evaluation results.

## 🦾 Real-World G1 Workflow

The [real-world guide](real-world/README.md) covers PICO teleoperation, synchronized recording, dataset validation and transfer, fine-tuning, and remote policy deployment to a Unitree G1 with Dex3 hands.

The policy runs on a model-server GPU; the workstation runs the bridge and SONIC controller. This workflow does not require Isaac Sim or Isaac Lab. Start with the [observed versions and required assets](real-world/docs/00-versions-assets.md), then follow the [installation guide](real-world/docs/01-install.md) and [terminal-by-terminal deployment commands](real-world/deploy/start.md).

The old `real-world/GR00T-WholeBodyControl` submodule is replaced by a pinned runtime bundle installed under `real-world/external/`. Use a fresh checkout when migrating. The runtime archive and task checkpoints are not bundled in Git, and their public download locations still need to be supplied. Offline validation does not replace clean Linux and on-site hardware acceptance; see the [validation record](real-world/docs/verification.md).

## 📄 License

The code in this repository is released under the [MIT License](LICENSE).
