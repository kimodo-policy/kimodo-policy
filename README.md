<h1 align="center">
  Kimodo-Policy: From Text-to-Motion Generators to<br/>
  Humanoid Vision-Language-Action Policies
</h1>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://kimodo-policy.github.io/"><img src="https://img.shields.io/badge/Project_Page-Kimodo--Policy-blue?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href=""><img src="https://img.shields.io/badge/HuggingFace-Checkpoints-yellow?logo=huggingface&logoColor=black" alt="Hugging Face checkpoints"></a>
  <a href=""><img src="https://img.shields.io/badge/HuggingFace-Datasets-yellow?logo=huggingface&logoColor=black" alt="Hugging Face datasets"></a>
</p>

<p align="center">
  <img src="asset/picture/teaser.png" alt="Kimodo-Policy overview" width="96%">
</p>

Kimodo-Policy is a vision-conditioned policy for whole-body manipulation with the G1 humanoid. This repository contains the policy model, training launchers, dataset adapters, and simulation evaluation tools for HumanoidArena and SIMPLE. Large model checkpoints and datasets are distributed separately.

The policy is built on a frozen Kimodo motion backbone. DINOv3 extracts image features, and a ControlNet injects visual conditioning into the action denoising network. Hand control is handled by a separate head. The standard configurations run at 30 FPS with a 100-frame history and a 50-frame action chunk; refer to the selected YAML file for exact settings.

- Code: [Yunheng-Wang/kimodo-policy](https://github.com/Yunheng-Wang/kimodo-policy)
- Pretrained and task checkpoints: [Hugging Face model repository](https://huggingface.co/YunhengWang/kimodo-policy/tree/main)
- Training launchers: [scripts/](scripts/)
- Evaluation guides: [HumanoidArena](evaluation/humanoidarena_eval.md) and [SIMPLE](evaluation/simple_eval.md)

## Contents

- [HumanoidArena Training & Evaluation](#humanoidarena-training--evaluation)

## HumanoidArena Training & Evaluation

This section covers training and evaluation on HumanoidArena. Training runs in the Kimodo Conda environment. Evaluation uses the HumanoidArena simulator in unitree_sim_env and the Kimodo model server in lerobot.

### Environment Setup

For training, activate the Kimodo environment and point the dataset variable to the local HumanoidArena dataset:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
test -d "$HUMANOID_ARENA_ROOT" || {
  echo "Missing HumanoidArena dataset: $HUMANOID_ARENA_ROOT" >&2
  exit 2
}
~~~

For evaluation, keep the simulator and model server in separate environments:

~~~bash
export CONDA_BASE="$(conda info --base)"
export HUMANOIDARENA_ROOT="$PROJECT_ROOT/HumanoidArena"
export KIMODO_SIM_ENV="$CONDA_BASE/envs/unitree_sim_env"
export KIMODO_SERVER_PYTHON="$CONDA_BASE/envs/lerobot/bin/python"
export SONIC_POLICY_ROOT="$HUMANOIDARENA_ROOT/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
export TEXT_EMBEDDING_CACHE="$PROJECT_ROOT/data/cache/HumanoidArena"
export OMNI_KIT_ACCEPT_EULA=YES
~~~

Before a full run, verify that the simulator Python, model-server Python, SONIC artifacts, text cache, and Isaac Lab task YAML are available. The evaluation wrappers provide dry-run validation without launching Isaac Sim.

### Data Preparation

Download the official [HumanoidArena_dataset_v3_1 dataset](https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1/tree/main) before training. From the repository root, download it into the default directory used by the Arena YAML files:

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python -m pip install --upgrade huggingface_hub
hf download WilliamWang16/HumanoidArena_dataset_v3_1 \
  --repo-type dataset \
  --local-dir "$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"
~~~

If the dataset is gated or requires authentication, run hf auth login first.

Audit the dataset before training:

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python utils/check_datasets.py \
  --config scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.yaml \
  --workers 1 \
  --limit 5 \
  --output "$PROJECT_ROOT/log/humanoidarena_dataset_audit.json"
~~~

The checker validates episode metadata, task selection, and action continuity. It does not replace a small GPU run that verifies video decoding and memory usage.

### Kimodo-Policy Checkpoints

This release provides the trained Kimodo-Policy weights used for the HumanoidArena experiments. The first table lists policies trained independently on each Arena task, either from the large-scale Kimodo-Policy pretraining stage or directly from scratch. The second table lists policies trained jointly on all seven Arena tasks, again with either scratch or 419h-pretrained initialization. The `Pretrain` column identifies the initialization, and the reported results are taken from Tables 1 and 2 of the paper.

| Task | Pretrain | Kimodo-Policy checkpoint | Results&nbsp;(SR) |
| --- | --- | --- | --- |
| doubledesk | ✗ | [humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | 40.0 ± 0.0% |
| doubledesk | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | 28.3 ± 2.4% |
| football | ✗ | [humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | 28.3 ± 6.2% |
| football | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | 26.7 ± 4.7% |
| pp_box | ✗ | [humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | 80.0 ± 4.1% |
| pp_box | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | 78.3 ± 8.5% |
| boxing | ✗ | [humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | 73.3 ± 13.1% |
| boxing | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | 70.0 ± 8.2% |
| open_door | ✗ | [humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | 90.0 ± 4.1% |
| open_door | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | 93.3 ± 2.4% |
| sit_sofa | ✗ | [humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | 100.0 ± 0.0% |
| sit_sofa | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | 96.7 ± 2.4% |
| vision_navi | ✗ | [humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 75.0 ± 10.8% |
| vision_navi | ✓ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | 76.7 ± 18.9% |

#### HumanoidArena Multi-Task Checkpoints

Each row below is one policy jointly trained on the seven HumanoidArena tasks. The Results column reports the Overall SR from Table 2.

| Task | Pretrain | Kimodo-Policy checkpoint | HOI Avg. | HSI Avg. | Overall Results |
| --- | --- | --- | --- | --- | --- |
| multi-task | ✗ | [humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 51.7 ± 20.7% | 79.6 ± 16.5% | 67.6% |
| multi-task | ✓ | [ft_419h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 56.1 ± 24.2% | 84.2 ± 14.1% | 72.1% |

### Training on Arena

All Arena launchers use BF16 and Accelerate. KIMODO_GPUS controls the visible GPUs, and the global batch size changes when the number of GPUs changes. Set a different KIMODO_MASTER_PORT for simultaneous jobs on one host.

#### Multi-task SONIC training from scratch

This configuration trains one policy on the seven-task Sonic RefPose mixture for 500,000 steps and saves to log/HumanoidArena_Multi_Task/ by default:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh
~~~

Use the SEED, large_hand4, or large_hand8 launcher in the same directory for the corresponding model variant.

#### Multi-task fine-tuning from a Kimodo checkpoint

Pass a complete pretraining checkpoint as the only positional argument:

~~~bash
INIT_CKPT=/path/to/checkpoint_1000000
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/ft_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh \
  "$INIT_CKPT"
~~~

The 105h and 419h released fine-tuning families use the same launcher interface; select the matching YAML through KIMODO_CONFIG when reproducing a particular published run.

#### Single-task training

Train one task from scratch:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/datasets/HumanoidArena_dataset_v3_1"

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic
~~~

Fine-tune the same task from a pretrained checkpoint:

~~~bash
KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Single_Task/ft_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse.sh \
  doubledesk sonic /path/to/checkpoint_1000000
~~~

Supported task aliases include doubledesk, football, pp_box, boxing, open_door, sit_sofa, and vision_navi. The script maps these aliases to the task names used by the dataset adapter. Outputs are saved under log/HumanoidArena_Single_Task/ with the task and backend included in the run name.

### Evaluation

Evaluation starts two processes: an Isaac Sim/Isaac Lab worker from unitree_sim_env and a Kimodo inference server from lerobot. Set the environment variables from the Environment Setup subsection first. The checkpoint passed to the wrapper must be a Kimodo training checkpoint; SONIC and TWIST2 simulator assets are separate files.

#### SONIC evaluation

The following command evaluates one task with three GPUs, deterministic simulator startup, and no persistent simulator reuse:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_doubledesk"
~~~

For continuous multi-task evaluation, use evaluation/humanoidarena_eval_multi_task_sonic.sh and pass the task list in the order required by the benchmark. See [HumanoidArena Evaluation](evaluation/humanoidarena_eval.md) for video recording, RTC, seeds, and result aggregation.

#### TWIST2 evaluation

TWIST2 uses the same Kimodo server environment but requires the TWIST2 ONNX asset and the task-specific Isaac Lab YAML:

~~~bash
bash evaluation/humanoidarena_eval_signal_task_twist2.sh \
  --project "$PROJECT_ROOT" \
  --task football \
  --checkpoint /path/to/checkpoint_200000 \
  --gpus 5,6,7 \
  --seeds 0,1,2 \
  --repeats 20 \
  --results-dir "$PROJECT_ROOT/eval_results/twist2_football"
~~~

Every evaluation run should have its own results directory. The wrappers write summary statistics, per-episode outputs, and optional videos there. A failed preflight usually indicates a missing simulator asset, cache file, environment variable, or incomplete checkpoint; the wrapper prints the missing path before launching the worker.
