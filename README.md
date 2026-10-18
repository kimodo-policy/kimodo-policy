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
  <sub><sup>1</sup>HKUST(GZ)&thinsp;·&thinsp;<sup>2</sup>JD Explore Academy&thinsp;·&thinsp;<sup>3</sup>BJUT&thinsp;·&thinsp;<sup>4</sup>Cambridge</sub><br>
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

<p align="center">
  <img src="asset/picture/teaser.png" alt="Kimodo-Policy overview: benchmark results, motion-prior-based architecture, and whole-body robot execution" width="100%">
</p>

## Open-Source Roadmap

- [x] [HumanoidArena Training & Evaluation](#humanoidarena-training--evaluation)
- [x] SIMPLE Benchmark: [Training](scripts/Simple_Single_Task/) & [Evaluation](evaluation/simple_eval.md)
- [ ] Real-World Data Collection, Training & Deployment
- [ ] More Benchmarks (Coming Soon)

## HumanoidArena Training & Evaluation

This section covers training and evaluation on HumanoidArena. Training runs in the Kimodo Conda environment. Evaluation uses the HumanoidArena simulator in unitree_sim_env and the Kimodo model server in lerobot.

### Environment Setup

For training, activate the Kimodo environment and point the dataset variable to the local HumanoidArena dataset:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"
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

Download the released [HumanoidArena_dataset_v3_1 dataset](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/HumanoidArena_dataset_v3_1) before training. From the repository root, download the dataset subtree and point the Arena YAML files to it through HUMANOID_ARENA_ROOT:

~~~bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python -m pip install --upgrade huggingface_hub
hf download YunhengWang/kimodo-policy \
  --repo-type dataset \
  --include "data/HumanoidArena_dataset_v3_1/**" \
  --local-dir "$PROJECT_ROOT"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"
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

This release provides the trained Kimodo-Policy weights used for the HumanoidArena experiments. The first table lists policies trained independently on each Arena task, either from the 419h pretraining stage or directly from scratch. The second table lists policies trained jointly on all seven Arena tasks, including scratch variants and 105h/419h fine-tuning. The `Pretrain` column identifies the initialization. Single-task results and the two original multi-task entries are from Tables 1 and 2 of the paper; additional multi-task entries summarize the released evaluation files. Click a result to open its corresponding evaluation folder.

| Task | Pretrain | Kimodo-Policy checkpoint | Results&nbsp;(SR) |
| --- | --- | --- | --- |
| doubledesk | ❌ | [humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | [40.0 ± 0.0%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) |
| doubledesk | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_double_desk](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) | [28.3 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_double_desk_sonic) |
| football | ❌ | [humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | [28.3 ± 6.2%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) |
| football | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_football](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) | [26.7 ± 4.7%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_football_sonic) |
| pp_box | ❌ | [humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | [80.0 ± 4.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) |
| pp_box | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HOI_pp_box](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) | [78.3 ± 8.5%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HOI_pp_box_sonic) |
| boxing | ❌ | [humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | [73.3 ± 13.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) |
| boxing | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_boxing](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) | [70.0 ± 8.2%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_boxing_sonic) |
| open_door | ❌ | [humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | [90.0 ± 4.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) |
| open_door | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_open_door](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) | [93.3 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_open_door_sonic) |
| sit_sofa | ❌ | [humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | [100.0 ± 0.0%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) |
| sit_sofa | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_sit_sofa](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) | [96.7 ± 2.4%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_sit_sofa_sonic) |
| vision_navi | ❌ | [humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | [75.0 ± 10.8%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) |
| vision_navi | ✅ | [ft_419h_humanoidarena_single_gbs64_20w_HSI_vision_navi](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) | [76.7 ± 18.9%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Single_Task/ft_419h_humanoidarena_single_gbs64_20w_controlnet4_detach_true_mse_HSI_vision_navi_sonic) |

#### HumanoidArena Multi-Task Checkpoints

Each row below is one policy jointly trained on the seven HumanoidArena tasks. HOI averages cover doubledesk, football, and pp_box; HSI averages cover boxing, open_door, sit_sofa, and vision_navi. The averages and population standard deviations are computed across task/seed success rates within each group. Overall SR pools all seven tasks. Results link to the released evaluation folders; an unavailable result is marked as not published.

| Task | Pretrain | Kimodo-Policy checkpoint | HOI Avg. | HSI Avg. | Overall Results |
| --- | --- | --- | --- | --- | --- |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 51.7 ± 20.7% | 79.6 ± 16.5% | [67.6%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w_SEED](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED) | 52.8 ± 22.4% | 72.9 ± 15.2% | [64.3%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_SEED) |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w_large](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large) | 45.6 ± 19.4% | 85.8 ± 15.3% | [68.6%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large) |
| multi-task | ❌ | [humanoidarena_sonicx7_gbs128_50w_large_hand8](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse_large_hand8) | — | — | Not published |
| multi-task | ✅ 105h | [ft_105h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_105h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 48.9 ± 23.4% | 82.5 ± 13.0% | [68.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/ft_105h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |
| multi-task | ✅ 419h | [ft_419h_humanoidarena_sonicx7_gbs128_50w](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) | 56.1 ± 24.2% | 84.2 ± 14.1% | [72.1%](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/HumanoidArena_Multi_Task/ft_419h_humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse) |

### Training on Arena

All Arena launchers use BF16 and Accelerate. KIMODO_GPUS controls the visible GPUs, and the global batch size changes when the number of GPUs changes. Set a different KIMODO_MASTER_PORT for simultaneous jobs on one host.

#### Multi-task SONIC training from scratch

This configuration trains one policy on the seven-task Sonic RefPose mixture for 500,000 steps and saves to log/HumanoidArena_Multi_Task/ by default:

~~~bash
conda activate kimodo
export KIMODO_ENV="$CONDA_PREFIX"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"

KIMODO_GPUS=0,1,2,3 \
  bash scripts/HumanoidArena_Multi_Task/humanoidarena_sonicx7_gbs128_50w_controlnet4_detach_true_mse.sh
~~~

Use the SEED, large, or large_hand8 launcher in the same directory for the corresponding model variant.

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
export HUMANOID_ARENA_ROOT="$PROJECT_ROOT/data/HumanoidArena_dataset_v3_1"

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

## License

The code in this repository is released under the [MIT License](LICENSE).
