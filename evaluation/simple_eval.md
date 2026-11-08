# SIMPLE training and evaluation

Run all commands from the Kimodo-Policy repository root. This guide uses our
vendored SIMPLE integration, the `kimodo-policy` Conda environment for training,
and `SIMPLE/.venv` for simulation. Do not replace the vendored integration with a
new upstream checkout when reproducing this release.

## 1. Environment

Use Ubuntu 22.04 or newer for a fresh installation: the locked dependencies
require glibc 2.34 or newer. Visual evaluation needs an RTX-capable GPU and a
working NVIDIA graphics driver; see NVIDIA's
[GPU requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
The release smoke test used an RTX 4090. Training alone does not need RTX rendering.

```bash
bash setup.sh --only simple --system-deps --accept-isaac-eula
source .setup/env.sh
conda activate kimodo-policy
```

Review the Isaac Sim license before passing `--accept-isaac-eula`. Training alone
only needs `bash setup.sh --only training`; Isaac Sim and cuRobo are evaluation
dependencies. The installer builds CUDA extensions locally. Python environments,
`.uv-cache`, downloaded Python runtimes, `__pycache__`, and compiled `.so` files
are not release assets. SIMPLE needs a CUDA 12.x toolkit (`nvcc`, preferably 12.8)
and NVIDIA graphics libraries, including EGL and Vulkan. In a container, expose
both compute and graphics driver capabilities; `nvidia-smi` alone does not prove
that Isaac rendering is available.

Use `source .setup/env.sh` in each new terminal. It sets `KIMODO_MODEL_SITE_PACKAGES`
to the training environment so the evaluator can load the model dependencies
while keeping SIMPLE's own PyTorch/cuRobo runtime.

## 2. Base models, task policy and text embeddings

The task checkpoint contains the trained policy modules. The frozen Kimodo
backbone, motion statistics and DINOv3 encoder are separate dependencies:

```bash
hf download YunhengWang/kimodo-policy \
  --include 'checkpoints/Kimodo-G1-RP-v1/**' \
            'checkpoints/dinov3-vitl16-pretrain-lvd1689m/**' \
  --local-dir .
```

For a complete Close Door example, download the released task checkpoint:

```bash
export TASK=G1WholebodyCloseDoorTeleop-v0
POLICY_DIR="Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand_${TASK}/2026-09-20_15-15-36/checkpoint_200000"
hf download YunhengWang/kimodo-policy --include "$POLICY_DIR/**" --local-dir .
export CHECKPOINT="$PWD/$POLICY_DIR"
```

Choose other policies from [Simple_Single_Task](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Simple_Single_Task)
and use their actual timestamp/checkpoint directory. `config.json` and
`training_state.pt` must be in the same directory. The eight released Teleop
policies were fine-tuned from the 419h pretraining checkpoint for 200,000 steps.
There are no released MP task policies in this collection.

The eight evaluated-task text embeddings are bundled in Git under `data/cache/Simple/` and are
ready to use after cloning; no Hugging Face download is needed. Training and
evaluation of these instructions do not need the full Llama/LLM2Vec weights.
For new instructions, download the configured `Meta-Llama-3-8B-Instruct` and both
`LLM2Vec-Meta-Llama-3-8B-Instruct-mntp*` folders from the
[base model collection](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/checkpoints)
and let training generate the missing caches. Changing the instruction requires
a matching embedding, not just renaming a `.pt` file.

## 3. Evaluation resources

Official sources are the [SIMPLE asset repository](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE)
and [fixed evaluation episodes](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).
For the frozen resource versions assembled for this release, use our
[backup collection](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/assets/simple_assets):

```bash
python utils/prepare_simple.py --assets
export SIMPLE_DATA_DIR="$PWD/SIMPLE/data"
export EVAL_DATA_ROOT="$SIMPLE_DATA_DIR/simple-eval"
```

The helper pins a revision, verifies SHA-256 and extracts the archives. It
reuses downloaded archives on subsequent runs. To use another disk, add
`--data-dir /path/to/simpledata`, then export `SIMPLE_DATA_DIR` and `EVAL_DATA_ROOT`
accordingly. The final layout is:

```text
SIMPLE/data/
├── assets/{graspnet,articulated,objects}/
├── robots/{g1,g1_sonic,...}/
├── scenes/hssd/<six room directories>/
├── vMaterials_2/
└── simple-eval/<TASK>/dr-level-{0,1,2}/meta/episodes.jsonl
data/cache/Simple/*.pt  # already included in Git
```

The frozen episodes cover 12 tasks, with 10 episodes at each of three levels.
The original `scenes_hssd.tar.gz` contains only one room: also install the five
additional room archives, articulated objects and containers. The helper installs
them all. Teleop WBC ONNX controllers are included in the vendored source tree;
MP AMO weights are in `robots.tar.gz`. HSSD paths are resolved in a per-scene user
cache; no `/ai` or `/data/local-data` directory is required.

## 4. Training data and commands

Prefer the [official task demonstrations](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple):

```bash
export TASK=G1WholebodyCloseDoorTeleop-v0
hf download USC-PSI-Lab/psi-data "simple/$TASK.zip" \
  --repo-type dataset --local-dir datasets/upstream
mkdir -p datasets/Simple
unzip "datasets/upstream/simple/$TASK.zip" -d datasets/Simple
export KIMODO_SIMPLE_ROOT="$PWD/datasets/Simple"
```

If the official data changes or the task is unavailable, use the
[frozen training snapshots](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/Simple):

```bash
python utils/prepare_simple.py --training-task "$TASK"
```

There are eight training snapshots (794 demonstrations), restricted to the
evaluated tasks and including
`G1WholebodyXMoveBendPickTeleop-v0`, which is absent from the current official
training directory. The loader reads the LeRobot `meta/`, `data/` and `videos/`
layout directly; no simulation replay or conversion is needed.

Check the selected task before training:

```bash
KIMODO_SIMPLE_TASK="$TASK" python utils/check_datasets.py \
  --config scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.yaml \
  --workers 4 --fail-on-errors --output simple_data_check.json
```

This checks metadata, parquet shapes/values/task indices and video presence. It
does not decode every video frame or apply Arena's floating-base continuity
thresholds to SIMPLE's command-derived root motion.

Fine-tune from a [pretraining checkpoint](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Pre_Train),
replacing the checkpoint path below with its downloaded directory:

```bash
conda activate kimodo-policy
export KIMODO_ENV="$CONDA_PREFIX"
KIMODO_GPUS=0,1,2,3 \
  bash scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.sh \
  "$TASK" /path/to/pretrain/checkpoint_1000000
```

The default is four GPUs, batch 16 per GPU (global batch 64), and 200,000 steps.
Set `KIMODO_GPUS` for the available devices; changing their number changes the
global batch size. `KIMODO_ENV` selects the environment containing `accelerate`.

To initialize the trainable policy modules from scratch, omit `--init-checkpoint`:

```bash
export KIMODO_SIMPLE_TASK="$TASK"
export KIMODO_SIMPLE_SAVE_ROOT="log/Simple_Scratch/$TASK"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  train.py --config scripts/Simple_Single_Task/ft_simple_single_gbs64_20w_controlnet4_detach_true_mse_continuous_hand.yaml
```

Here “scratch” refers to the trainable policy modules; the frozen Kimodo motion
backbone and visual encoder are still required. Use `train.py --resume` for
resuming an interrupted training run; initialization and resume are different.

## 5. Evaluation

With `TASK`, `CHECKPOINT`, resource paths and model files prepared as above:

```bash
# Match the execution horizon used in each released evaluation.
case "$TASK" in
  G1WholebodyOpenFaucetTeleop-v0|G1WholebodyPushOfficeChairTeleop-v0|G1WholebodyXMoveBendPickTeleop-v0)
    EXECUTION_FRAMES=50 ;;
  *) EXECUTION_FRAMES=15 ;;
esac

bash evaluation/simple_eval_signal_task_sonic.sh \
  --task "$TASK" --checkpoint "$CHECKPOINT" \
  --gpus 0,1,2 --levels 0,1,2 --seeds 0 --episodes 10 \
  --simple-data-dir "$SIMPLE_DATA_DIR" \
  --eval-data-root "$EVAL_DATA_ROOT" \
  --dtype fp32 --diffusion-steps 10 --execution-frames "$EXECUTION_FRAMES" \
  --rtc 0 --max-navigation-speed 10.0 --max-steps auto
```

This example uses three GPUs, one for each level. Use `--gpus 0` to run the levels
sequentially on one GPU; the wrapper defaults to one GPU when the option is
omitted. Level 0/1/2 use their fixed `environment_config` values, rather than new
randomly sampled scenes. Keep the task, policy and episode data matched.

The released evaluations use FP32, 10 diffusion steps and RTC disabled. Open
Faucet, Push Office Chair and Move & Bend Pick execute 50 frames per update; the
other five released tasks execute 15. The command above selects this value from
`TASK`. The checkpoint predicts 50 frames; `--execution-frames 15` executes the
first 15 and replans. The wrapper defaults to 50, so keep the explicit task-specific
argument and consult the linked result folder's `run_config.txt`. Likewise retain
`--max-navigation-speed 10.0` (the diagnostic default is 1.5). Other RTC settings
retain their defaults: overlap 0, frozen frames 0, ramp power 1.0.

For a path/launch-plan check, append `--dry-run`. For a short end-to-end smoke
run, use `--gpus 0 --levels 0 --episodes 1 --max-steps 2`. A smoke run is not a
success-rate measurement; restore `--max-steps auto` and 10 episodes per level
for a benchmark run. Initial CUDA compilation and Isaac startup can take time.

Results are written to a new `eval_results/simple/.../` directory, or to a unique
`--results-dir` you supply. Each level contains `pipeline.log`, episode JSONs and
videos; `summary.json` aggregates the levels and `.done` marks wrapper completion.
For the full protocol, also check that all three levels contain 10 results.
Published results and visualizations are under
[Simple_Single_Task results](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/eval_results/Simple_Single_Task).

### • MP tasks

The same wrapper accepts MP task names and a compatible policy trained for that
task. It selects AMO automatically; no extra `--mp` flag is needed. MP uses its
own standing warm-up and task time limit. The bundled AMO TorchScript currently
uses `cuda:0`, so an MP evaluation can occupy physical GPU 0 in addition to the
selected rendering/model GPU. Isaac may also create contexts on visible GPUs;
`--gpus` is a device selection, not strict GPU memory isolation. These resource
constraints do not apply to training-only use.

See the [release validation record](simple_release_validation.md) for the checks
performed on this integration and the remaining validation limits.
