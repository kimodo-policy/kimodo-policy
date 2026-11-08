# HumanoidArena Evaluation

This guide evaluates the released Kimodo-Policy checkpoints with the SONIC controller. Run all commands from the repository root. The standard evaluation configuration uses **3 GPUs (`0,1,2`)**, three seeds (`0,1,2`), and 20 episodes per seed: **60 episodes per task**. Change the GPU IDs to match your machine.

## 1. Install the environments

Follow [Installation & Environment Setup](../README.md#installation-environment-setup), or install only the shared model environment and Arena runtime:

```bash
# Review and accept the Isaac Sim license before using this option.
bash setup.sh --only arena --accept-isaac-eula
source .setup/env.sh
conda activate kimodo-policy
```

The policy server uses `kimodo-policy` (Python 3.10). The Isaac Sim / Isaac Lab simulator uses `unitree_sim_env` (Python 3.11). The generated `.setup/env.sh` selects their Python executables and the runtime paths; the evaluation wrappers launch each process in the correct environment. Load this file again in every new terminal. You do not need to activate the simulator environment manually.

## 2. Prepare weights and simulation assets

Before running an evaluation:

1. Download the **Kimodo-G1-RP-v1 backbone and DINOv3 encoder** using the commands in [Prepare Runtime Assets](../README.md#prepare-runtime-assets). Preserve the complete folders under `checkpoints/`, including configurations and motion statistics. For a SEED policy, use its matching motion backbone.
2. Download a **task-specific Kimodo-Policy checkpoint** from the [single-task or multi-task tables](../README.md#-kimodo-policy-single-task-training-on-humanoidarena). Pass the actual `checkpoint_STEP` directory containing `config.json` and `training_state.pt`, rather than its parent experiment directory. A single-task policy should be evaluated on its corresponding task; use a jointly trained policy for multi-task evaluation.
3. Prepare **Arena simulation assets and SONIC controller files** following [HumanoidArena Data Preparation](../README.md#-humanoidarena-training--evaluation). Use the official downloads first, or our frozen backup when the upstream files have changed. Training demonstrations alone do not include all evaluation resources.
4. Keep the bundled task text embeddings under `data/cache/`. They are included in this repository; no text encoder download is needed for the released task instructions.

Training demonstrations are needed for training, but are not required just to evaluate a released policy. Policy checkpoints, the motion backbone, the vision encoder, and SONIC controller files serve different roles and are not interchangeable.

## 3. Check the launch configuration

Set a real checkpoint path, then validate the paths and launch arguments without starting the simulator:

```bash
export CHECKPOINT=/path/to/checkpoint_STEP
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint "$CHECKPOINT" \
  --gpus 0 \
  --seeds 0 \
  --repeats 1 \
  --max-steps 30 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_smoke" \
  --dry-run
```

After this passes, remove `--dry-run` for a short end-to-end smoke test. The 30-step limit is only for checking startup and communication; do not use its success rate as a benchmark result. A successful dry run validates configuration and paths, not simulator execution or model inference.

## 4. Evaluate a single task

The example below matches the main README. Use a fresh results directory for each run. Leave `--max-steps` unset for formal evaluation so the wrapper reads the selected task's configured episode length.

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
bash evaluation/humanoidarena_eval_signal_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task doubledesk \
  --checkpoint "$CHECKPOINT" \
  --gpus 0,1,2 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --record-video-every-n 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_doubledesk"
```

`--record-video-every-n 1` records every episode; use a larger interval to reduce disk usage, or `0` to disable recording. The main README omits this option and uses the wrapper's recording default. RTC is disabled, each episode uses a separate simulator process, and deterministic startup is enabled. When `--rtc 0` is set, RTC overlap, frozen-frame, and ramp settings do not affect trajectory blending.

## 5. Evaluate a multi-task policy

The multi-task wrapper evaluates tasks sequentially in the order requested, using the selected GPUs for the seed jobs within each task. Supported groups are:

| Group | Tasks, in evaluation order |
| --- | --- |
| `hoi` | `doubledesk`, `football`, `pp_box` |
| `hsi` | `boxing`, `open_door`, `sit_sofa`, `vision_navi` |
| `all` | All three HOI tasks, followed by all four HSI tasks |

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
bash evaluation/humanoidarena_eval_multi_task_sonic.sh \
  --project "$PROJECT_ROOT" \
  --task all \
  --checkpoint /path/to/multi_task/checkpoint_STEP \
  --gpus 0,1,2 \
  --seeds 0,1,2 \
  --repeats 20 \
  --dtype fp32 \
  --diffusion-steps 10 \
  --execution-frames 15 \
  --rtc 0 \
  --persistent-sim 0 \
  --deterministic-eval 1 \
  --record-video-every-n 1 \
  --results-dir "$PROJECT_ROOT/eval_results/arena_multi_task"
```

Use `--task hoi`, `--task hsi`, or an explicit list such as `--task doubledesk football pp_box` to select a subset. Add `--dry-run` to validate all selected task launch plans. The full seven-task configuration runs 420 episodes in total.

## 6. Inspect results

For a single task, the results directory contains:

- `aggregate_summary.json`: total episodes, successes, overall success rate, failure reasons, and per-seed statistics.
- `seed_0/`, `seed_1/`, `seed_2/`: individual seed outputs, including `summary.json`, `final.csv`, episode records, and `pipeline.log`.
- Optional videos under each seed's `videos/success/` and `videos/failure/` directories.

For a multi-task run, each task has its own subdirectory and aggregate summary. For example:

```bash
python -m json.tool eval_results/arena_doubledesk/aggregate_summary.json
python -m json.tool eval_results/arena_multi_task/doubledesk/aggregate_summary.json
```

Check the wrapper's exit status and any `.failed` markers before treating a result as complete. If a run fails, inspect the relevant seed's `pipeline.log`. Use distinct results directories and port ranges when launching independent evaluations concurrently (`--port-base` sets the starting port).

> [!NOTE]
> HumanoidArena is built on Isaac Sim. Small differences in visual rendering and simulation may lead to slight variations in evaluation results, even with the same seeds and deterministic startup settings.
