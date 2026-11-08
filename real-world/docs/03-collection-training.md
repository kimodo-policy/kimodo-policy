# 3. Teleoperation, Recording, Validation, and Fine-Tuning

Complete chapters 1 and 2. Open separate workstation terminals; start each with:

```bash
cd "$HOME/kimodo-policy/real-world"  # Adjust to your clone
source env.sh
```

For released-data training only, skip sections 3.1–3.6: [download and verify a dataset](released-assets.md), then follow section 3.7 on the model host.

## 3.1 Workstation terminal A: XRT service

Skip this if it is already running:

```bash
bash collection/bin/xrt-service
```

## 3.2 Workstation terminal B: PICO manager

```bash
bash collection/bin/pico-manager 1.75
```

Replace 1.75 with the operator's measured height in metres. Confirm valid PICO tracking and GMR's 29-joint mapping.

## 3.3 Workstation terminal C: SONIC

The on-site operator must support the robot and prepare the emergency stop:

```bash
bash collection/bin/sonic-control
```

Wait for `Init Done`. Follow the controller's mode procedure: ABXY switches OFF/PLANNER and AX switches PLANNER/POSE. Mode changes affect the control source. Do not run a model deployment bridge concurrently with the PICO manager.

## 3.4 Workstation terminal D: recorder

```bash
bash collection/bin/collect \
  'Pick up the banana from the white table in front of you and place it on the white table behind you.'
```

Use the exact intended instruction for other tasks. Preserve the instruction above for the original banana weights. Left grip+A starts recording or stops and saves an episode; left grip+B discards it. Wait for save completion before renaming or moving data.

Each recorder process creates `$WBC_ROOT/outputs/teleop-TIMESTAMP`. Multiple episodes from that process share the directory. Preserve `meta/`, `data/`, and `videos/` together.

## 3.5 Stop and validate

Finish and save the episode, wait for confirmation, and exit the recorder. Secure the robot and follow the local controller shutdown procedure; do not terminate all terminals together.

```bash
export DATASET="$WBC_ROOT/outputs/teleop-your_timestamp"
bash collection/bin/validate-dataset "$DATASET"
bash collection/bin/scan-datasets "$WBC_ROOT/outputs" \
  --report "$REAL_WORLD_ROOT/runtime/collection-report.json"
```

Investigate nonzero exits, missing videos, NaNs, frame/timestamp discrepancies, and source-frame anomalies. A regular 30 Hz output index alone does not prove uninterrupted source tracking; inspect synchronization/audit fields too.

## 3.6 Transfer data from the workstation

Set `SERVER_DATA` to the actual absolute path on the model server:

```bash
export SERVER_DATA=/path/to/kimodo-policy/real-world/data
export TASK=banana
ssh "$KIMODO_SERVER_SSH" "mkdir -p '$SERVER_DATA/teleop-$TASK'"
rsync -a --partial --info=progress2 "$DATASET/" \
  "$KIMODO_SERVER_SSH:$SERVER_DATA/teleop-$TASK/"
rsync -acn --itemize-changes "$DATASET/" \
  "$KIMODO_SERVER_SSH:$SERVER_DATA/teleop-$TASK/"
```

The final command is read-only and should report no content differences. Use this for a new single dataset destination. Do not merge independently numbered datasets with plain rsync: renumber episodes and update metadata/video references first. Never overwrite a different existing dataset under the same task name.

## 3.7 Model host: eight-GPU fine-tuning

Source `env.sh` in the model host's real-world directory, then:

```bash
cd "$KIMODO_PROJECT_ROOT"
export KIMODO_ENV="$(dirname "$(dirname "$KIMODO_SERVER_PYTHON")")"
export KIMODO_GPUS=0,1,2,3,4,5,6,7
export REAL_WORLD_ROOT="$KIMODO_PROJECT_ROOT/real-world"
unset KIMODO_RESUME
bash scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.sh \
  /path/to/pretrained/checkpoint_1000000 teleop-banana
```

Per-GPU batch size is 8, gradient accumulation is 1, and eight ranks produce GBS 64. The launcher's default is four GPUs, so set all eight explicitly. Independent concurrent jobs need distinct `KIMODO_MASTER_PORT` values and sufficient GPU memory.

To resume the same task with unchanged training configuration:

```bash
export KIMODO_RESUME=1
bash scripts/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse.sh \
  /path/to/task/checkpoint_STEP teleop-banana
```

`--init-checkpoint` initializes trainable weights for a new fine-tune. `--resume` restores training state and validates configuration compatibility. Do not confuse them.

The public `train.py` uses W&B (offline by default in the real-world YAML). The original private SwanLab setup is not automatically installed by this release. Configure your own logging integration separately; keep credentials outside Git.

`prepare_text_embeddings` generates/loads instruction caches under `$KIMODO_PROJECT_ROOT/data/cache/RealWorld`. Missing caches require the LLM2Vec assets. For a new task instruction, set the server's `KIMODO_TEXT_CACHE` in `local.env` to this training cache directory. Do not substitute the bundled eight-task cache for newly trained text.

After training, return to the real-world directory before deployment checks:

```bash
cd "$KIMODO_PROJECT_ROOT/real-world"
source env.sh
nano deploy/deploy_config.json  # Set the task checkpoint's absolute path
"$KIMODO_SERVER_PYTHON" tools/preflight_server.py banana
```

Replace `banana` with your task, then follow the [deployment guide](../deploy/start.md).
