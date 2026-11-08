# Released Real-World Checkpoints and Training Data

Run these commands from the **repository root on the model/training host**, after installing `kimodo-policy`. The eight evaluated tasks provide 236 demonstrations and 157,769 frames at 30 Hz (approximately 1.46 hours). Keep each dataset's `meta/`, `data/`, and `videos/` together. Text embeddings remain in Git.

| Task | Checkpoint step | Demonstrations | Policy checkpoint | Training data |
| --- | ---: | ---: | --- | --- |
| `banana` | 20,000 | 33 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-banana/2026-10-15_15-49-05/checkpoint_20000) | [`teleop-banana`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-banana) |
| `boxing` | 25,000 | 29 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-boxing/2026-10-20_01-00-06/checkpoint_25000) | [`teleop-boxing`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-boxing) |
| `pushing` | 25,000 | 25 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-pushing/2026-10-20_00-54-33/checkpoint_25000) | [`teleop-pushing`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-pushing) |
| `transportation` | 30,000 | 24 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-transportation/2026-10-20_01-01-28/checkpoint_30000) | [`teleop-transportation`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-transportation) |
| `sitting` | 20,000 | 30 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-sitting/2026-10-20_02-50-36/checkpoint_20000) | [`teleop-sitting`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-sitting) |
| `football` | 30,000 | 38 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-football/2026-10-20_09-32-25/checkpoint_30000) | [`teleop-football`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-football) |
| `pnp` | 25,000 | 28 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-pnp/2026-10-20_09-36-05/checkpoint_25000) | [`teleop-pnp`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-pnp) |
| `long` | 30,000 | 29 | [Download](https://huggingface.co/YunhengWang/kimodo-policy/tree/main/Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-long/2026-10-20_09-36-21/checkpoint_30000) | [`teleop-long`](https://huggingface.co/datasets/YunhengWang/kimodo-policy/tree/main/data/RealWorld/teleop-long) |

## Download one task

This example downloads `banana` and its selected step-20,000 policy. For another task, replace both the task and checkpoint directory using the table above.

```bash
conda activate kimodo-policy
export TASK=banana
export POLICY_DIR="Real_World/ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_teleop-banana/2026-10-15_15-49-05/checkpoint_20000"
hf download YunhengWang/kimodo-policy \
  --include "$POLICY_DIR/**" --local-dir .
export CHECKPOINT="$PWD/$POLICY_DIR"

hf download YunhengWang/kimodo-policy --repo-type dataset \
  --include "data/RealWorld/teleop-$TASK/**" \
  --local-dir .setup/real-world-data
mkdir -p real-world/data
cp -a ".setup/real-world-data/data/RealWorld/teleop-$TASK" real-world/data/
```

For training, set `REAL_WORLD_ROOT="$PWD/real-world"` and pass `teleop-banana` to the real-world launcher. For deployment, set `tasks.banana.checkpoint` in `real-world/deploy/deploy_config.json` to the **absolute path** printed by `echo "$CHECKPOINT"`. Use an absolute path for every selected task to avoid accidentally selecting another training run.

## Download all eight tasks

```bash
conda activate kimodo-policy
mkdir -p real-world/data
for TASK in banana boxing pushing transportation sitting football pnp long; do
  hf download YunhengWang/kimodo-policy \
    --include "Real_World/*_teleop-$TASK/**" --local-dir .
  hf download YunhengWang/kimodo-policy --repo-type dataset \
    --include "data/RealWorld/teleop-$TASK/**" --local-dir .setup/real-world-data
  cp -a ".setup/real-world-data/data/RealWorld/teleop-$TASK" real-world/data/
done
```

The checkpoint files total approximately **8.81 GB** and the training data **1.54 GB** (decimal units), excluding shared base models and the workstation runtime. The data copy retains the download cache, so reserve additional disk space for both copies. Smoke-test runs, unrelated tasks, and internal recording/merge logs are excluded.

## Verify downloads

The [release manifest](../manifests/released-assets.json) records each checkpoint, dataset, file size, and SHA-256 hash. The example below checks all downloaded files for the selected task, including the installed training data:

```bash
export TASK=banana
python - <<'PY_VERIFY'
import hashlib, json, os
from pathlib import Path
root = Path.cwd()
manifest = json.loads((root / "real-world/manifests/released-assets.json").read_text())
task = next(t for t in manifest["tasks"] if t["task"] == os.environ["TASK"])
checked = 0
for kind, prefix in [("models", task["checkpoint_path"] + "/"),
                     ("datasets", "data/RealWorld/" + task["dataset"] + "/")]:
    for name, expected in manifest["files"][kind].items():
        if not name.startswith(prefix):
            continue
        path = root / name if kind == "models" else root / "real-world/data" / name.removeprefix("data/RealWorld/")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert path.stat().st_size == expected["size"], f"Size mismatch: {path}"
        assert digest.hexdigest() == expected["sha256"], f"Hash mismatch: {path}"
        checked += 1
assert checked > 0
print(f"Verified {checked} files for {task['task']}")
PY_VERIFY
```

Each policy includes `config.json`, `training_state.pt`, and eight rank RNG files. Only machine-specific path metadata was made portable; learned model tensors and dataset contents are unchanged. The frozen Kimodo-G1-RP-v1 backbone and DINOv3 weights are separate dependencies; download them using [Prepare Runtime Assets](../../README.md#prepare-runtime-assets).

These checks establish file integrity. They do not replace task-pose calibration, camera/PICO setup, or an on-site hardware check. Robot execution additionally requires the matching workstation runtime archive, whose public distribution remains pending. Continue with the [installation guide](01-install.md), [training guide](03-collection-training.md), or [deployment guide](../deploy/start.md).
