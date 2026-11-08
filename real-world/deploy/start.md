# 4. Remote Policy Deployment — Terminal-by-Terminal Guide

Complete [installation](../docs/01-install.md) and [network/camera validation](../docs/02-network-camera.md). PICO is not required for model deployment; stop the PICO manager first.

Start each terminal with:

```bash
cd "$HOME/kimodo-policy/real-world"  # Adjust to this host's clone
source env.sh
export TASK=boxing
```

Supported tasks: `banana`, `boxing`, `pushing`, `transportation`, `sitting`, `football`, `pnp`, and `long`. Use the same task on both hosts. `long` maps to `teleop-long` and uses `long_initial_pose.json`. Task selection does not rewrite JSON. Configure interpreters/paths in `local.env` and checkpoint/inference settings in `deploy/deploy_config.json`.

## 4.1 Model host, terminal H: validate and start the server

Set `tasks.<task>.checkpoint` to an existing absolute checkpoint directory. Numeric defaults describe earlier runs; no task weights are bundled.

```bash
"$KIMODO_SERVER_PYTHON" tools/preflight_server.py "$TASK"
bash deploy/deploy.sh show "$TASK"
bash deploy/deploy.sh server "$TASK" --gpu 0
```

Check the selected checkpoint, instruction, RTC, and execution settings. In another server terminal:

```bash
curl --fail http://127.0.0.1:18080/health
```

Use your configured port if it differs from 18080. Verify the actual task and settings reported by the running server. Resolve missing weights, base assets, or caches before involving the robot.

## 4.2 Workstation, terminal T: SSH tunnel

```bash
bash deploy/tunnel.sh
```

The command normally remains in the foreground without output. In another workstation terminal, query the same `/health` URL and confirm the expected server task. If the port is occupied, identify the existing tunnel or server; do not kill unrelated processes.

## 4.3 Workstation, terminal C: SONIC

Stop any old bridge and PICO manager. With the robot supported and emergency stop available:

```bash
bash collection/bin/check-network
bash collection/bin/check-camera
bash deploy/sonic.sh
```

Wait for `Init Done`. This step may cause a posture transition and must be operated on site.

## 4.4 Workstation, terminal B: check and execute

```bash
bash deploy/deploy.sh check "$TASK" --max-steps 150
bash deploy/deploy.sh dry-run "$TASK" --max-steps 150
```

`check` reads real camera/state and requests inference without publishing actions. `dry-run` uses synthetic inputs for software checks. Neither proves task success on hardware.

After on-site confirmation:

```bash
bash deploy/deploy.sh run "$TASK"
```

Initialization uses the task's recorded pose and waits for measured posture/history readiness before starting closed-loop inference. If initialization is rejected, inspect data alignment, calibration, and robot placement; do not bypass limits just to start the task.

## 4.5 Inference configuration

The following fields belong inside the JSON `server` object:

```json
{"diffusion_steps":10,"execution_frames":15,"rtc":true,
 "rtc_overlap_frames":8,"rtc_frozen_frames":2,"rtc_ramp_power":1.0}
```

The common example predicts 50 model frames and executes 15. Model rate is 30 Hz; reference streaming is 50 Hz. A 15-frame prefix spans 0.5 seconds and becomes 25 control references. Task-specific pose and hand-filter settings retain their recorded mechanisms.

Temporary overrides:

```bash
bash deploy/deploy.sh server "$TASK" --gpu 0 \
  --execution-frames 15 --diffusion-steps 10
```

Restart the server and check `/health` after configuration changes. Newly trained instructions require the matching training-generated text cache.

## 4.6 Stop, HOLD, and switch tasks

Ctrl+C in the workstation `run` terminal enters HOLD: the bridge continues streaming the last reference pose and hand targets. A terminal that remains active is expected. Keep it and SONIC running while the on-site operator secures the robot. After safely stopping SONIC, type `exit` followed by Enter as prompted. Another Ctrl+C keeps holding; HOLD is not a mechanical brake or power-loss safeguard.

To switch tasks, finish this shutdown, stop the old model server with Ctrl+C in its original terminal, and start the new task server. A tunnel using the same port can remain open. Verify `/health`, then repeat sections 4.3–4.4. Never leave an old bridge attached to a different task's server.

Logs are written to `deploy/logs/TIMESTAMP-TASK-MODE-PID/`; SONIC logs also live under `deploy/logs`. For `busy`, inspect queue/request timing. For no motion, inspect initialization, blocked actions, and history readiness. For hand oscillation, compare raw and filtered hand values. Keep the matching logs, checkpoint identity, and health response for diagnosis.

## Transport naming

Owned configuration uses `state_kp_port` / `action_kp_port`, and CLI uses `--state-kp-host/port` / `--action-kp-host/port`. Legacy arguments remain compatible. The underlying ZeroMQ library and SONIC protocol are unchanged.
