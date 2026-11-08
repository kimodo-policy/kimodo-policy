<div align="center">
  <h1>Code Structure</h1>
  <p>Module overview for the isaaclab_twist2_g1 project</p>
  <p><i>Taowen Wang</i></p>
</div>

---

## Contents

- [Overall Data Flow](#overall-data-flow)
- [Key Files](#key-files)
- [IsaacLab Simulation](#isaaclab-simulation)
- [PICO Data Reading and Conversion](#pico-data-reading-and-conversion)
- [Image Streaming](#image-streaming)
- [Action Provider](#action-provider)
- [Data Collection](#data-collection)

---

## Overall Data Flow

```
PICO headset
  │  Pose data (SMPLX)
  ▼
xrobot_teleop_to_robot_w_hand.py
  │  Motion retargeting (GMR) → 35D mimic_obs
  │  Publish to Redis
  ▼
action_provider_wh_twist2.py
  │  Read Redis → Build a 1402D observation
  │  ONNX inference (TWIST2 policy) → 29D action
  ▼
sim_main.py (IsaacLab)
  │  Step the simulation to control the G1 robot
  ▼
image_server.py
  │  H.264 streaming (ffmpeg over TCP)
  ▼
PICO headset (first-person video feedback)
```

---

## Key Files

| Module | File path |
| -------------- | --------------------------------------------------------------------------------------------- |
| Simulation entry point | `sim_main.py` |
| Simulation launcher | `run.sh` |
| Teleoperation launcher | `TWIST2/teleop.sh` |
| Data collection launcher | `TWIST2/data_record.sh` |
| PICO data conversion | `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py` |
| Action Provider | `action_provider/action_provider_wh_twist2.py`                                               |
| Provider factory | `action_provider/create_action_provider.py` |
| Image streaming | `image_server/image_server.py` |
| Data collection program | `TWIST2/deploy_real/server_data_record.py` |
| Task environment configuration | `tasks/g1_tasks/move_cylinder_g1_29dof_dex3_wholebody/move_cylinder_g1_29dof_dex3_hw_env_cfg.py` |

---

## IsaacLab Simulation

**Entry point:** `sim_main.py`. Select a task with `--task`, for example:

```bash
python sim_main.py \
  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \
  --robot_type g129 \
  --device cuda
```

**Physics parameters:**

| Parameter | Value |
| ---- | -- |
| Physics engine | PhysX |
| Simulation timestep dt | 0.001s (1ms) |
| Policy frequency | 100Hz (decimation=10) |

**Changing the simulation environment:**

Edit the task configuration `move_cylinder_g1_29dof_dex3_hw_env_cfg.py` to change:

- Initial robot position: `init_pos=(-3.9, -2.81811, 0.8)`
- Physics parameters: `self.sim.dt`, `self.decimation`, and friction coefficients
- Camera configuration: `front_camera`, `world_camera`, `left_wrist_camera`, etc.

---

## PICO Data Reading and Conversion

**Main file:** `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py`

### Reading Data

Use `XRobotStreamer.get_current_frame()` to retrieve live frames:

```python
smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = \
    self.teleop_data_streamer.get_current_frame()
```

### Motion Retargeting (GMR)

Use `GeneralMotionRetargeting` to convert human SMPLX data into robot joint angles:

```python
GMR(src_human="xrobot", tgt_robot="unitree_g1", actual_human_height=1.79)
qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
```

### Building the 35D mimic_obs

```
[xy_vel(2), z_pos(1), roll_pitch(2), yaw_vel(1), joints(29)] = 35D
```

### Publishing to Redis

| Redis Key | Dimensions | Contents |
| -------------------------------------- | ---- | -------------- |
| `action_body_unitree_g1_with_hands`    | 35   | mimic_obs      |
| `action_hand_left_unitree_g1_with_hands` | 7 | Left-hand joint angles |
| `action_hand_right_unitree_g1_with_hands` | 7 | Right-hand joint angles |
| `action_neck_unitree_g1_with_hands` | 2 | Neck yaw/pitch |
| `t_action` | - | Timestamp |

### Hand Control

Controller buttons interpolate hand opening and closing: `index_trig` closes the hand; `grip` opens it.

---

## Image Streaming

**Main file:** `image_server/image_server.py`

Four transport protocols are supported:

| Protocol | Class | Description |
| ---- | ---- | ---- |
| ZMQ (default) | `_ZmqImagePublisher` | Port 5555 |
| Redis | `_RedisImagePublisher` | Stores JPEG images and metadata |
| DDS | `_DdsImagePublisher` | Unitree SDK2 |
| XRobot (H.264) | `_XRobotImagePublisher` | ffmpeg over TCP, streamed to PICO |

**XRobot streaming parameters (configured in run.sh):**

```bash
--image_transport xrobot
--image_xrobot_host 10.42.0.35   # PICO headset IP
--image_xrobot_port 12345
--image_xrobot_width 640
--image_xrobot_height 480
--image_xrobot_bitrate 4194304   # 4Mbps
--image_fps 30
```

Images are read from shared memory (`MultiImageReader`), encoded as JPEG, and then compressed to H.264 with ffmpeg for transmission.

---

## Action Provider

**Main file:** `action_provider/action_provider_wh_twist2.py`

**Class:** `DDSRLActionProvider`

### Observation Construction (1402D)

```
action_mimic   (35D)  ← Read from Redis
obs_proprio    (92D)  ← Read from Isaac
  ├─ ang_vel   (3)  × 0.25
  ├─ roll_pitch (2)
  ├─ dof_pos_delta (29)
  ├─ dof_vel   (29) × 0.05
  └─ last_action (29)
obs_hist      (1270D) ← Previous 10 frames (127×10)
future_obs     (35D)  ← Current mimic observation
─────────────────────
Total: 1402D
```

### Policy Inference

```python
obs = self.compute_observations()  # [1, 1402]
action = self.policy(obs)          # ONNX inference; output shape [1, 29]
```

### Action Conversion

1. The policy outputs 29 values in MuJoCo actuator order.
2. Map them to Isaac joint indices using `self.twist2_action_indices`.
3. Apply `action_scale=0.5`: `target = raw_action × 0.5 + default_pos`.
4. Populate the full action vector, including hands and neck.

### Simulation Step

```python
for _ in range(self._twist2_decimation):  # decimation=10
    self.env.scene["robot"].set_joint_position_target(full_action)
    self.env.sim.step(render=False)
```

---

## Data Collection

**Main file:** `TWIST2/deploy_real/server_data_record.py`
**Launcher:** `TWIST2/data_record.sh`

### Recorded Data

**Visual data:**
- Receive images from the robot on ZMQ port 5555
- Stereo images with resolution 640×2 × 360 × 3

**State data (Redis):**

| Redis Key | Dimensions | Contents |
| -------------------------------------- | ---- | ---- |
| `state_body_unitree_g1_with_hands`     | 34   | ang_vel(3) + roll_pitch(2) + dof_pos(29) |
| `state_hand_left_unitree_g1_with_hands` | 7 | Left-hand joint angles |
| `state_hand_right_unitree_g1_with_hands` | 7 | Right-hand joint angles |
| `state_neck_unitree_g1_with_hands` | 2 | Neck angles |

**Action data (Redis):** Uses the action keys published by teleoperation, as listed above.

### Storage Format

`EpisodeWriter` stores each episode separately, including RGB images, states, actions, and timestamps, at a recording frequency of 30Hz.

### Recording Controls

| Controller button | Function |
| -------- | ---- |
| Left `key_two` | Start/stop recording |
| Left `axis_click` | Exit |