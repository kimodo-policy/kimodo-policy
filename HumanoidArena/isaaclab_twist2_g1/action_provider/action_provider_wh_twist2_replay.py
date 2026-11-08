# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Replay action provider for TWIST2 recorded data.
Based on action_provider_wh_twist2.py, replacing Redis data source with npz file.
"""

from action_provider.action_base import ActionProvider
from typing import Optional
import torch
import os
import numpy as np
import onnxruntime as ort


class ReplayActionProvider(ActionProvider):
    """Action provider for replaying recorded TWIST2 data"""

    def __init__(self, env, args_cli):
        super().__init__("ReplayActionProvider")

        # Set random seed for reproducibility
        if hasattr(args_cli, 'seed') and args_cli.seed is not None:
            import random
            import numpy as np
            seed = args_cli.seed
            print(f"[{self.name}] Setting random seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            # Enable deterministic mode for PyTorch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Store seed for ONNX Runtime configuration
            self.onnx_seed = seed
        else:
            self.onnx_seed = None
            print(f"[{self.name}] No seed specified, using non-deterministic mode")

        self.env = env
        self.replay_file = args_cli.replay_file
        self.replay_mode = args_cli.replay_mode  # "inference" or "direct"
        self.replay_loop = args_cli.replay_loop
        self.enable_dex3_replay = False
        self._initial_state_compared = False

        # Validate replay file
        if not os.path.exists(self.replay_file):
            raise FileNotFoundError(f"[{self.name}] Replay file not found: {self.replay_file}")

        # Load replay data
        print(f"[{self.name}] Loading replay data from: {self.replay_file}")
        self._load_replay_data()

        # Initialize replay state
        # IMPORTANT: Frame 0 is the initial state, Frame 0 action transitions to Frame 1
        # current_frame = 0: First get_action() will apply Frame 0 action (ONNX output)
        # current_frame = 1: Second get_action() will apply Frame 1 action, and so on
        self.current_frame = 0  # Start at 0 to apply Frame 0 action first
        self.total_frames = len(self.replay_data_qpos)
        print(f"[{self.name}] Loaded {self.total_frames} frames")
        print(f"[{self.name}] Frame 0 is initial state, Frame 0 action will transition to Frame 1")
        print(f"[{self.name}] Replay will apply actions starting from Frame 0")
        print(f"[{self.name}] Replay mode: {self.replay_mode}")
        print(f"[{self.name}] Loop: {self.replay_loop}")

        # Setup joint mapping (same as original)
        self._setup_joint_mapping()

        # Initialize history buffers for inference mode (CRITICAL for correct observation!)
        # These are needed to maintain observation history like the recording script
        self.n_obs_single = 127  # Size of single observation (35 mimic + 92 proprio)
        self.history_len = 10
        self._twist2_history = torch.zeros(
            self.history_len, self.n_obs_single,
            device=self.env.device, dtype=torch.float32
        )
        self._twist2_last_action = torch.zeros(
            1, 29, device=self.env.device, dtype=torch.float32
        )
        print(f"[{self.name}] History buffers initialized: history={self._twist2_history.shape}, last_action={self._twist2_last_action.shape}")

        # Initialize ONNX model if in inference mode
        if self.replay_mode == "inference":
            self.policy_path = args_cli.model_path
            if not os.path.exists(self.policy_path):
                raise FileNotFoundError(f"[{self.name}] Policy file not found: {self.policy_path}")
            print(f"[{self.name}] Loading ONNX model: {self.policy_path}")
            self._load_onnx_model()
        else:
            self.policy = None

        # Decimation and rendering settings (same as original)
        self._twist2_decimation = getattr(env.cfg, 'decimation', 10)
        self._render_interval = 1  # Render every N policy steps
        self._render_counter = 0
        self._obs_interval = 1  # Compute observations every N policy steps
        self._obs_counter = 0

        # Action buffer
        self._full_action_buf = torch.zeros(
            self.env.scene["robot"].data.default_joint_pos.shape[1],
            device=self.env.device,
            dtype=torch.float32
        )

        # Flag to track if initial state has been set
        self._initial_state_set = False

        # Print detailed configuration for verification
        print(f"[{self.name}] Replay action provider initialized successfully")
        print(f"[{self.name}] ===== Configuration Verification =====")
        print(f"[{self.name}]   Decimation: {self._twist2_decimation}")
        print(f"[{self.name}]   Physics dt: {self.env.physics_dt}")
        print(f"[{self.name}]   Control dt: {self.env.physics_dt * self._twist2_decimation}")
        print(f"[{self.name}]   Device: {self.env.device}")
        print(f"[{self.name}]   Seed (env): {self.env.cfg.seed if hasattr(self.env.cfg, 'seed') else 'Not set'}")
        print(f"[{self.name}]   Seed (action provider): {self.onnx_seed if hasattr(self, 'onnx_seed') else 'Not set'}")
        print(f"[{self.name}]   PyTorch deterministic: {torch.backends.cudnn.deterministic}")
        print(f"[{self.name}]   PyTorch benchmark: {torch.backends.cudnn.benchmark}")

        # Print PD controller parameters
        try:
            robot = self.env.scene["robot"]
            if hasattr(robot, 'actuators') and 'twist2' in robot.actuators:
                actuator = robot.actuators['twist2']
                print(f"[{self.name}]   PD Stiffness (first 5): {actuator.stiffness[:5] if hasattr(actuator, 'stiffness') else 'N/A'}")
                print(f"[{self.name}]   PD Damping (first 5): {actuator.damping[:5] if hasattr(actuator, 'damping') else 'N/A'}")
                print(f"[{self.name}]   Effort limit (first 5): {actuator.effort_limit[:5] if hasattr(actuator, 'effort_limit') else 'N/A'}")
        except Exception as e:
            print(f"[{self.name}]   Could not read PD parameters: {e}")

        print(f"[{self.name}] =====================================")

    def _load_replay_data(self):
        """Load replay data from npz file"""
        data = np.load(self.replay_file, allow_pickle=True)

        # Load observation data (for inference mode)
        if 'robot_obs_buf' in data:
            self.replay_data_obs = data['robot_obs_buf']  # [N, 1432]
            print(f"[{self.name}] Loaded observation data: {self.replay_data_obs.shape}")
        else:
            self.replay_data_obs = None
            print(f"[{self.name}] Warning: No observation data found in replay file")

        if 'human_hand_left' in data and 'human_hand_right' in data:
            self.replay_data_hand_left = data['human_hand_left']  # [N, 7]
            self.replay_data_hand_right = data['human_hand_right']  # [N, 7]
            print(f"[{self.name}] Loaded left hand data: {self.replay_data_hand_left.shape}")
            print(f"[{self.name}] Loaded right hand data: {self.replay_data_hand_right.shape}")
        else:
            self.replay_data_hand_left = None
            self.replay_data_hand_right = None
            print(f"[{self.name}] No recorded hand data found in replay file")

        # Load qpos data (for direct mode)
        # Use twist2_inference_qpos which is the ONNX output (target positions for PD controller)
        # This is: target_29 = raw_action * 0.5 + default_pos (after clip)
        if 'robot_twist2_inference_qpos' in data:
            self.replay_data_qpos = data['robot_twist2_inference_qpos']  # [N, 29]
            print(f"[{self.name}] Loaded qpos data (ONNX inference output): {self.replay_data_qpos.shape}")
        else:
            raise ValueError(f"[{self.name}] No qpos data found in replay file (expected 'robot_twist2_inference_qpos')")

        # Also load actual positions for comparison/debugging
        if 'robot_qpos_before_decimation' in data:
            self.replay_data_qpos_actual = data['robot_qpos_before_decimation']  # [N, 29]
            print(f"[{self.name}] Loaded actual qpos for debugging: {self.replay_data_qpos_actual.shape}")
        else:
            self.replay_data_qpos_actual = None

        # Load root state data (CRITICAL for correct replay!)
        if 'robot_root_position' in data and 'robot_root_orientation' in data:
            self.replay_data_root_pos = data['robot_root_position']  # [N, 3]
            self.replay_data_root_quat = data['robot_root_orientation']  # [N, 4] (w,x,y,z)
            print(f"[{self.name}] Loaded root position: {self.replay_data_root_pos.shape}")
            print(f"[{self.name}] Loaded root orientation: {self.replay_data_root_quat.shape}")
            print(f"[{self.name}] Initial root position: {self.replay_data_root_pos[0]}")
            print(f"[{self.name}] Initial root orientation: {self.replay_data_root_quat[0]}")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root state data found in replay file!")
            print(f"[{self.name}] Robot will use default spawn position/orientation")
            self.replay_data_root_pos = None
            self.replay_data_root_quat = None

        # Load velocity data (CRITICAL for physics simulation!)
        # Prefer world frame velocities if available (from new recording format)
        if 'robot_root_lin_vel_world' in data:
            self.replay_data_root_lin_vel = data['robot_root_lin_vel_world']  # [N, 3] world frame
            print(f"[{self.name}] Loaded root linear velocity (world frame): {self.replay_data_root_lin_vel.shape}")
        elif 'robot_root_lin_vel_local' in data:
            self.replay_data_root_lin_vel = data['robot_root_lin_vel_local']  # [N, 3] local frame (legacy)
            print(f"[{self.name}] Loaded root linear velocity (local frame - legacy): {self.replay_data_root_lin_vel.shape}")
            print(f"[{self.name}] ⚠️  WARNING: Using local frame velocity, may need coordinate conversion")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root linear velocity data found!")
            self.replay_data_root_lin_vel = None

        if 'robot_root_ang_vel_world' in data:
            self.replay_data_root_ang_vel = data['robot_root_ang_vel_world']  # [N, 3] world frame
            print(f"[{self.name}] Loaded root angular velocity (world frame): {self.replay_data_root_ang_vel.shape}")
        elif 'robot_root_ang_vel_local' in data:
            self.replay_data_root_ang_vel = data['robot_root_ang_vel_local']  # [N, 3] local frame (legacy)
            print(f"[{self.name}] Loaded root angular velocity (local frame - legacy): {self.replay_data_root_ang_vel.shape}")
            print(f"[{self.name}] ⚠️  WARNING: Using local frame velocity, may need coordinate conversion")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root angular velocity data found!")
            self.replay_data_root_ang_vel = None

        if 'robot_qvel_before_decimation' in data:
            self.replay_data_qvel = data['robot_qvel_before_decimation']  # [N, 29]
            print(f"[{self.name}] Loaded joint velocities: {self.replay_data_qvel.shape}")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No joint velocity data found!")
            self.replay_data_qvel = None

        if 'env_obj_football_position' in data:
            self.replay_data_football_pos = data['env_obj_football_position']
            self.replay_data_football_lin_vel = data.get('env_obj_football_linear_velocity')
            self.replay_data_football_ang_vel = data.get('env_obj_football_angular_velocity')
            print(f"[{self.name}] Loaded football state: {self.replay_data_football_pos.shape}")
        else:
            self.replay_data_football_pos = None
            self.replay_data_football_lin_vel = None
            self.replay_data_football_ang_vel = None

        # Load physics state data (for analysis, not for writing back)
        if 'robot_applied_torque_before_decimation' in data:
            self.replay_data_applied_torque = data['robot_applied_torque_before_decimation']  # [N, 29]
            print(f"[{self.name}] Loaded applied torques (for analysis): {self.replay_data_applied_torque.shape}")
            self._enable_torque_analysis = True
        else:
            print(f"[{self.name}] ℹ️  No applied torque data (old recording format)")
            self.replay_data_applied_torque = None
            self._enable_torque_analysis = False

        if 'robot_body_net_contact_forces' in data:
            self.replay_data_contact_forces = data['robot_body_net_contact_forces']  # [N, num_bodies, 3]
            print(f"[{self.name}] Loaded contact forces (for analysis): {self.replay_data_contact_forces.shape}")
            self._enable_contact_analysis = True
        else:
            print(f"[{self.name}] ℹ️  No contact force data (old recording format)")
            self.replay_data_contact_forces = None
            self._enable_contact_analysis = False

        # Initialize analysis counters
        self._torque_diff_sum = 0.0
        self._torque_diff_count = 0
        self._contact_diff_sum = 0.0
        self._contact_diff_count = 0

        print(f"[{self.name}] Replay data loaded successfully")

    def _load_onnx_model(self):
        """Load ONNX model for inference mode"""
        try:
            # Create ONNX runtime session
            sess_options = ort.SessionOptions()

            # Configure for deterministic inference if seed is set
            if self.onnx_seed is not None:
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                print(f"[{self.name}] ONNX Runtime configured for deterministic inference (seed={self.onnx_seed})")

            # Match TWIST2 recording path: prefer CUDA whenever onnxruntime exposes it.
            available_providers = ort.get_available_providers()
            print(f"[{self.name}] Available ONNX providers: {available_providers}")

            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print(f"[{self.name}] Using CUDAExecutionProvider")
            else:
                providers = ["CPUExecutionProvider"]
                print(f"[{self.name}] Using CPUExecutionProvider")

            self.policy = ort.InferenceSession(
                self.policy_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.policy.get_inputs()[0].name
            self.output_name = self.policy.get_outputs()[0].name

            print(f"[{self.name}] ONNX model loaded successfully")
            print(f"[{self.name}] Input: {self.input_name}, Output: {self.output_name}")
            print(f"[{self.name}] Active provider: {self.policy.get_providers()}")

        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to load ONNX model: {e}")

    def _setup_joint_mapping(self):
        """Setup joint mapping for TWIST2 (29 DOF) - same as original"""
        # TWIST2 29-dof action order (MuJoCo actuator order) - copied from recording script
        self.twist2_action_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # Get all joint names and create mapping (same as recording script)
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        # Dynamically map joint names to Isaac Lab indices (CRITICAL FIX!)
        missing = [n for n in self.twist2_action_joint_names if n not in self.joint_to_index]
        if missing:
            raise ValueError(f"TWIST2 joints missing in Isaac asset: {missing}")
        self.twist2_action_indices = [self.joint_to_index[n] for n in self.twist2_action_joint_names]

        # Get default joint positions (keep [1, 29] shape like recording script)
        self.twist2_default_pos = self.env.scene["robot"].data.default_joint_pos[:, self.twist2_action_indices]

        self.left_hand_joint_mapping = {
            "left_hand_thumb_0_joint": 0,
            "left_hand_thumb_1_joint": 1,
            "left_hand_thumb_2_joint": 2,
            "left_hand_middle_0_joint": 3,
            "left_hand_middle_1_joint": 4,
            "left_hand_index_0_joint": 5,
            "left_hand_index_1_joint": 6,
        }
        self.right_hand_joint_mapping = {
            "right_hand_thumb_0_joint": 0,
            "right_hand_thumb_1_joint": 1,
            "right_hand_thumb_2_joint": 2,
            "right_hand_middle_0_joint": 3,
            "right_hand_middle_1_joint": 4,
            "right_hand_index_0_joint": 5,
            "right_hand_index_1_joint": 6,
        }

        left_hand_missing = [n for n in self.left_hand_joint_mapping if n not in self.joint_to_index]
        right_hand_missing = [n for n in self.right_hand_joint_mapping if n not in self.joint_to_index]
        if (
            self.replay_data_hand_left is not None
            and self.replay_data_hand_right is not None
            and not left_hand_missing
            and not right_hand_missing
        ):
            self._left_hand_target_indices = [
                self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()
            ]
            self._left_hand_source_indices = [idx for idx in self.left_hand_joint_mapping.values()]
            self._right_hand_target_indices = [
                self.joint_to_index[name] for name in self.right_hand_joint_mapping.keys()
            ]
            self._right_hand_source_indices = [idx for idx in self.right_hand_joint_mapping.values()]
            self._left_hand_target_idx_t = torch.tensor(
                self._left_hand_target_indices, dtype=torch.long, device=self.env.device
            )
            self._left_hand_source_idx_t = torch.tensor(
                self._left_hand_source_indices, dtype=torch.long, device=self.env.device
            )
            self._right_hand_target_idx_t = torch.tensor(
                self._right_hand_target_indices, dtype=torch.long, device=self.env.device
            )
            self._right_hand_source_idx_t = torch.tensor(
                self._right_hand_source_indices, dtype=torch.long, device=self.env.device
            )
            self._left_hand_buf = torch.empty(
                len(self._left_hand_source_indices), device=self.env.device, dtype=torch.float32
            )
            self._right_hand_buf = torch.empty(
                len(self._right_hand_source_indices), device=self.env.device, dtype=torch.float32
            )
            self.enable_dex3_replay = True
            print(f"[{self.name}] Dex3 hand replay enabled")
        elif self.replay_data_hand_left is not None or self.replay_data_hand_right is not None:
            print(
                f"[{self.name}] Dex3 hand replay disabled: "
                f"missing left joints={left_hand_missing}, missing right joints={right_hand_missing}"
            )

        print(f"[{self.name}] Joint mapping setup complete (29 DOF)")
        print(f"[{self.name}] Joint order: {self.twist2_action_joint_names[:5]}... (showing first 5)")
        print(f"[{self.name}] Isaac indices: {self.twist2_action_indices[:5]}... (showing first 5)")
        print(f"[{self.name}] Default pos shape: {self.twist2_default_pos.shape}")

    def _apply_recorded_hand_action(self, full_action: torch.Tensor, frame_idx: int) -> None:
        if not self.enable_dex3_replay:
            return

        self._left_hand_buf.copy_(
            torch.from_numpy(self.replay_data_hand_left[frame_idx]).to(self.env.device, dtype=torch.float32)
        )
        self._right_hand_buf.copy_(
            torch.from_numpy(self.replay_data_hand_right[frame_idx]).to(self.env.device, dtype=torch.float32)
        )
        left_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
        right_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
        full_action.index_copy_(0, self._left_hand_target_idx_t, left_vals)
        full_action.index_copy_(0, self._right_hand_target_idx_t, right_vals)

    def _log_initial_state_comparison(self) -> None:
        if self._initial_state_compared:
            return
        self._initial_state_compared = True

        print(f"\n[{self.name}] ========== INITIAL STATE COMPARISON (Frame 0, pre-action) ==========")
        try:
            robot = self.env.scene["robot"]
            root_state = robot.data.root_state_w[0]
            joint_pos = robot.data.joint_pos[0, self.twist2_action_indices].cpu().numpy()
            joint_vel = robot.data.joint_vel[0, self.twist2_action_indices].cpu().numpy()

            root_pos = root_state[0:3].cpu().numpy()
            root_quat = root_state[3:7].cpu().numpy()
            root_lin_vel = root_state[7:10].cpu().numpy()
            root_ang_vel = root_state[10:13].cpu().numpy()

            expected_joint_pos = self.replay_data_qpos_actual[0] if self.replay_data_qpos_actual is not None else self.replay_data_qpos[0]
            asset_msg = f"[{self.name}] Asset state: joint_pos_err={np.linalg.norm(joint_pos - expected_joint_pos):.6f} rad"
            if self.replay_data_root_pos is not None:
                asset_msg += f", root_pos_err={np.linalg.norm(root_pos - self.replay_data_root_pos[0]):.6f} m"
            if self.replay_data_root_quat is not None:
                asset_msg += f", root_quat_err={np.linalg.norm(root_quat - self.replay_data_root_quat[0]):.6f}"
            print(asset_msg)
            if (
                self.replay_data_qvel is not None
                and self.replay_data_root_lin_vel is not None
                and self.replay_data_root_ang_vel is not None
            ):
                print(
                    f"[{self.name}] PhysX state:"
                    f" joint_vel_err={np.linalg.norm(joint_vel - self.replay_data_qvel[0]):.6f} rad/s,"
                    f" root_lin_vel_err={np.linalg.norm(root_lin_vel - self.replay_data_root_lin_vel[0]):.6f} m/s,"
                    f" root_ang_vel_err={np.linalg.norm(root_ang_vel - self.replay_data_root_ang_vel[0]):.6f} rad/s"
                )

            if (
                self.replay_data_football_pos is not None
                and self.replay_data_football_lin_vel is not None
                and self.replay_data_football_ang_vel is not None
                and "object" in self.env.scene.keys()
            ):
                football = self.env.scene["object"]
                football_state = football.data.root_state_w[0]
                football_pos = football_state[0:3].cpu().numpy()
                football_lin_vel = football_state[7:10].cpu().numpy()
                football_ang_vel = football_state[10:13].cpu().numpy()
                print(
                    f"[{self.name}] Football asset state:"
                    f" pos_err={np.linalg.norm(football_pos - self.replay_data_football_pos[0]):.6f} m,"
                    f" lin_vel_err={np.linalg.norm(football_lin_vel - self.replay_data_football_lin_vel[0]):.6f} m/s,"
                    f" ang_vel_err={np.linalg.norm(football_ang_vel - self.replay_data_football_ang_vel[0]):.6f} rad/s"
                )

            if self._enable_torque_analysis and self.replay_data_applied_torque is not None:
                replay_torque = robot.data.applied_torque[0, self.twist2_action_indices].cpu().numpy()
                print(
                    f"[{self.name}] PhysX actuator state:"
                    f" applied_torque_err={np.linalg.norm(replay_torque - self.replay_data_applied_torque[0]):.6f} Nm"
                )

            if self._enable_contact_analysis and self.replay_data_contact_forces is not None:
                replay_contact = robot.data.body_net_contact_force_w[0].cpu().numpy()
                print(
                    f"[{self.name}] PhysX contact state:"
                    f" body_contact_err={np.linalg.norm(replay_contact - self.replay_data_contact_forces[0]):.6f} N"
                )
        except Exception as exc:
            print(f"[{self.name}] Initial state comparison failed: {exc}")
        print(f"[{self.name}] ===============================================================\n")

    # def set_initial_state_before_simulation(self, force=False):
    #     """Set initial state from Frame 0 BEFORE simulation starts
    #
    #     CRITICAL: This method must be called from sim_main_replay.py AFTER creating
    #     the action provider but BEFORE starting the control loop. After calling this,
    #     env.reset() should be called again to properly initialize the physics engine.
    #
    #     Uses Frame 0 data for initialization. Replay will start from Frame 1.
    #     This ensures state-action alignment: Frame 0 state + Frame 1 action.
    #
    #     Args:
    #         force: If True, force re-setting even if already set (useful after env.reset())
    #     """
    #     if self._initial_state_set and not force:
    #         print(f"[{self.name}] Initial state already set, skipping")
    #         return
    #
    #     print(f"[{self.name}] 🔧 Setting initial state from Frame 0 (replay starts from Frame 1)...")
    #
    #     # DEBUG: Read state BEFORE setting
    #     robot = self.env.scene["robot"]
    #     print(f"\n[{self.name}]   📊 State BEFORE setting:")
    #     print(f"[{self.name}]   Joint vel (first 5): {robot.data.joint_vel[0, self.twist2_action_indices[:5]].cpu().numpy()}")
    #     print(f"[{self.name}]   Root lin_vel: {robot.data.root_state_w[0, 7:10].cpu().numpy()}")
    #
    #     # 1. Set robot root state (position, orientation, velocities)
    #     if self.replay_data_root_pos is not None and self.replay_data_root_quat is not None:
    #         robot = self.env.scene["robot"]
    #         root_state = robot.data.default_root_state.clone()
    #
    #         # Set position and orientation
    #         root_state[0, 0:3] = torch.from_numpy(self.replay_data_root_pos[0]).to(self.env.device, dtype=torch.float32)
    #         root_state[0, 3:7] = torch.from_numpy(self.replay_data_root_quat[0]).to(self.env.device, dtype=torch.float32)
    #
    #         # Set velocities (prefer world frame)
    #         if self.replay_data_root_lin_vel is not None:
    #             root_state[0, 7:10] = torch.from_numpy(self.replay_data_root_lin_vel[0]).to(self.env.device, dtype=torch.float32)
    #             print(f"[{self.name}]   ✓ Set root lin_vel: {self.replay_data_root_lin_vel[0]}")
    #         else:
    #             root_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)
    #             print(f"[{self.name}]   ⚠️  No root lin_vel data, using zeros")
    #
    #         if self.replay_data_root_ang_vel is not None:
    #             root_state[0, 10:13] = torch.from_numpy(self.replay_data_root_ang_vel[0]).to(self.env.device, dtype=torch.float32)
    #             print(f"[{self.name}]   ✓ Set root ang_vel: {self.replay_data_root_ang_vel[0]}")
    #         else:
    #             root_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)
    #             print(f"[{self.name}]   ⚠️  No root ang_vel data, using zeros")
    #
    #         robot.write_root_state_to_sim(root_state)
    #         print(f"[{self.name}]   ✓ Set root pos: {self.replay_data_root_pos[0]}")
    #         print(f"[{self.name}]   ✓ Set root quat: {self.replay_data_root_quat[0]}")
    #     else:
    #         print(f"[{self.name}]   ⚠️  WARNING: No root state data, using spawn default")
    #
    #     # 2. Set joint positions and velocities
    #     # CRITICAL: For initialization, use ACTUAL positions (qpos_actual), not target positions (qpos)
    #     # qpos = PD controller target (ONNX output)
    #     # qpos_actual = actual joint positions after physics simulation
    #     # For initial state, we need the actual state, not the target!
    #     if self.replay_data_qpos_actual is not None:
    #         initial_qpos = self.replay_data_qpos_actual[0]  # Use ACTUAL positions for initialization
    #         initial_qpos_tensor = torch.from_numpy(initial_qpos).to(self.env.device, dtype=torch.float32).unsqueeze(0)
    #
    #         # Create full joint position array
    #         full_initial_pos = self.env.scene["robot"].data.default_joint_pos.clone()
    #         full_initial_pos[0, self.twist2_action_indices] = initial_qpos_tensor
    #
    #         # Get initial joint velocities
    #         if self.replay_data_qvel is not None:
    #             initial_qvel = self.replay_data_qvel[0]
    #             initial_qvel_tensor = torch.from_numpy(initial_qvel).to(self.env.device, dtype=torch.float32).unsqueeze(0)
    #             full_initial_vel = torch.zeros_like(full_initial_pos)
    #             full_initial_vel[0, self.twist2_action_indices] = initial_qvel_tensor
    #             print(f"[{self.name}]   ✓ Set joint vel (first five): {initial_qvel[:5]}")
    #         else:
    #             full_initial_vel = torch.zeros_like(full_initial_pos)
    #             print(f"[{self.name}]   ⚠️  No joint vel data, using zeros")
    #
    #         # Set joint state
    #         self.env.scene["robot"].write_joint_state_to_sim(
    #             position=full_initial_pos,
    #             velocity=full_initial_vel
    #         )
    #         print(f"[{self.name}]   ✓ Set joint pos (actual, first five): {initial_qpos[:5]}")
    #     elif self.replay_data_qpos is not None:
    #         # Fallback to target positions if actual positions not available
    #         print(f"[{self.name}]   ⚠️  WARNING: Using target positions (qpos) for initialization - actual positions (qpos_actual) not available")
    #         initial_qpos = self.replay_data_qpos[0]
    #         initial_qpos_tensor = torch.from_numpy(initial_qpos).to(self.env.device, dtype=torch.float32).unsqueeze(0)
    #         full_initial_pos = self.env.scene["robot"].data.default_joint_pos.clone()
    #         full_initial_pos[0, self.twist2_action_indices] = initial_qpos_tensor
    #         full_initial_vel = torch.zeros_like(full_initial_pos)
    #         if self.replay_data_qvel is not None:
    #             initial_qvel = self.replay_data_qvel[0]
    #             initial_qvel_tensor = torch.from_numpy(initial_qvel).to(self.env.device, dtype=torch.float32).unsqueeze(0)
    #             full_initial_vel[0, self.twist2_action_indices] = initial_qvel_tensor
    #         self.env.scene["robot"].write_joint_state_to_sim(
    #             position=full_initial_pos,
    #             velocity=full_initial_vel
    #         )
    #         print(f"[{self.name}]   ✓ Set joint pos (target, first five): {initial_qpos[:5]}")
    #     else:
    #         print(f"[{self.name}]   ⚠️  WARNING: No joint position data available")
    #
    #     # 3. Set football state if available
    #     try:
    #         if "object" in self.env.scene.keys():
    #             # Check if football data exists in recording
    #             data = np.load(self.replay_file, allow_pickle=True)
    #             if 'env_obj_football_position' in data:
    #                 football = self.env.scene["object"]
    #                 football_state = football.data.default_root_state.clone()
    #
    #                 # Set position
    #                 football_pos = data['env_obj_football_position'][0]
    #                 football_state[0, 0:3] = torch.from_numpy(football_pos).to(self.env.device, dtype=torch.float32)
    #
    #                 # Set velocities if available
    #                 if 'env_obj_football_linear_velocity' in data:
    #                     football_lin_vel = data['env_obj_football_linear_velocity'][0]
    #                     football_state[0, 7:10] = torch.from_numpy(football_lin_vel).to(self.env.device, dtype=torch.float32)
    #                 else:
    #                     football_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)
    #
    #                 if 'env_obj_football_angular_velocity' in data:
    #                     football_ang_vel = data['env_obj_football_angular_velocity'][0]
    #                     football_state[0, 10:13] = torch.from_numpy(football_ang_vel).to(self.env.device, dtype=torch.float32)
    #                 else:
    #                     football_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)
    #
    #                 football.write_root_state_to_sim(football_state)
    #                 print(f"[{self.name}]   ✓ Set football pos: {football_pos}")
    #             else:
    #                 print(f"[{self.name}]   ⚠️  No football data in recording")
    #     except Exception as e:
    #         print(f"[{self.name}]   ⚠️  Failed to set football state: {e}")
    #
    #     # CRITICAL: Must call write_data_to_sim() to write buffer data to PhysX!
    #     # Without this, the initial state is only in Isaac Lab buffers, not in physics engine
    #     print(f"\n[{self.name}]   🔧 Writing initial state to PhysX...")
    #     self.env.scene.write_data_to_sim()
    #     print(f"[{self.name}]   ✓ Initial state written to PhysX")
    #
    #     # CRITICAL: Must read back from PhysX to verify!
    #     # Simply reading robot.data.joint_vel only reads Isaac Lab buffers, not PhysX state
    #     # We need to call scene.update() to read from PhysX
    #     print(f"\n[{self.name}]   🔧 Reading state from PhysX to verify...")
    #     # self.env.scene.update(dt=self.env.physics_dt)
    #     print(f"[{self.name}]   ✓ State read from PhysX")
    #
    #     # Now verify the state was correctly set in PhysX
    #     print(f"\n[{self.name}]   🔍 Verifying initial state (read from PhysX)...")
    #     robot = self.env.scene["robot"]
    #     actual_root_state = robot.data.root_state_w[0]
    #     actual_joint_pos = robot.data.joint_pos[0, self.twist2_action_indices]
    #     actual_joint_vel = robot.data.joint_vel[0, self.twist2_action_indices]
    #
    #     # Verify root linear velocity
    #     print(f"[{self.name}]   Expected root lin_vel: {self.replay_data_root_lin_vel[0]}")
    #     print(f"[{self.name}]   Actual root lin_vel:   {actual_root_state[7:10].cpu().numpy()}")
    #     lin_vel_error = np.linalg.norm(actual_root_state[7:10].cpu().numpy() - self.replay_data_root_lin_vel[0])
    #     print(f"[{self.name}]   Lin vel error: {lin_vel_error:.6f} m/s")
    #
    #     # Verify joint velocities
    #     print(f"[{self.name}]   Expected joint vel (first 5): {self.replay_data_qvel[0][:5]}")
    #     print(f"[{self.name}]   Actual joint vel (first 5):   {actual_joint_vel[:5].cpu().numpy()}")
    #     joint_vel_error = np.linalg.norm(actual_joint_vel.cpu().numpy() - self.replay_data_qvel[0])
    #     print(f"[{self.name}]   Joint vel error: {joint_vel_error:.6f} rad/s")
    #
    #     # Verify joint positions
    #     if self.replay_data_qpos_actual is not None:
    #         expected_pos = self.replay_data_qpos_actual[0]
    #     else:
    #         expected_pos = self.replay_data_qpos[0]
    #     print(f"[{self.name}]   Expected joint pos (first 5): {expected_pos[:5]}")
    #     print(f"[{self.name}]   Actual joint pos (first 5):   {actual_joint_pos[:5].cpu().numpy()}")
    #     joint_pos_error = np.linalg.norm(actual_joint_pos.cpu().numpy() - expected_pos)
    #     print(f"[{self.name}]   Joint pos error: {joint_pos_error:.6f} rad")
    #
    #     if lin_vel_error > 0.01 or joint_vel_error > 0.1 or joint_pos_error > 0.01:
    #         print(f"[{self.name}]   ⚠️  WARNING: Large error detected after reading from PhysX!")
    #         print(f"[{self.name}]   This indicates PhysX may not have correctly received the initial state.")
    #     else:
    #         print(f"[{self.name}]   ✅ PhysX state verification passed")
    #
    #     # DO NOT set PD controller targets here!
    #     # The first get_action() call will apply Frame 0 action to transition to Frame 1.
    #
    #     # DO NOT call env.sim.step() here!
    #     # scene.update() already read the state, stepping would change it
    #
    #     # Mark as set (but don't advance current_frame yet)
    #     self._initial_state_set = True
    #     print(f"[{self.name}]   ✅ Initial state set, current_frame = {self.current_frame}")
    #     print(f"[{self.name}]   ⚠️  First get_action() will apply Frame 0 action to transition to Frame 1")
    #
    #     # Read back the actual state after initialization and log it
    #     # Note: This now reads from PhysX, not just buffers
    #     self._log_initial_state_verification()

    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from replay data - same structure as original get_action"""
        try:
            # CRITICAL: First call (current_frame == 0) applies Frame 0 action (ONNX output)
            # This matches the recording flow and transitions state from Frame 0 to Frame 1
            if self.current_frame == 0:
                self._log_initial_state_comparison()
                print(f"\n[{self.name}] ========== FIRST get_action() CALL: Applying Frame 0 Action ==========")
                print(f"[{self.name}] Applying Frame 0 action (ONNX output) to match recording flow")
                print(f"[{self.name}] After decimation loop, state should transition to Frame 1")
                print(f"[{self.name}] ================================================\n")

                # CRITICAL: Use Frame 0 action (ONNX output), NOT Frame 0 actual positions
                # This matches the recording flow where Frame 0 action was applied
                target_29 = torch.from_numpy(self.replay_data_qpos[0]).unsqueeze(0).to(
                    self.env.device, dtype=torch.float32)  # [1, 29]

                # Fill full action buffer
                full_action = self._full_action_buf
                full_action.zero_()
                full_action.copy_(self.env.scene["robot"].data.default_joint_pos.squeeze(0))
                full_action.index_copy_(0, torch.tensor(self.twist2_action_indices, device=self.env.device,
                                                        dtype=torch.long), target_29.squeeze(0))
                self._apply_recorded_hand_action(full_action, 0)

                # Run decimation loop to stabilize Frame 0 state
                for i in range(self._twist2_decimation):
                    self.env.scene["robot"].set_joint_position_target(full_action)
                    self.env.scene.write_data_to_sim()

                    is_last_step = (i == self._twist2_decimation - 1)
                    should_render = is_last_step and (self._render_counter % self._render_interval == 0)

                    if should_render:
                        self.env.sim.step(render=True)
                    else:
                        self.env.sim.step(render=False)

                    self.env.scene.update(dt=self.env.physics_dt)

                self._render_counter += 1

                # Verify state after applying Frame 0 action
                # After decimation, state should match Frame 1 (not Frame 0!)
                print(f"\n[{self.name}] ========== AFTER APPLYING Frame 0 ACTION ==========")
                robot = self.env.scene["robot"]
                actual_root_state = robot.data.root_state_w[0]
                actual_joint_pos = robot.data.joint_pos[0, self.twist2_action_indices]
                actual_joint_vel = robot.data.joint_vel[0, self.twist2_action_indices]

                # Compare with Frame 1 state (state after Frame 0 action was applied)
                print(f"[{self.name}] Expected Frame 1 root pos: {self.replay_data_root_pos[1]}")
                print(f"[{self.name}] Actual root pos: {actual_root_state[:3].cpu().numpy()}")
                pos_error = np.linalg.norm(actual_root_state[:3].cpu().numpy() - self.replay_data_root_pos[1])
                print(f"[{self.name}] Position error: {pos_error:.6f} m")

                if self.replay_data_root_lin_vel is not None:
                    print(f"[{self.name}] Expected Frame 1 root lin_vel: {self.replay_data_root_lin_vel[1]}")
                    print(f"[{self.name}] Actual root lin_vel: {actual_root_state[7:10].cpu().numpy()}")
                    vel_error = np.linalg.norm(actual_root_state[7:10].cpu().numpy() - self.replay_data_root_lin_vel[1])
                    print(f"[{self.name}] Velocity error: {vel_error:.6f} m/s")

                print(f"[{self.name}] Expected Frame 1 joint pos (first 5): {self.replay_data_qpos_actual[1][:5]}")
                print(f"[{self.name}] Actual joint pos (first 5): {actual_joint_pos[:5].cpu().numpy()}")
                joint_pos_error = np.linalg.norm(actual_joint_pos.cpu().numpy() - self.replay_data_qpos_actual[1])
                print(f"[{self.name}] Joint position error: {joint_pos_error:.6f} rad")

                if self.replay_data_qvel is not None:
                    print(f"[{self.name}] Expected Frame 1 joint vel (first 5): {self.replay_data_qvel[1][:5]}")
                    print(f"[{self.name}] Actual joint vel (first 5): {actual_joint_vel[:5].cpu().numpy()}")
                    joint_vel_error = np.linalg.norm(actual_joint_vel.cpu().numpy() - self.replay_data_qvel[1])
                    print(f"[{self.name}] Joint velocity error: {joint_vel_error:.6f} rad/s")

                print(f"[{self.name}] ================================================\n")

                # Advance to frame 1 for actual replay
                self.current_frame = 1
                print(f"[{self.name}] Frame 0 action applied, state transitioned to Frame 1, advancing current_frame to 1")

                return full_action

            # Check if replay finished
            if self.current_frame >= self.total_frames:
                if self.replay_loop:
                    print(f"[{self.name}] Replay finished, looping back to frame 1")
                    self.current_frame = 1  # Loop back to frame 1 (frame 0 is initialization)
                    # Reset initial state flag for loop
                    self._initial_state_set = False
                else:
                    print(f"[{self.name}] Replay finished")
                    return None

            # Monitor root movement and detect potential falls
            if self.current_frame > 0 and self.current_frame % 10 == 0:
                current_root_state = self.env.scene["robot"].data.root_state_w[0]
                current_root_pos = current_root_state[:3].cpu().numpy()
                current_root_quat = current_root_state[3:7].cpu().numpy()
                current_root_lin_vel = current_root_state[7:10].cpu().numpy()
                current_root_ang_vel = current_root_state[10:13].cpu().numpy()

                print(f"[{self.name}] Frame {self.current_frame}: Root pos = {current_root_pos}")

                # Check for potential fall (z position too low or extreme tilt)
                if current_root_pos[2] < 0.3:  # Robot base below 30cm
                    print(f"[{self.name}]   ⚠️  WARNING: Low z position {current_root_pos[2]:.4f}m - possible fall!")

                # Check orientation (w component should be close to 1 for upright)
                if abs(current_root_quat[0]) < 0.7:  # w < 0.7 means >45° tilt
                    print(f"[{self.name}]   ⚠️  WARNING: Large tilt detected! quat = {current_root_quat}")

                # Check velocities
                lin_speed = np.linalg.norm(current_root_lin_vel)
                ang_speed = np.linalg.norm(current_root_ang_vel)
                if lin_speed > 2.0:  # >2 m/s
                    print(f"[{self.name}]   ⚠️  WARNING: High linear velocity {lin_speed:.4f} m/s")
                if ang_speed > 5.0:  # >5 rad/s
                    print(f"[{self.name}]   ⚠️  WARNING: High angular velocity {ang_speed:.4f} rad/s")

            full_action = self._full_action_buf
            full_action.zero_()

            # 1. Get action based on replay mode
            if self.replay_mode == "inference":
                # Inference mode: use recorded observation directly for ONNX inference
                if self.replay_data_obs is None:
                    raise ValueError(f"[{self.name}] No observation data available for inference mode")

                # Get recorded observation buffer [1432] - use it directly!
                recorded_obs_buf = self.replay_data_obs[self.current_frame]  # [1432]
                obs_tensor = torch.from_numpy(recorded_obs_buf).unsqueeze(0).float()  # [1, 1432]

                # ONNX inference with recorded observation
                ort_inputs = {self.input_name: obs_tensor.cpu().numpy()}
                ort_outputs = self.policy.run(None, ort_inputs)
                action_data = torch.from_numpy(ort_outputs[0])  # [1, 29] - raw ONNX output

                # 2. Action preparation (same as original)
                raw_action = torch.clip(action_data.to(self.env.device, dtype=torch.float32), -10.0, 10.0)
                target_29 = raw_action * 0.5 + self.twist2_default_pos  # [1,29]

                # Debug: Print action statistics
                if self.current_frame % 100 == 0:
                    print(f"[{self.name}] Frame {self.current_frame}: Inference mode")
                    print(f"  raw_action range: [{raw_action.min():.4f}, {raw_action.max():.4f}]")
                    print(f"  target_29 range: [{target_29.min():.4f}, {target_29.max():.4f}]")
            else:
                # Direct mode: use the recorded TWIST2 target positions that were sent to PD.
                target_29 = torch.from_numpy(self.replay_data_qpos[self.current_frame]).unsqueeze(0).to(
                    self.env.device, dtype=torch.float32)  # [1, 29]

            # Progress reporting with detailed debug info
            if self.current_frame % 100 == 0:
                print(f"[{self.name}] Replay progress: {self.current_frame}/{self.total_frames} (Frame 0 used for init)")
                # Print target_29 statistics
                target_29_np = target_29.cpu().numpy().squeeze(0)
                print(f"[{self.name}]   target_29 range: [{target_29_np.min():.4f}, {target_29_np.max():.4f}]")
                print(f"[{self.name}]   target_29 mean: {target_29_np.mean():.4f}, std: {target_29_np.std():.4f}")
                # Print first 5 joint targets
                print(f"[{self.name}]   First 5 joints: {target_29_np[:5]}")
                # If in direct mode, compare with recorded qpos
                if self.replay_mode == "direct":
                    recorded_qpos = self.replay_data_qpos[self.current_frame]
                    qpos_diff = np.abs(target_29_np - recorded_qpos).max()
                    print(f"[{self.name}]   Direct mode: qpos diff = {qpos_diff:.6f} (should be ~0)")


            # 3. Fill full action buffer
            # Fill defaults first
            full_action.copy_(self.env.scene["robot"].data.default_joint_pos.squeeze(0))
            # Overwrite the 29 controlled joints
            full_action.index_copy_(0, torch.tensor(self.twist2_action_indices, device=self.env.device,
                                                    dtype=torch.long), target_29.squeeze(0))
            self._apply_recorded_hand_action(full_action, self.current_frame)

            # Note: We don't set root state for frames > 0
            # The initial state (frame 0) is set correctly, and physics evolves naturally
            # This avoids the timing mismatch issue and provides better stability


            # 4. Physics simulation loop (same as original)
            for i in range(self._twist2_decimation):
                # Set joint targets and write to sim
                self.env.scene["robot"].set_joint_position_target(full_action)
                self.env.scene.write_data_to_sim()

                # Physics step with optional rendering
                is_last_step = (i == self._twist2_decimation - 1)
                should_render = is_last_step and (self._render_counter % self._render_interval == 0)

                if should_render:
                    self.env.sim.step(render=True)
                else:
                    self.env.sim.step(render=False)

                # Scene update
                self.env.scene.update(dt=self.env.physics_dt)

            self._render_counter += 1

            # 5. Physics state analysis (compare recording vs replay)
            self._analyze_physics_state(self.current_frame)

            # 6. Observation computation (same as original)
            self._obs_counter += 1
            if self._obs_counter % self._obs_interval == 0:
                self.env.observation_manager.compute()

            # Advance to next frame
            self.current_frame += 1

            return full_action

        except Exception as e:
            print(f"[{self.name}] Get replay action failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_physics_state(self, frame_idx: int):
        """Analyze physics state differences between recording and replay

        Compares applied torques and contact forces to understand dynamics differences.
        Note: These states cannot be directly written, but analysis helps diagnose issues.

        Args:
            frame_idx: Current frame index
        """
        if frame_idx >= self.total_frames:
            return

        # Only analyze every 100 frames to avoid performance impact
        if frame_idx % 100 != 0:
            return

        robot = self.env.scene["robot"]

        # Analyze applied torques
        if self._enable_torque_analysis and self.replay_data_applied_torque is not None:
            try:
                # Get current applied torque from replay simulation
                replay_torque = robot.data.applied_torque[0, self.twist2_action_indices].cpu().numpy()  # [29]

                # Get recorded applied torque from recording
                recorded_torque = self.replay_data_applied_torque[frame_idx]  # [29]

                # Compute difference
                torque_diff = np.abs(replay_torque - recorded_torque)
                max_torque_diff = np.max(torque_diff)
                mean_torque_diff = np.mean(torque_diff)

                # Accumulate statistics
                self._torque_diff_sum += mean_torque_diff
                self._torque_diff_count += 1

                # Print analysis
                if frame_idx % 500 == 0:  # Print every 500 frames
                    avg_torque_diff = self._torque_diff_sum / max(1, self._torque_diff_count)
                    print(f"\n[{self.name}] 🔧 Applied Torque Analysis (Frame {frame_idx}):")
                    print(f"  Max torque diff: {max_torque_diff:.4f} Nm")
                    print(f"  Mean torque diff: {mean_torque_diff:.4f} Nm")
                    print(f"  Avg torque diff (overall): {avg_torque_diff:.4f} Nm")

                    # Find joints with largest differences
                    top_3_indices = np.argsort(torque_diff)[-3:][::-1]
                    print(f"  Top 3 joints with largest torque diff:")
                    for idx in top_3_indices:
                        print(f"    Joint {idx}: recorded={recorded_torque[idx]:.4f}, replay={replay_torque[idx]:.4f}, diff={torque_diff[idx]:.4f} Nm")

            except Exception as e:
                print(f"[{self.name}] Warning: Failed to analyze applied torque: {e}")

        # Analyze contact forces
        if self._enable_contact_analysis and self.replay_data_contact_forces is not None:
            try:
                # Get current contact forces from replay simulation
                replay_contact = robot.data.body_net_contact_force_w[0].cpu().numpy()  # [num_bodies, 3]

                # Get recorded contact forces from recording
                recorded_contact = self.replay_data_contact_forces[frame_idx]  # [num_bodies, 3]

                # Compute difference (L2 norm for each body)
                contact_diff = np.linalg.norm(replay_contact - recorded_contact, axis=1)  # [num_bodies]
                max_contact_diff = np.max(contact_diff)
                mean_contact_diff = np.mean(contact_diff)

                # Accumulate statistics
                self._contact_diff_sum += mean_contact_diff
                self._contact_diff_count += 1

                # Print analysis
                if frame_idx % 500 == 0:  # Print every 500 frames
                    avg_contact_diff = self._contact_diff_sum / max(1, self._contact_diff_count)
                    print(f"\n[{self.name}] 🤝 Contact Force Analysis (Frame {frame_idx}):")
                    print(f"  Max contact diff: {max_contact_diff:.4f} N")
                    print(f"  Mean contact diff: {mean_contact_diff:.4f} N")
                    print(f"  Avg contact diff (overall): {avg_contact_diff:.4f} N")

                    # Find bodies with largest contact differences
                    top_3_bodies = np.argsort(contact_diff)[-3:][::-1]
                    print(f"  Top 3 bodies with largest contact diff:")
                    for body_idx in top_3_bodies:
                        recorded_force = np.linalg.norm(recorded_contact[body_idx])
                        replay_force = np.linalg.norm(replay_contact[body_idx])
                        print(f"    Body {body_idx}: recorded={recorded_force:.4f}, replay={replay_force:.4f}, diff={contact_diff[body_idx]:.4f} N")

            except Exception as e:
                print(f"[{self.name}] Warning: Failed to analyze contact forces: {e}")

    def cleanup(self):
        """Cleanup resources"""
        print(f"[{self.name}] Cleaning up replay action provider")

        super().cleanup()
