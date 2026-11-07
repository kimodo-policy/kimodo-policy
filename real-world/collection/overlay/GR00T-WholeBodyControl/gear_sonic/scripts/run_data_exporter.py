"""
Sonic VLA data exporter for G1 -- NO ROS 2 DEPENDENCY.

All data sources use ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (port 5557, from C++ zmq_output_handler)
  2. SMPL pose    -> ZMQ SUB on ``pose`` topic     (port 5556, from pico_manager_thread_server)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor

Robot config (``script_config`` in info.json) is read from the ``robot_config``
ZMQ topic re-published every ~2 s by the C++ process.  If the config is not
received within the timeout the exporter exits with an error.

Virtual environment setup (run from repo root):
    bash install_scripts/install_data_collection.sh
    source .venv_data_collection/bin/activate

Usage (from repo root):
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cup"
    python gear_sonic/scripts/run_data_exporter.py --task-prompt "walk forward" --dataset-name my_session
"""

from collections import deque
import copy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import time

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation as R
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.scripts.validate_g1_vla_dataset import validate_dataset
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE as POSE_HEADER_SIZE

_MIN_VALID_TARGET_ROOT_NORM_M = 0.2
_MAX_CONTINUOUS_TARGET_ROOT_STEP_M = 0.2
_MAX_CONTINUOUS_TARGET_ROOT_ROTATION_RAD = np.deg2rad(30.0)
_DEFAULT_ROOT_HEIGHT_M = 0.793
_HAND_CLOSED_JOINT_THRESHOLD_RAD = 0.2
_MAX_STATE_CAMERA_RECEIVE_SKEW_SEC = 0.08
_MAX_POSE_AGE_SEC = 0.10
_MIN_REFERENCE_JOINT_NORM_RAD = 1e-6
_RECORDING_STALL_WARNING_SEC = 0.5
_RECORDING_STALL_ABORT_SEC = 1.0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SonicDataExporterConfig:
    """CLI config for the ROS-free Sonic data exporter."""

    # Dataset
    dataset_name: str | None = None
    """Dataset name (auto-generated if creating new)."""

    task_prompt: str = "demo"
    """Language task prompt."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    convert_to_humanoid_arena: bool = True
    """Convert the saved dataset to HumanoidArena V3.1 when the exporter exits."""

    humanoid_arena_output_root: str | None = None
    """HumanoidArena output root. Defaults to '<dataset>_humanoidarena_v3_1_aligned'."""

    humanoid_arena_task_name: str | None = None
    """Required HumanoidArena task bucket used for Kimodo loader mapping."""

    humanoid_arena_root_height: float = _DEFAULT_ROOT_HEIGHT_M
    """Absolute root height for HumanoidArena action[2] when root-z-mode is constant/offset."""

    humanoid_arena_root_z_mode: str = "constant"
    """How to map episode-relative action.root_p z: constant, offset, or relative."""

    humanoid_arena_root_position_axis_order: str = "xyz"
    """Source action.root_p axes to use as HumanoidArena xyz."""

    humanoid_arena_video_mode: str = "copy"
    """How to materialize HumanoidArena videos: copy, symlink, or skip."""

    humanoid_arena_overwrite: bool = True
    """Overwrite the derived HumanoidArena output directory if it already exists."""

    data_collection_frequency: int = 30
    """Data collection frequency (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Sonic / SMPL pose (from pico_manager_thread_server)
    sonic_zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose messages."""

    sonic_zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose messages."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # Robot config
    robot_config_timeout: float = 0
    """Seconds to wait for the ZMQ robot_config message at startup (0 = wait forever)."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist). Requires cameras to be available."""

    text_to_speech: bool = True
    """Use text-to-speech voice feedback."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][4096-byte JSON header][concatenated binary fields]
    """
    HEADER_SIZE = POSE_HEADER_SIZE

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset : offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


# ---------------------------------------------------------------------------
# Data Collector
# ---------------------------------------------------------------------------


class GrootDataCollector:
    """Collects data from G1 robot in Sonic CPP + SMPL mode -- no ROS 2.

    Data sources (all ZMQ):
      - ``g1_debug`` topic        -> proprio (body_q, hand_q, actions, base_quat, ...)
      - ``pose`` topic            -> SMPL pose (smpl_joints, body_quat_w, hand_joints, ...)
      - ``planner`` topic         -> planner commands (vr_position, vr_orientation, ...)
      - ``manager_state`` topic   -> current stream mode + toggle flags
      - Camera client             -> ego-view images
    """

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech=None,
        frequency: int = 20,
        sonic_data_zmq_host: str = "localhost",
        sonic_data_zmq_port: int = 5556,
        state_zmq_host: str = "localhost",
        state_zmq_port: int = 5557,
    ):
        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.loop_period = 1.0 / frequency
        self.data_exporter = data_exporter
        self.robot_model = robot_model

        self._episode_state = EpisodeState()
        self._keyboard_listener = ZMQKeyboardSubscriber()

        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None
        self.latest_sonic_msg = None
        self.latest_planner_msg = None
        self._last_recorded_state_sequence: int | None = None
        self._last_recorded_camera_sequence: int | None = None
        self._last_recorded_reference_sequence: int | None = None
        self._last_recorded_smpl_sequence: int | None = None
        self._episode_timestamp0: float | None = None
        self._episode_grid_index: int | None = None

        self.current_stream_mode = 0

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._state_subscriber = ZMQStateSubscriber(
            host=state_zmq_host,
            port=state_zmq_port,
        )

        self._sonic_zmq_ctx = None
        self._sonic_zmq_socket = None
        try:
            self._sonic_zmq_ctx = zmq.Context()
            self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
            self._sonic_zmq_socket.connect(f"tcp://{sonic_data_zmq_host}:{sonic_data_zmq_port}")
            self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 100)
            self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
            self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
            self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
            time.sleep(0.5)
            print(f"[Sonic] Connected to ZMQ at {sonic_data_zmq_host}:{sonic_data_zmq_port}")
            print("[Sonic] Subscribed to: pose, planner, manager_state")
        except Exception as e:
            print(f"[Sonic] Warning: Failed to initialize ZMQ subscriber: {e}")
            self._sonic_zmq_socket = None

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3, reset_timeout_sec=5, time_delta=0.1
        )

        self._last_latency_log_time = 0.0
        self._last_source_wait_log_time = 0.0
        self._recording_stall_started_at: float | None = None
        self._last_resampled_frame: dict | None = None
        self._last_resampled_grid_index: int | None = None
        self._last_camera_capture_source_timestamp: float | None = None
        self._pending_camera_skipped_captures = 0
        self._last_recording_status_log_time = 0.0
        self._initial_yaw = None
        # Per-episode root references.  Action root targets come from the C++
        # visualisation payload; the measured root quaternion comes from IMU.
        self._episode_state_root_q0: np.ndarray | None = None
        self._episode_action_root_p0: np.ndarray | None = None
        self._episode_action_root_q0: np.ndarray | None = None
        self._episode_action_root_output_p0 = np.zeros(3, dtype=np.float64)
        self._episode_action_root_output_q0 = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._episode_action_root_last_p = np.zeros(3, dtype=np.float64)
        self._episode_action_root_last_q = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._episode_action_root_last_z = np.array(
            [_DEFAULT_ROOT_HEIGHT_M], dtype=np.float64
        )
        self._episode_action_root_raw_p_last: np.ndarray | None = None
        self._episode_action_root_raw_q_last: np.ndarray | None = None
        self._episode_action_root_target_was_valid = False
        self._last_recorded_state_sequence = None
        self._last_recorded_camera_sequence = None
        self._last_recorded_reference_sequence = None
        self._last_recorded_smpl_sequence = None
        self._episode_timestamp0 = None
        self._recording_stall_started_at = None
        self._last_resampled_frame = None
        self._last_resampled_grid_index = None
        self._last_camera_capture_source_timestamp = None
        self._pending_camera_skipped_captures = 0

        print(f"Recording to {self.data_exporter.meta.root}")
        print(
            "Recording synchronization: transient source skew is tolerated; "
            f"continuous stalls abort after {_RECORDING_STALL_ABORT_SEC:.1f}s"
        )

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        # Highlight lifecycle events without changing recording or saving logic.
        label = None
        color = "33"
        if message.startswith("Started recording episode"):
            label, color = ">>> REC | 开始录制", "32"
        elif message.startswith(("Stopping recording episode", "Saving buffered episode")):
            label = "[STOP] SAVING | 停止录制，正在保存"
        elif message.startswith("Finished saving episode"):
            label, color = "[OK] SAVED | 保存完成", "36"
        elif message.startswith(("Discarded episode", "[Recording aborted]")):
            label, color = "[X] DISCARDED | 已丢弃", "31"
        elif message.startswith(("Cannot start recording", "Skipping save", "Error saving episode")):
            label, color = "[!] ATTENTION | 录制或保存异常", "31"
        if label is not None:
            banner = f"\n{'=' * 68}\n{label}\n{message}\n{'=' * 68}\n"
            # Keep redirected logs readable; honor the standard NO_COLOR switch.
            if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
                banner = f"\033[1;{color}m{banner}\033[0m"
            print(banner, flush=True)
            if self.text_to_speech is not None and say:
                self.text_to_speech.say(message, blocking=blocking)
        elif self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message, flush=True)

    def _reset_episode_root_references(self) -> None:
        """Reset root references so the next recording starts its own frame."""
        self._episode_state_root_q0 = None
        self._episode_action_root_p0 = None
        self._episode_action_root_q0 = None
        self._episode_action_root_output_p0 = np.zeros(3, dtype=np.float64)
        self._episode_action_root_output_q0 = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._episode_action_root_last_p = np.zeros(3, dtype=np.float64)
        self._episode_action_root_last_q = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._episode_action_root_last_z = np.array(
            [_DEFAULT_ROOT_HEIGHT_M], dtype=np.float64
        )
        self._episode_action_root_raw_p_last = None
        self._episode_action_root_raw_q_last = None
        self._episode_action_root_target_was_valid = False

    def _reset_episode_stream_cursors(self) -> None:
        """Reset per-episode source cursors before accepting the first frame."""
        self._last_recorded_state_sequence = None
        self._last_recorded_camera_sequence = None
        self._last_recorded_reference_sequence = None
        self._last_recorded_smpl_sequence = None
        self._episode_timestamp0 = None
        self._episode_grid_index = None
        self._last_resampled_frame = None
        self._last_resampled_grid_index = None
        self._last_camera_capture_source_timestamp = None
        self._pending_camera_skipped_captures = 0

    @staticmethod
    def _normalise_quaternion(value: object, default: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1) if value is not None else None
        if arr is None or arr.size != 4 or not np.all(np.isfinite(arr)):
            return default.copy()
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            return default.copy()
        return arr / norm

    @staticmethod
    def _has_valid_root_target(proprio: dict) -> bool:
        target_p = np.asarray(
            proprio.get("base_trans_reference_target", []), dtype=np.float64
        ).reshape(-1)
        target_q = np.asarray(
            proprio.get("base_quat_reference_target", []), dtype=np.float64
        ).reshape(-1)
        return bool(
            target_p.size == 3
            and np.all(np.isfinite(target_p))
            and np.linalg.norm(target_p) >= _MIN_VALID_TARGET_ROOT_NORM_M
            and target_q.size == 4
            and np.all(np.isfinite(target_q))
            and np.linalg.norm(target_q) >= 1e-8
        )

    @staticmethod
    def _heading_quaternion(q: np.ndarray) -> np.ndarray:
        """Return the yaw-only heading component of a scalar-first quaternion."""
        yaw = R.from_quat(q, scalar_first=True).as_euler("xyz")[2]
        return R.from_euler("z", yaw).as_quat(scalar_first=True).astype(np.float64)

    @classmethod
    def _heading_relative_quaternion(cls, q0: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Return heading(q0)^-1 * q in scalar-first (w, x, y, z) order."""
        heading0 = R.from_quat(cls._heading_quaternion(q0), scalar_first=True)
        relative = heading0.inv() * R.from_quat(q, scalar_first=True)
        return relative.as_quat(scalar_first=True).astype(np.float64)

    @classmethod
    def _heading_relative_position(
        cls,
        p0: np.ndarray, q0: np.ndarray, p: np.ndarray
    ) -> np.ndarray:
        """Express ``p - p0`` in the first-frame heading coordinate frame."""
        first_heading_rotation = R.from_quat(
            cls._heading_quaternion(q0), scalar_first=True
        )
        return first_heading_rotation.inv().apply(p - p0).astype(np.float64)

    def _add_episode_root_features(self, frame_data: dict, proprio: dict) -> None:
        """Add episode-relative state root-q and target action root pose.

        ``base_trans_reference_target``/``base_quat_reference_target`` are the
        unmodified GMR reference-motion root pose emitted by the C++ deployer.
        The heading-adjusted PD/control visualisation targets are intentionally
        not used. ``base_trans_measured`` is intentionally
        not used: the current deployer publishes a fixed visualisation default,
        not a measured real-robot pelvis position.
        """
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        state_q = self._normalise_quaternion(proprio.get("base_quat"), identity)
        if self._episode_state_root_q0 is None:
            self._episode_state_root_q0 = state_q.copy()
        frame_data["observation.root_q_relative"] = self._heading_relative_quaternion(
            self._episode_state_root_q0, state_q
        )

        target_p = np.asarray(
            proprio.get("base_trans_reference_target", []), dtype=np.float64
        ).reshape(-1)
        target_q_raw = np.asarray(
            proprio.get("base_quat_reference_target", []), dtype=np.float64
        ).reshape(-1)
        target_valid = self._has_valid_root_target(proprio)
        frame_data["action.root_target_valid"] = np.array(
            [float(target_valid)], dtype=np.float32
        )
        frame_data["action.root_target_discontinuous"] = np.array(
            [0.0], dtype=np.float32
        )

        if not target_valid:
            raise ValueError("GMR root reference is invalid; refusing to save a placeholder frame")

        target_q = self._normalise_quaternion(target_q_raw, identity)
        target_z = np.array([float(target_p[2])], dtype=np.float64)
        position_discontinuous = (
            self._episode_action_root_raw_p_last is not None
            and np.linalg.norm(target_p - self._episode_action_root_raw_p_last)
            > _MAX_CONTINUOUS_TARGET_ROOT_STEP_M
        )
        rotation_discontinuous = False
        if self._episode_action_root_raw_q_last is not None:
            rotation_step = (
                R.from_quat(self._episode_action_root_raw_q_last, scalar_first=True).inv()
                * R.from_quat(target_q, scalar_first=True)
            ).magnitude()
            rotation_discontinuous = (
                rotation_step > _MAX_CONTINUOUS_TARGET_ROOT_ROTATION_RAD
            )
        target_discontinuous = position_discontinuous or rotation_discontinuous
        frame_data["action.root_target_discontinuous"] = np.array(
            [float(target_discontinuous)], dtype=np.float32
        )
        if target_discontinuous:
            raise ValueError("GMR root reference is discontinuous; refusing to reconnect trajectory")
        if self._episode_action_root_p0 is None or self._episode_action_root_q0 is None:
            self._episode_action_root_p0 = target_p.copy()
            self._episode_action_root_q0 = target_q.copy()
        action_root_p = self._heading_relative_position(
            self._episode_action_root_p0, self._episode_action_root_q0, target_p
        )
        action_root_q = self._heading_relative_quaternion(
            self._episode_action_root_q0, target_q
        )

        self._episode_action_root_last_p = action_root_p.copy()
        self._episode_action_root_last_q = action_root_q.copy()
        self._episode_action_root_last_z = target_z.copy()
        self._episode_action_root_raw_p_last = target_p.copy()
        self._episode_action_root_raw_q_last = target_q.copy()
        self._episode_action_root_target_was_valid = True
        frame_data["action.root_p"] = action_root_p
        frame_data["action.root_q"] = action_root_q
        frame_data["action.root_z"] = target_z

    def _poll_state_zmq(self):
        """Poll the ``g1_debug`` ZMQ topic for robot state (non-blocking)."""
        msg = self._state_subscriber.get_msg(clear=True)
        if msg is None:
            return

        msg["receive_monotonic_timestamp"] = time.monotonic()
        msg["receive_wall_timestamp"] = time.time()

        self.latest_proprio_msg = msg

    def _check_recording_commands(self):
        """Check keyboard + ZMQ toggle flags for recording commands."""
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            state = self._episode_state.get_state()
            if state == self._episode_state.IDLE:
                missing_sources = self._missing_recording_sources()
                if missing_sources:
                    self._print_and_say(
                        "Cannot start recording: missing " + ", ".join(missing_sources),
                        say=False,
                    )
                    return

                self._episode_state.change_state()
                self._initial_yaw = None
                self._reset_episode_stream_cursors()
                self._reset_episode_root_references()
                self._last_recording_status_log_time = 0.0
                self._print_and_say(
                    f"Started recording episode {self.current_episode_index} | "
                    "proprio=True image=True",
                    blocking=False,
                )
            elif state == self._episode_state.RECORDING:
                episode_index = self.current_episode_index
                frame_count = self.data_exporter.episode_buffer.get("size", 0)
                self._episode_state.change_state()
                self._print_and_say(
                    f"Stopping recording episode {episode_index} with "
                    f"{frame_count} frames, preparing to save",
                    blocking=False,
                )
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._reset_episode_root_references()
                self._print_and_say("Discarded episode", blocking=False)

    def _missing_recording_sources(self) -> list[str]:
        """Return required sources that have not produced a message yet."""
        missing = []
        if self.latest_proprio_msg is None:
            missing.append("proprio")
        else:
            if not self._has_valid_measured_state(self.latest_proprio_msg):
                missing.append("valid measured G1 state")
            if not self._has_valid_root_target(self.latest_proprio_msg):
                missing.append("valid root target")
            if not self._has_valid_raw_gmr_root(self.latest_proprio_msg):
                missing.append("matching raw GMR root")
            if not self._has_valid_reference_joint_target(self.latest_proprio_msg):
                missing.append("valid reference joint target")
            if not self._has_state_sync_metadata(self.latest_proprio_msg):
                missing.append("state monotonic timestamp/sequence")
            if not self._has_reference_sync_metadata(self.latest_proprio_msg):
                missing.append("reference monotonic timestamp/sequence")
            if self.current_stream_mode in (1, 4) and not self._reference_target_is_gmr(
                self.latest_proprio_msg
            ):
                missing.append("GMR reference joint target")
        if self.latest_image_msg is None:
            missing.append("image")
        elif not self._has_camera_sync_metadata(self.latest_image_msg):
            missing.append("camera monotonic timestamp/sequence")
        if self.current_stream_mode in (1, 4):
            latest_sonic_msg = getattr(self, "latest_sonic_msg", None)
            if latest_sonic_msg is None:
                missing.append("SMPL/GMR source")
            elif not self._has_pose_sync_metadata(latest_sonic_msg):
                missing.append("SMPL/GMR timestamp/sequence")
        return missing

    @staticmethod
    def _has_valid_measured_state(proprio: dict) -> bool:
        joint_q = np.asarray(proprio.get("body_q", []), dtype=np.float64).reshape(-1)
        joint_dq = np.asarray(proprio.get("body_dq", []), dtype=np.float64).reshape(-1)
        root_q = np.asarray(proprio.get("base_quat", []), dtype=np.float64).reshape(-1)
        return bool(
            joint_q.shape == (29,)
            and joint_dq.shape == (29,)
            and root_q.shape == (4,)
            and np.all(np.isfinite(joint_q))
            and np.all(np.isfinite(joint_dq))
            and np.all(np.isfinite(root_q))
            and np.isclose(np.linalg.norm(root_q), 1.0, atol=1e-4)
        )

    @staticmethod
    def _has_valid_raw_gmr_root(proprio: dict) -> bool:
        target_p = np.asarray(
            proprio.get("base_trans_reference_target", []), dtype=np.float64
        ).reshape(-1)
        target_q = np.asarray(
            proprio.get("base_quat_reference_target", []), dtype=np.float64
        ).reshape(-1)
        raw_p = np.asarray(proprio.get("gmr_root_p_raw", []), dtype=np.float64).reshape(-1)
        raw_q = np.asarray(proprio.get("gmr_root_q_raw", []), dtype=np.float64).reshape(-1)
        return bool(
            target_p.shape == raw_p.shape == (3,)
            and target_q.shape == raw_q.shape == (4,)
            and np.all(np.isfinite(raw_p))
            and np.all(np.isfinite(raw_q))
            and np.allclose(target_p, raw_p, rtol=0.0, atol=1e-9)
            and np.allclose(target_q, raw_q, rtol=0.0, atol=1e-9)
        )

    def _recording_stall_abort_reason(
        self, missing_sources: list[str], stall_duration: float
    ) -> str | None:
        if self.data_exporter.episode_buffer.get("size", 0) == 0:
            return None
        if stall_duration < _RECORDING_STALL_ABORT_SEC:
            return None
        return (
            f"source synchronization stalled for {stall_duration:.1f}s: "
            + ", ".join(missing_sources)
        )

    @staticmethod
    def _scalar_int(value: object, default: int = -1) -> int:
        arr = np.asarray(value).reshape(-1) if value is not None else np.array([])
        return int(arr[0]) if arr.size == 1 else default

    @staticmethod
    def _scalar_float(value: object, default: float = 0.0) -> float:
        arr = np.asarray(value, dtype=np.float64).reshape(-1) if value is not None else np.array([])
        return float(arr[0]) if arr.size == 1 and np.isfinite(arr[0]) else default

    @classmethod
    def _has_state_sync_metadata(cls, proprio: dict) -> bool:
        return bool(
            cls._scalar_int(proprio.get("index")) >= 0
            and cls._scalar_float(proprio.get("state_monotonic_timestamp")) > 0.0
            and cls._scalar_float(proprio.get("receive_monotonic_timestamp")) > 0.0
        )

    @classmethod
    def _has_camera_sync_metadata(cls, image_msg: dict) -> bool:
        return bool(
            cls._scalar_int(image_msg.get("sequence")) >= 0
            and cls._scalar_float(image_msg.get("server_monotonic_timestamp")) > 0.0
            and cls._scalar_float(image_msg.get("receive_monotonic_timestamp")) > 0.0
        )

    @classmethod
    def _has_reference_sync_metadata(cls, proprio: dict) -> bool:
        return bool(
            cls._scalar_int(proprio.get("reference_source_frame")) >= 0
            and cls._scalar_float(proprio.get("reference_monotonic_timestamp")) > 0.0
        )

    @classmethod
    def _has_pose_sync_metadata(cls, pose_msg: dict) -> bool:
        frame_index = np.asarray(pose_msg.get("frame_index", [])).reshape(-1)
        left_hand = np.asarray(pose_msg.get("left_hand_joints", [])).reshape(-1)
        right_hand = np.asarray(pose_msg.get("right_hand_joints", [])).reshape(-1)
        return bool(
            frame_index.size == 1
            and int(frame_index[0]) >= 0
            and cls._scalar_float(pose_msg.get("source_monotonic_timestamp")) > 0.0
            and cls._scalar_float(pose_msg.get("receive_monotonic_timestamp")) > 0.0
            and left_hand.shape == (7,)
            and right_hand.shape == (7,)
            and np.all(np.isfinite(left_hand))
            and np.all(np.isfinite(right_hand))
        )

    def _frame_sync_blockers(self) -> list[str]:
        assert self.latest_proprio_msg is not None
        assert self.latest_image_msg is not None
        blockers = []
        state_receive = self._scalar_float(
            self.latest_proprio_msg.get("receive_monotonic_timestamp")
        )
        camera_receive = self._scalar_float(
            self.latest_image_msg.get("receive_monotonic_timestamp")
        )
        if self.latest_image_msg.get("is_new", True) and abs(
            state_receive - camera_receive
        ) > _MAX_STATE_CAMERA_RECEIVE_SKEW_SEC:
            blockers.append("state/camera timestamp alignment")

        if self.current_stream_mode in (1, 4):
            assert self.latest_sonic_msg is not None
            pose_receive = self._scalar_float(
                self.latest_sonic_msg.get("receive_monotonic_timestamp")
            )
            if time.monotonic() - pose_receive > _MAX_POSE_AGE_SEC:
                blockers.append("fresh SMPL/GMR source")
        return blockers

    def _log_recording_progress(self) -> None:
        """Print a low-rate heartbeat that proves frames are entering the buffer."""
        now = time.monotonic()
        last_log = getattr(self, "_last_recording_status_log_time", 0.0)
        if now - last_log < 1.0:
            return

        self._last_recording_status_log_time = now
        self._print_and_say(
            f"[Recording] episode={self.current_episode_index} "
            f"frames={self.data_exporter.episode_buffer.get('size', 0)} | "
            f"proprio={self.latest_proprio_msg is not None} "
            f"image={self.latest_image_msg is not None}",
            say=False,
        )

    def _poll_sonic_zmq_messages(self):
        """Poll ZMQ for pose, planner, and manager_state messages (non-blocking)."""
        if self._sonic_zmq_socket is None:
            return

        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            self.current_stream_mode = int(data["stream_mode"].flat[0])

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="planner")
        except Exception:
            return

        planner_mode = int(data["mode"].flat[0]) if "mode" in data else 0
        planner_movement = (
            data["movement"].flatten().astype(np.float32)
            if "movement" in data and data["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        planner_facing = (
            data["facing"].flatten().astype(np.float32)
            if "facing" in data and data["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        planner_speed = float(data["speed"].flat[0]) if "speed" in data else -1.0
        planner_height = float(data["height"].flat[0]) if "height" in data else -1.0

        vr_3pt_position = None
        if "vr_position" in data and data["vr_position"].size == 9:
            vr_3pt_position = data["vr_position"].flatten().astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in data and data["vr_orientation"].size == 12:
            vr_3pt_orientation = data["vr_orientation"].flatten().astype(np.float32)

        self.latest_planner_msg = {
            "planner_mode": planner_mode,
            "planner_movement": planner_movement,
            "planner_facing": planner_facing,
            "planner_speed": planner_speed,
            "planner_height": planner_height,
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
            "receive_monotonic_timestamp": time.monotonic(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        G1_L_WRIST_ROLL_IDX = 23
        G1_L_WRIST_PITCH_IDX = 25
        G1_L_WRIST_YAW_IDX = 27
        G1_R_WRIST_ROLL_IDX = 24
        G1_R_WRIST_PITCH_IDX = 26
        G1_R_WRIST_YAW_IDX = 28

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Sonic] Error unpacking pose message: {e}")
            return

        try:
            if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
                return

            left_wrist_joints = None
            right_wrist_joints = None
            if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
                joint_pos = pose_data["joint_pos"][0]
                left_wrist_joints = np.array(
                    [
                        joint_pos[G1_L_WRIST_ROLL_IDX],
                        joint_pos[G1_L_WRIST_PITCH_IDX],
                        joint_pos[G1_L_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )
                right_wrist_joints = np.array(
                    [
                        joint_pos[G1_R_WRIST_ROLL_IDX],
                        joint_pos[G1_R_WRIST_PITCH_IDX],
                        joint_pos[G1_R_WRIST_YAW_IDX],
                    ],
                    dtype=np.float32,
                )

            frame_index = None
            if "frame_index" in pose_data:
                frame_index = np.array([pose_data["frame_index"].flat[-1]], dtype=np.int64)

            smpl_pose = np.zeros(63, dtype=np.float32)
            if "smpl_pose" in pose_data:
                raw_pose = pose_data["smpl_pose"]
                if raw_pose.ndim == 3:
                    smpl_pose = raw_pose[0].flatten().astype(np.float32)
                elif raw_pose.ndim == 2:
                    smpl_pose = raw_pose.flatten().astype(np.float32)
                elif raw_pose.ndim == 1 and raw_pose.size == 63:
                    smpl_pose = raw_pose.astype(np.float32)

            left_hand_joints = self._extract_hand_joints(pose_data, "left_hand_joints")
            right_hand_joints = self._extract_hand_joints(pose_data, "right_hand_joints")

            vr_3pt_position = None
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
                vr_3pt_position = pose_data["vr_position"].flatten().astype(np.float32)
            vr_3pt_orientation = None
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12:
                vr_3pt_orientation = pose_data["vr_orientation"].flatten().astype(np.float32)

            self.latest_sonic_msg = {
                "smpl_joints": pose_data["smpl_joints"][-1],
                "smpl_pose": smpl_pose,
                "body_quat_w": (
                    pose_data["body_quat_w"][-1] if "body_quat_w" in pose_data else None
                ),
                "left_hand_joints": left_hand_joints,
                "right_hand_joints": right_hand_joints,
                "left_wrist_joints": left_wrist_joints,
                "right_wrist_joints": right_wrist_joints,
                "vr_3pt_position": vr_3pt_position,
                "vr_3pt_orientation": vr_3pt_orientation,
                "frame_index": frame_index,
                "source_monotonic_timestamp": float(
                    pose_data["timestamp_monotonic"].flat[0]
                )
                if "timestamp_monotonic" in pose_data
                else 0.0,
                "source_wall_timestamp": float(
                    pose_data["timestamp_realtime"].flat[0]
                )
                if "timestamp_realtime" in pose_data
                else 0.0,
                "left_trigger": float(pose_data["left_trigger"].flat[0])
                if "left_trigger" in pose_data
                else None,
                "right_trigger": float(pose_data["right_trigger"].flat[0])
                if "right_trigger" in pose_data
                else None,
                "receive_timestamp": time.time(),
                "receive_monotonic_timestamp": time.monotonic(),
            }
        except Exception as e:
            if not hasattr(self, "_sonic_error_count"):
                self._sonic_error_count = 0
            self._sonic_error_count += 1
            if self._sonic_error_count == 1 or self._sonic_error_count % 100 == 0:
                print(f"[Sonic] Error processing pose message: {e}")

    @staticmethod
    def _extract_hand_joints(pose_data: dict, key: str) -> np.ndarray:
        arr = pose_data.get(key)
        if arr is not None:
            if arr.ndim > 1:
                arr = arr[0]
            return arr.astype(np.float32)
        return np.zeros(7, dtype=np.float32)

    @staticmethod
    def _extract_bool(pose_data: dict, key: str) -> bool:
        val = pose_data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    def _log_latency_periodic(
        self,
        sonic_latency_ms: float | None = None,
    ):
        current_time = time.time()
        if current_time - self._last_latency_log_time >= 1.0:
            self._last_latency_log_time = current_time
            parts = []
            if sonic_latency_ms is not None:
                parts.append(f"Sonic Pose: {sonic_latency_ms:.1f}ms")
            if parts:
                print(f"[Latency] {', '.join(parts)}")

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image '{image_key}' for feature '{feature_name}' "
                        f"not found in image message. Available: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            episode_index = self.current_episode_index
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            output_path = Path(self.data_exporter.meta.root).resolve()
            if buffer_size > 0:
                self.data_exporter.save_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                self._print_and_say(
                    f"Finished saving episode {episode_index}: {buffer_size} frames "
                    f"-> {output_path}"
                )
            else:
                self._print_and_say(
                    f"Skipping save for episode {episode_index}: no frames collected",
                    say=False,
                )
            self._episode_state.change_state()
        return True

    def _add_data_frame(self):
        t_start = time.monotonic()

        state = self._episode_state.get_state()
        if state == self._episode_state.NEED_TO_SAVE:
            return self._finalize_frame(t_start)
        if state != self._episode_state.RECORDING:
            return False

        missing_sources = self._missing_recording_sources()
        if not missing_sources:
            missing_sources = self._frame_sync_blockers()
        if missing_sources:
            now = time.monotonic()
            stall_started_at = getattr(self, "_recording_stall_started_at", None)
            if stall_started_at is None:
                self._recording_stall_started_at = now
                return False
            last_log = getattr(self, "_last_source_wait_log_time", 0.0)
            stall_duration = now - stall_started_at
            abort_reason = self._recording_stall_abort_reason(
                missing_sources, stall_duration
            )
            if abort_reason is not None:
                self._print_and_say(
                    f"[Recording aborted] {abort_reason}; episode marked discarded",
                    say=False,
                )
                self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._reset_episode_stream_cursors()
                self._reset_episode_root_references()
                return False
            if (
                stall_duration >= _RECORDING_STALL_WARNING_SEC
                and now - last_log >= 1.0
            ):
                self._last_source_wait_log_time = now
                self._print_and_say(
                    f"[Recording waiting {stall_duration:.1f}s] missing "
                    + ", ".join(missing_sources)
                    + " | "
                    f"proprio={self.latest_proprio_msg is not None} "
                    f"image={self.latest_image_msg is not None}",
                    say=False,
                )
            return False

        self._recording_stall_started_at = None
        try:
            frame_added = self._add_data_frame_sonic(t_start)
        except ValueError as exc:
            self._print_and_say(
                f"[Recording aborted] {exc}; episode marked discarded",
                say=False,
            )
            if self.data_exporter.episode_buffer.get("size", 0) > 0:
                self.data_exporter.save_episode_as_discarded()
            else:
                self.data_exporter.skip_and_start_new_episode()
            self._episode_state.reset_state()
            self._initial_yaw = None
            self._reset_episode_root_references()
            return False
        if frame_added:
            self._log_recording_progress()
        return frame_added

    def _add_data_frame_sonic(self, t_start: float) -> bool:
        """Build one data frame in Sonic CPP + SMPL mode."""
        assert self.latest_proprio_msg is not None
        proprio = self.latest_proprio_msg

        whole_q = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["body_q"],
            left_hand_actuated_joint_values=proprio["left_hand_q"],
            right_hand_actuated_joint_values=proprio["right_hand_q"],
        )
        whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=proprio["last_action"],
            left_hand_actuated_joint_values=proprio["last_left_hand_action"],
            right_hand_actuated_joint_values=proprio["last_right_hand_action"],
        )
        reference_joint_q = self._extract_reference_joint_target(proprio)
        control_joint_q_target = self._extract_body_joint_configuration(whole_action_wbc)

        self.robot_model.cache_forward_kinematics(whole_q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        observation_eef_state = np.concatenate(eef_parts)

        frame_data: dict = {
            "observation.state": whole_q,
            "observation.eef_state": observation_eef_state,
            "action.wbc": whole_action_wbc,
            # Preserve the raw commanded Dex3 target.  ``action.hand_binary``
            # remains a compatibility/diagnostic field; continuous training
            # must read this 14-D joint target instead of the binary command.
            "action.hand_q": np.concatenate(
                (
                    whole_action_wbc[[26, 27, 28, 22, 23, 24, 25]],
                    whole_action_wbc[[40, 41, 42, 36, 37, 38, 39]],
                )
            ).astype(np.float32, copy=False),
            "observation.joint_q": self._extract_body_joint_configuration(whole_q),
            "observation.joint_dq": self._extract_body_joint_velocity(proprio),
            "audit.g1_joint_q_measured_raw": np.asarray(
                proprio["body_q"], dtype=np.float64
            ).reshape(29).copy(),
            "audit.g1_joint_dq_measured_raw": np.asarray(
                proprio["body_dq"], dtype=np.float64
            ).reshape(29).copy(),
            "audit.g1_root_q_measured_raw": np.asarray(
                proprio["base_quat"], dtype=np.float64
            ).reshape(4).copy()
            / np.linalg.norm(np.asarray(proprio["base_quat"], dtype=np.float64)),
            "observation.hand_binary": self._hand_binary_from_joint_positions(
                proprio["left_hand_q"], proprio["right_hand_q"]
            ),
            "action.joint_q": reference_joint_q,
            "audit.gmr_joint_q_raw": np.asarray(
                proprio["gmr_joint_q_raw"], dtype=np.float64
            ).reshape(29).copy(),
            "audit.gmr_root_p_raw": np.asarray(
                proprio["gmr_root_p_raw"], dtype=np.float64
            ).reshape(3).copy(),
            "audit.gmr_root_q_raw": self._normalise_quaternion(
                proprio["gmr_root_q_raw"], np.zeros(4, dtype=np.float64)
            ),
            "control.joint_q_target": control_joint_q_target,
        }

        self._add_sync_features(frame_data, proprio)

        self._add_cpp_state_features(frame_data, proprio)
        self._add_episode_root_features(frame_data, proprio)

        sonic_latency_ms = self._add_sonic_pose_features(frame_data)

        self._add_images_to_frame_data(frame_data)

        self._apply_fixed_rate_alignment(frame_data)

        self._log_latency_periodic(sonic_latency_ms)

        self._validate_core_frame(frame_data)
        assert self.latest_image_msg is not None
        self.data_exporter.add_frame(frame_data)
        self._episode_grid_index = (self._episode_grid_index or 0) + 1
        self._last_recorded_state_sequence = self._scalar_int(proprio.get("index"))
        self._last_recorded_camera_sequence = self._scalar_int(
            self.latest_image_msg.get("sequence")
        )
        self._last_recorded_reference_sequence = self._scalar_int(
            proprio.get("reference_source_frame")
        )
        if self.current_stream_mode in (1, 4) and self.latest_sonic_msg is not None:
            self._last_recorded_smpl_sequence = self._scalar_int(
                self.latest_sonic_msg.get("frame_index")
            )
        return self._finalize_frame(t_start)

    def _apply_fixed_rate_alignment(self, frame_data: dict) -> None:
        """Align one output row to the fixed data clock.

        State/action/motion are rebuilt on every collector tick. Camera data is
        allowed to remain stale, so only the image and raw camera identity are
        repeated when the camera has not produced a new frame.
        """
        assert self.latest_image_msg is not None
        grid_index = self._episode_grid_index or 0
        camera_is_new = bool(self.latest_image_msg.get("is_new", True))
        camera_sequence = self._scalar_int(self.latest_image_msg.get("sequence"))
        camera_source_ts = self._scalar_float(
            self.latest_image_msg.get("server_monotonic_timestamp")
        )

        frame_data["timestamp"] = np.array(
            [grid_index / self.frequency], dtype=np.float32
        )
        frame_data["sync.video_sequence"] = np.array([grid_index], dtype=np.int64)
        frame_data["sync.video_capture_sequence"] = np.array(
            [camera_sequence], dtype=np.int64
        )
        frame_data["sync.video_capture_source_monotonic_timestamp"] = np.array(
            [camera_source_ts], dtype=np.float64
        )
        frame_data["sync.camera_resampled"] = np.array(
            [0.0 if camera_is_new else 1.0], dtype=np.float32
        )
        frame_data["sync.camera_source_gap_frames"] = np.array(
            [0 if camera_is_new else 1], dtype=np.int64
        )
        frame_data["sync.camera_source_skipped_captures"] = np.array(
            [0], dtype=np.int64
        )

    def _add_resampled_camera_frames(
        self, frame_data: dict, capture_sequence: int, capture_timestamp: float
    ) -> None:
        """Write a fixed-rate grid without compressing source-camera gaps.

        The camera capture timestamp remains authoritative. Missing grid slots
        use the previous complete bundle and are explicitly marked; raw
        capture identity is retained in dedicated audit columns.
        """
        if self._episode_timestamp0 is None:
            self._episode_timestamp0 = capture_timestamp
        previous_capture_timestamp = self._last_camera_capture_source_timestamp
        if (
            previous_capture_timestamp is not None
            and capture_timestamp <= previous_capture_timestamp
        ):
            raise ValueError("camera source timestamp repeated or rolled back")
        self._last_camera_capture_source_timestamp = capture_timestamp
        elapsed = max(0.0, capture_timestamp - self._episode_timestamp0)
        target_grid_index = int(round(elapsed * self.frequency))
        previous_grid_index = self._last_resampled_grid_index
        if previous_grid_index is None:
            previous_grid_index = -1
        if target_grid_index == previous_grid_index:
            self._pending_camera_skipped_captures += 1
            return
        if target_grid_index < previous_grid_index:
            raise ValueError("camera source timestamp mapped backward on the 30 Hz grid")

        for grid_index in range(previous_grid_index + 1, target_grid_index + 1):
            is_current_capture = grid_index == target_grid_index
            if is_current_capture or self._last_resampled_frame is None:
                output = copy.deepcopy(frame_data)
                raw_sequence = capture_sequence
                raw_timestamp = capture_timestamp
            else:
                output = copy.deepcopy(self._last_resampled_frame)
                raw_sequence = int(output["sync.video_capture_sequence"][0])
                raw_timestamp = float(output["sync.video_capture_source_monotonic_timestamp"][0])
            output["timestamp"] = np.array(
                [grid_index / self.frequency], dtype=np.float32
            )
            output["sync.video_sequence"] = np.array([grid_index], dtype=np.int64)
            output["sync.video_source_monotonic_timestamp"] = np.array(
                [self._episode_timestamp0 + grid_index / self.frequency], dtype=np.float64
            )
            output["sync.video_capture_sequence"] = np.array(
                [raw_sequence], dtype=np.int64
            )
            output["sync.video_capture_source_monotonic_timestamp"] = np.array(
                [raw_timestamp], dtype=np.float64
            )
            output["sync.camera_resampled"] = np.array(
                [0.0 if is_current_capture else 1.0], dtype=np.float32
            )
            output["sync.camera_source_gap_frames"] = np.array(
                [0 if is_current_capture else 1], dtype=np.int64
            )
            output["sync.camera_source_skipped_captures"] = np.array(
                [self._pending_camera_skipped_captures if is_current_capture else 0],
                dtype=np.int64,
            )
            self.data_exporter.add_frame(output)
            self._last_resampled_frame = copy.deepcopy(output)

        self._pending_camera_skipped_captures = 0
        self._last_resampled_grid_index = target_grid_index

    def _add_sync_features(self, frame_data: dict, proprio: dict) -> None:
        assert self.latest_image_msg is not None
        state_sequence = self._scalar_int(proprio.get("index"))
        state_source_ts = self._scalar_float(proprio.get("state_monotonic_timestamp"))
        state_receive_ts = self._scalar_float(
            proprio.get("receive_monotonic_timestamp")
        )
        camera_sequence = self._scalar_int(self.latest_image_msg.get("sequence"))
        camera_source_ts = self._scalar_float(
            self.latest_image_msg.get("server_monotonic_timestamp")
        )
        camera_receive_ts = self._scalar_float(
            self.latest_image_msg.get("receive_monotonic_timestamp")
        )
        reference_sequence = self._scalar_int(
            proprio.get("reference_source_frame")
        )
        reference_source_ts = self._scalar_float(
            proprio.get("reference_monotonic_timestamp")
        )
        smpl_sequence = -1
        smpl_source_ts = 0.0
        smpl_receive_ts = 0.0
        if self.latest_sonic_msg is not None:
            smpl_sequence = self._scalar_int(self.latest_sonic_msg.get("frame_index"))
            smpl_source_ts = self._scalar_float(
                self.latest_sonic_msg.get("source_monotonic_timestamp")
            )
            smpl_receive_ts = self._scalar_float(
                self.latest_sonic_msg.get("receive_monotonic_timestamp")
            )
        if self._episode_timestamp0 is None:
            self._episode_timestamp0 = camera_source_ts

        frame_data.update(
            {
                # This is the only timestamp used for LeRobot `timestamp`.
                # It is the camera server's monotonic capture/output clock,
                # not the collector loop clock and not a producer timestamp.
                "timestamp": np.array(
                    [
                        camera_source_ts
                        - (self._episode_timestamp0
                           if self._episode_timestamp0 is not None
                           else camera_source_ts)
                    ],
                    dtype=np.float32,
                ),
                "sync.state_sequence": np.array([state_sequence], dtype=np.int64),
                # Receive timestamps describe skew only. Source timestamps are
                # retained separately; `timestamp` above is the camera clock.
                "sync.state_monotonic_timestamp": np.array(
                    [state_receive_ts], dtype=np.float64
                ),
                "sync.reference_sequence": np.array(
                    [reference_sequence], dtype=np.int64
                ),
                "sync.reference_monotonic_timestamp": np.array(
                    [state_receive_ts], dtype=np.float64
                ),
                "sync.control_sequence": np.array([state_sequence], dtype=np.int64),
                "sync.control_monotonic_timestamp": np.array(
                    [state_receive_ts], dtype=np.float64
                ),
                "sync.video_sequence": np.array([camera_sequence], dtype=np.int64),
                "sync.video_monotonic_timestamp": np.array(
                    [camera_receive_ts], dtype=np.float64
                ),
                "sync.smpl_sequence": np.array([smpl_sequence], dtype=np.int64),
                "sync.smpl_monotonic_timestamp": np.array(
                    [smpl_receive_ts], dtype=np.float64
                ),
                "sync.state_video_receive_skew_ms": np.array(
                    [1000.0 * (state_receive_ts - camera_receive_ts)],
                    dtype=np.float64,
                ),
                "sync.smpl_state_receive_skew_ms": np.array(
                    [1000.0 * (smpl_receive_ts - state_receive_ts)],
                    dtype=np.float64,
                ),
                "sync.reference_video_receive_skew_ms": np.array(
                    [1000.0 * (state_receive_ts - camera_receive_ts)],
                    dtype=np.float64,
                ),
                "sync.control_video_receive_skew_ms": np.array(
                    [1000.0 * (state_receive_ts - camera_receive_ts)],
                    dtype=np.float64,
                ),
                "sync.state_source_monotonic_timestamp": np.array(
                    [state_source_ts], dtype=np.float64
                ),
                "sync.reference_source_monotonic_timestamp": np.array(
                    [reference_source_ts], dtype=np.float64
                ),
                "sync.video_source_monotonic_timestamp": np.array(
                    [camera_source_ts], dtype=np.float64
                ),
                "sync.smpl_source_monotonic_timestamp": np.array(
                    [smpl_source_ts], dtype=np.float64
                ),
            }
        )

    @staticmethod
    def _validate_core_frame(frame_data: dict) -> None:
        required_finite = (
            "observation.joint_q",
            "observation.joint_dq",
            "observation.root_q_relative",
            "action.joint_q",
            "action.root_p",
            "action.root_q",
            "action.hand_binary",
            "action.hand_q",
            "control.joint_q_target",
            "audit.gmr_joint_q_raw",
            "audit.gmr_root_p_raw",
            "audit.gmr_root_q_raw",
            "audit.g1_joint_q_measured_raw",
            "audit.g1_joint_dq_measured_raw",
            "audit.g1_root_q_measured_raw",
        )
        for key in required_finite:
            values = np.asarray(frame_data[key], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{key} contains NaN or Inf")

        reference_q = np.asarray(frame_data["action.joint_q"], dtype=np.float64)
        if reference_q.shape != (29,) or np.linalg.norm(reference_q) < _MIN_REFERENCE_JOINT_NORM_RAD:
            raise ValueError("action.joint_q is missing a valid full-body GMR q29 reference")
        raw_reference_q = np.asarray(frame_data["audit.gmr_joint_q_raw"], dtype=np.float64)
        if raw_reference_q.shape != (29,) or not np.allclose(
            reference_q, raw_reference_q, rtol=0.0, atol=1e-9
        ):
            raise ValueError("action.joint_q does not exactly match raw GMR qpos[7:]")

        for key in ("observation.root_q_relative", "action.root_q"):
            quat = np.asarray(frame_data[key], dtype=np.float64)
            if quat.shape != (4,) or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5):
                raise ValueError(f"{key} must be a normalized WXYZ quaternion")

        if float(np.asarray(frame_data["action.root_target_valid"]).reshape(-1)[0]) < 0.5:
            raise ValueError("GMR root reference is invalid")
        if float(
            np.asarray(frame_data["action.root_target_discontinuous"]).reshape(-1)[0]
        ) >= 0.5:
            raise ValueError("GMR root reference jumped discontinuously")

        hand_binary = np.asarray(frame_data["action.hand_binary"], dtype=np.float64)
        if hand_binary.shape != (2,) or not np.all(np.isin(hand_binary, [0.0, 1.0])):
            raise ValueError("action.hand_binary must contain two binary values")
        hand_q = np.asarray(frame_data["action.hand_q"], dtype=np.float64)
        if hand_q.shape != (14,) or not np.all(np.isfinite(hand_q)):
            raise ValueError("action.hand_q must contain 14 finite raw hand joint targets")

    def _extract_body_joint_configuration(self, configuration: np.ndarray) -> np.ndarray:
        body_joint_q = np.asarray(
            self.robot_model.get_body_actuated_joints(configuration), dtype=np.float64
        )
        if body_joint_q.shape != (29,):
            raise ValueError(
                f"Expected 29 body joint values, got shape {body_joint_q.shape}"
            )
        return body_joint_q.copy()

    @staticmethod
    def _has_valid_reference_joint_target(proprio: dict) -> bool:
        target = np.asarray(proprio.get("body_q_target", []), dtype=np.float64).reshape(-1)
        raw_target = np.asarray(
            proprio.get("gmr_joint_q_raw", []), dtype=np.float64
        ).reshape(-1)
        return bool(
            target.shape == (29,)
            and raw_target.shape == (29,)
            and np.all(np.isfinite(target))
            and np.all(np.isfinite(raw_target))
            and np.linalg.norm(target) >= _MIN_REFERENCE_JOINT_NORM_RAD
            and np.allclose(target, raw_target, rtol=0.0, atol=1e-9)
        )

    @staticmethod
    def _reference_target_is_gmr(proprio: dict) -> bool:
        value = np.asarray(
            proprio.get("body_q_target_is_gmr", []), dtype=np.float64
        ).reshape(-1)
        return bool(value.shape == (1,) and np.isfinite(value[0]) and value[0] >= 0.5)

    @classmethod
    def _extract_reference_joint_target(cls, proprio: dict) -> np.ndarray:
        if not cls._has_valid_reference_joint_target(proprio):
            raise ValueError(
                "body_q_target must contain 29 finite Sonic reference joint positions"
            )
        return np.asarray(proprio["body_q_target"], dtype=np.float64).reshape(29).copy()

    @staticmethod
    def _hand_binary_from_joint_positions(
        left_hand_q: np.ndarray, right_hand_q: np.ndarray
    ) -> np.ndarray:
        hand_binary = []
        for side, values in (("left", left_hand_q), ("right", right_hand_q)):
            joints = np.asarray(values, dtype=np.float64).reshape(-1)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                raise ValueError(
                    f"Expected 7 finite measured {side} hand joints, got {joints.shape}"
                )
            hand_binary.append(
                float(np.max(np.abs(joints)) >= _HAND_CLOSED_JOINT_THRESHOLD_RAD)
            )
        return np.asarray(hand_binary, dtype=np.float32)

    @staticmethod
    def _extract_body_joint_velocity(proprio: dict) -> np.ndarray:
        body_joint_dq = np.asarray(proprio.get("body_dq", []), dtype=np.float64).reshape(-1)
        if body_joint_dq.shape != (29,):
            raise ValueError(
                f"Expected 29 measured body joint velocities, got shape {body_joint_dq.shape}"
            )
        if not np.all(np.isfinite(body_joint_dq)):
            raise ValueError("Measured body joint velocities contain NaN or Inf")
        return body_joint_dq.copy()

    def _add_cpp_state_features(self, frame_data: dict, proprio: dict) -> None:
        if "base_quat" in proprio:
            base_quat = np.asarray(proprio["base_quat"], dtype=np.float64)
            if base_quat.shape != (4,) or not np.all(np.isfinite(base_quat)):
                raise ValueError("Measured IMU base_quat is invalid")
            norm = np.linalg.norm(base_quat)
            if not np.isclose(norm, 1.0, atol=1e-4):
                raise ValueError("Measured IMU base_quat is not normalized")
            frame_data["observation.root_orientation"] = base_quat / norm
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)

            if "init_ref_data_root_rot_array" in proprio:
                frame_data["observation.cpp_rotation_offset"] = np.asarray(
                    proprio["init_ref_data_root_rot_array"], dtype=np.float64
                )
            else:
                frame_data["observation.cpp_rotation_offset"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
        else:
            raise ValueError("Measured IMU base_quat is missing")

        if "init_base_quat" in proprio:
            frame_data["observation.init_base_quat"] = np.asarray(
                proprio["init_base_quat"], dtype=np.float64
            )
        else:
            frame_data["observation.init_base_quat"] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        if "delta_heading" in proprio:
            dh = proprio["delta_heading"]
            if isinstance(dh, np.ndarray):
                dh = dh.item() if dh.size == 1 else dh[0]
            frame_data["teleop.delta_heading"] = np.array([float(dh)], dtype=np.float64)
        else:
            frame_data["teleop.delta_heading"] = np.zeros(1, dtype=np.float64)

        if "token_state" in proprio:
            frame_data["action.motion_token"] = np.asarray(proprio["token_state"], dtype=np.float64)
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _add_sonic_pose_features(self, frame_data: dict) -> float | None:
        """Add teleop features based on current stream mode."""
        sonic_latency_ms = None

        frame_data["teleop.stream_mode"] = np.array([self.current_stream_mode], dtype=np.int32)

        smpl_msg = self.latest_sonic_msg
        use_smpl = False
        if self.current_stream_mode in (1, 4) and smpl_msg is not None:
            receive_ts = smpl_msg.get("receive_monotonic_timestamp")
            if receive_ts is not None:
                age_sec = time.monotonic() - receive_ts
                sonic_latency_ms = age_sec * 1000
                self.sonic_timing_monitor.log_time_delta(age_sec)
                if sonic_latency_ms <= 100.0:
                    use_smpl = True
                elif (self.sonic_timing_monitor.failure_count + 1) % 10 == 0:
                    self._print_and_say(
                        f"Sonic pose stale ({sonic_latency_ms:.1f}ms old)",
                        say=False,
                    )
            else:
                use_smpl = True

        if self.current_stream_mode in (1, 4) and not use_smpl:
            raise ValueError("SMPL/GMR source is stale or invalid")

        planner_msg = self.latest_planner_msg
        use_planner = False
        if self.current_stream_mode == 5 and planner_msg is not None:
            receive_ts = planner_msg.get("receive_timestamp")
            if receive_ts is not None:
                age_sec = time.time() - receive_ts
                planner_latency_ms = age_sec * 1000
                if sonic_latency_ms is None:
                    sonic_latency_ms = planner_latency_ms
                if planner_latency_ms <= 200.0:
                    use_planner = True
            else:
                use_planner = True

        # SMPL features
        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            if joints.ndim == 2:
                joints = joints.flatten()
            frame_data["teleop.smpl_joints"] = np.ascontiguousarray(joints, dtype=np.float32)
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            if pose.ndim > 1:
                pose = pose.flatten()
            frame_data["teleop.smpl_pose"] = np.ascontiguousarray(pose, dtype=np.float32)
        else:
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        if use_smpl and smpl_msg.get("body_quat_w") is not None:
            body_quat_w = smpl_msg["body_quat_w"].astype(np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w, frame_data
            )
        else:
            frame_data["teleop.body_quat_w"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            )

        frame_data["teleop.left_wrist_joints"] = (
            smpl_msg["left_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("left_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.right_wrist_joints"] = (
            smpl_msg["right_wrist_joints"].astype(np.float32)
            if use_smpl and smpl_msg.get("right_wrist_joints") is not None
            else np.zeros(3, dtype=np.float32)
        )

        frame_data["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = (
            smpl_msg if self.current_stream_mode in (1, 4) and smpl_msg is not None
            else planner_msg if planner_msg is not None
            else smpl_msg
        )
        frame_data["teleop.left_hand_joints"] = (
            hand_msg["left_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("left_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )
        frame_data["teleop.right_hand_joints"] = (
            hand_msg["right_hand_joints"].astype(np.float32)
            if hand_msg is not None
            and hand_msg.get("right_hand_joints") is not None
            else np.zeros(7, dtype=np.float32)
        )
        hand_binary = []
        for side in ("left", "right"):
            trigger = hand_msg.get(f"{side}_trigger") if hand_msg is not None else None
            joints = frame_data[f"teleop.{side}_hand_joints"]
            if trigger is not None and np.isfinite(trigger):
                closed = float(trigger >= 0.5)
            else:
                closed = float(
                    np.max(np.abs(joints)) >= _HAND_CLOSED_JOINT_THRESHOLD_RAD
                )
            hand_binary.append(closed)
        frame_data["action.hand_binary"] = np.asarray(hand_binary, dtype=np.float32)

        # Planner command fields
        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg.get("planner_movement") is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg.get("planner_facing") is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner else [-1.0],
            dtype=np.float32,
        )

        # VR 3-point pose
        frame_data["teleop.vr_3pt_position"] = (
            planner_msg["vr_3pt_position"].astype(np.float32)
            if use_planner and planner_msg.get("vr_3pt_position") is not None
            else np.zeros(9, dtype=np.float32)
        )
        if use_planner and planner_msg.get("vr_3pt_orientation") is not None:
            frame_data["teleop.vr_3pt_orientation"] = quat_to_rot6d(
                planner_msg["vr_3pt_orientation"].astype(np.float32)
            )
        else:
            frame_data["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return sonic_latency_ms

    def _compute_target_body_orientation(
        self, body_quat_w: np.ndarray, frame_data: dict
    ) -> np.ndarray:
        """Compute yaw-normalised target body orientation as rot6d (6-dim)."""
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])

        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot

        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return quat_to_rot6d(target_quat)

    def save_and_cleanup(self):
        try:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                episode_index = self.current_episode_index
                output_path = Path(self.data_exporter.meta.root).resolve()
                self._print_and_say(
                    f"Saving buffered episode {episode_index}: {buffer_size} frames",
                    say=False,
                    blocking=False,
                )
                self.data_exporter.save_episode()
                self._print_and_say(
                    f"Finished saving episode {episode_index}: {buffer_size} frames "
                    f"-> {output_path}",
                    say=False,
                    blocking=False,
                )
            self._print_and_say(
                f"Recording complete: {Path(self.data_exporter.meta.root).resolve()}",
                say=False,
                blocking=True,
            )
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}", blocking=True)

        if self.data_exporter.episode_buffer.get("size", 0) == 0:
            for writer in self.data_exporter.video_writers.values():
                try:
                    writer.cancel()
                except Exception:
                    pass

        try:
            self._state_subscriber.close()
        except Exception:
            pass
        for sock in [self._sonic_zmq_socket]:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        for ctx in [self._sonic_zmq_ctx]:
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass

        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        try:
            while True:
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_zmq()

                    with self.telemetry.timer("poll_sonic"):
                        self._poll_sonic_zmq_messages()

                    with self.telemetry.timer("poll_image"):
                        img_msg = self._image_subscriber.read()
                        if img_msg is not None:
                            self.latest_image_msg = img_msg

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    end_time = time.monotonic()

                elapsed = time.monotonic() - t_start
                sleep_time = self.loop_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if (end_time - t_start) > self.loop_period:
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _dataset_has_saved_episodes(dataset_root: Path) -> bool:
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not episodes_path.is_file():
        return False
    return any(line.strip() for line in episodes_path.read_text(encoding="utf-8").splitlines())


def _default_humanoid_arena_output_root(dataset_root: Path) -> Path:
    return dataset_root.with_name(f"{dataset_root.name}_humanoidarena_v3_1_aligned")


def _post_convert_to_humanoid_arena(config: SonicDataExporterConfig, dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    if not _dataset_has_saved_episodes(dataset_root):
        print(f"[HumanoidArena] Skipping conversion: no saved episodes in {dataset_root}")
        return

    if config.humanoid_arena_output_root:
        output_root = Path(config.humanoid_arena_output_root).expanduser()
    else:
        output_root = _default_humanoid_arena_output_root(dataset_root)

    from gear_sonic.scripts.convert_to_humanoidarena_v3_1 import convert_datasets

    if config.humanoid_arena_task_name is None:
        raise ValueError(
            "--humanoid-arena-task-name is required when HumanoidArena conversion is enabled"
        )

    print(f"[HumanoidArena] Converting {dataset_root} -> {output_root}")
    summary = convert_datasets(
        [dataset_root],
        output_root,
        task_name=config.humanoid_arena_task_name,
        task_prompt=config.task_prompt,
        root_height=config.humanoid_arena_root_height,
        root_z_mode=config.humanoid_arena_root_z_mode,
        root_position_axis_order=config.humanoid_arena_root_position_axis_order,
        video_mode=config.humanoid_arena_video_mode,
        overwrite=config.humanoid_arena_overwrite,
    )
    print("[HumanoidArena] Conversion and semantic validation complete:")
    print(json.dumps(summary, indent=2))


def main(config: SonicDataExporterConfig):
    if config.data_collection_frequency != 30:
        raise ValueError(
            "Real G1 VLA collection requires --data-collection-frequency 30 so "
            "one camera frame maps to one dataset row"
        )
    if config.convert_to_humanoid_arena and config.humanoid_arena_task_name is None:
        raise ValueError(
            "Select the episode's real task with --humanoid-arena-task-name. "
            "Do not silently label unrelated demonstrations as HSI_vision_navi."
        )

    g1_rm = get_g1_robot_model()

    dataset_features = get_features_sonic_vla(g1_rm)
    modality_config = get_modality_config_sonic_vla(g1_rm)

    if config.record_wrist_cameras:
        print("[Camera] Wrist cameras enabled — adding to dataset schema")
        dataset_features.update(get_wrist_camera_features())
        wrist_modality = get_wrist_camera_modality_config()
        for key, value in wrist_modality.items():
            if key in modality_config:
                modality_config[key].update(value)
            else:
                modality_config[key] = value

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    robot_config = poll_robot_config_zmq(
        config.state_zmq_host, config.state_zmq_port, config.robot_config_timeout
    )

    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        script_config={
            **robot_config,
            "record_wrist_cameras": config.record_wrist_cameras,
            "action_joint_q_semantics": "sonic_reference_joint_position_mujoco_29",
            "action_root_semantics": "gmr_qpos_root_relative_to_episode_first_frame",
            "control_joint_q_target_semantics": "sonic_policy_pd_position_target_mujoco_29",
            "recording_alignment": "fixed_30hz_data_clock_with_camera_only_hold",
            "authoritative_timebase": "collector.fixed_30hz_grid",
            "gmr_root_semantics": "GMR qpos root with offset_to_ground=True; qpos root is pelvis position in meters",
            "gmr_root_frame_verification": {
                "model": "GMR/assets/unitree_g1/g1_mocap_29dof.xml",
                "qpos_layout": "qpos[:3]=pelvis position [m], qpos[3:7]=pelvis WXYZ quaternion, qpos[7:]=29 DoF",
                "ground_height_m": 0.0,
                "retarget_ground_offset": "lowest retargeted human point + 0.10 m",
                "round_trip_required": True,
            },
        },
    )

    data_collector = GrootDataCollector(
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        robot_model=g1_rm,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
        sonic_data_zmq_host=config.sonic_zmq_host,
        sonic_data_zmq_port=config.sonic_zmq_port,
        state_zmq_host=config.state_zmq_host,
        state_zmq_port=config.state_zmq_port,
    )
    data_collector.run()
    if not validate_dataset(Path(data_exporter.meta.root)):
        raise RuntimeError(
            "G1 VLA dataset preflight failed; do not use this dataset for training"
        )
    if config.convert_to_humanoid_arena:
        _post_convert_to_humanoid_arena(config, Path(data_exporter.meta.root))


if __name__ == "__main__":
    config = tyro.cli(SonicDataExporterConfig)

    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    main(config)
