"""Runtime bridge between the Kimodo 40D action schema and SONIC Protocol v1.

The SONIC repository intentionally remains an external runtime.  This module
only owns the boundary between the two projects:

* SONIC ``g1_debug`` state (MuJoCo order) -> Kimodo 64D state (canonical /
  IsaacLab order),
* Kimodo 40D semantic reference action -> SONIC streamed joint reference,
* deterministic packed KP reference-stream messages (SONIC Protocol v1).

No hardware command is issued by this module.  The caller decides whether to
publish the returned packet to the SONIC C++ process.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation


# SONIC's C++ output publishes body_q/body_dq in MuJoCo order.  The source and
# target arrays are the official mapping in policy_parameters.hpp.  Kimodo's
# canonical 29D order is the order consumed by SONIC's MotionSequence. The
# SONIC output interface applies its IsaacLab-to-MuJoCo mapping when building
# hardware targets, so this boundary must not permute the action a second time.
ISAACLAB_TO_MUJOCO = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

SONIC_PROTOCOL_VERSION = 1
# SONIC Protocol v1 uses a fixed 4096-byte JSON header.  Keep this value
# explicit here instead of relying on the size of the current field list: the
# C++ subscriber always advances the payload offset by exactly 4096 bytes.
SONIC_HEADER_SIZE = 4096
KIMODO_STATE_DIM = 64
KIMODO_ACTION_DIM = 40

# Keep this projection identical to ``data.realworld_loader``.  The training
# target is the normalized projection of the Dex3 middle-grip close pose:
# zero is open and one is fully closed.
REALWORLD_LEFT_HAND_CLOSE = np.asarray(
    [0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32
)
REALWORLD_RIGHT_HAND_CLOSE = -REALWORLD_LEFT_HAND_CLOSE
REALWORLD_HAND_CLOSE_POSES = np.stack(
    (REALWORLD_LEFT_HAND_CLOSE, REALWORLD_RIGHT_HAND_CLOSE), axis=0
)


class BridgeSafetyError(ValueError):
    """Raised when a state/action would be unsafe or ambiguous to publish."""


@dataclass(frozen=True)
class SafetyLimits:
    min_root_z: float = 0.50
    max_root_z: float = 1.20
    max_joint_abs: float = 4.0
    max_joint_step: float = 0.75
    max_joint_velocity: float = 35.0
    max_root_xy_step: float = 0.15
    max_root_z_step: float = 0.12
    max_root_xy_abs: float = 10.0
    max_reference_age_s: float = 0.25


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise BridgeSafetyError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise BridgeSafetyError(f"{name} contains NaN or Inf")
    return arr.copy()


def normalize_quat_wxyz(quat: Any, name: str = "quaternion") -> np.ndarray:
    q = _finite_array(quat, (4,), name)
    norm = float(np.linalg.norm(q))
    if norm < 1e-7:
        raise BridgeSafetyError(f"{name} has near-zero norm")
    return q / norm


def optional_scalar(value: Any, name: str) -> float | None:
    """Decode a finite scalar that may arrive as a Python/NumPy scalar/list."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        raise BridgeSafetyError(f"{name} must be one finite scalar")
    return float(arr[0])


def measured_hand_closure_from_state(state_msg: dict[str, Any]) -> np.ndarray:
    """Project measured SONIC Dex3 joints to the training closure convention."""
    if not isinstance(state_msg, dict):
        raise BridgeSafetyError("g1_debug state must be a dictionary")

    hand_values = []
    for side in ("left", "right"):
        value = state_msg.get(f"{side}_hand_q_measured")
        if value is None:
            value = state_msg.get(f"{side}_hand_q")
        if value is None:
            raise BridgeSafetyError(
                f"g1_debug is missing measured {side}_hand_q_measured/{side}_hand_q"
            )
        joints = _finite_array(value, (7,), f"g1_debug.{side}_hand_q")
        hand_values.append(joints)

    hand_q = np.stack(hand_values, axis=0)
    denominators = np.sum(REALWORLD_HAND_CLOSE_POSES**2, axis=1)
    closure = np.sum(hand_q * REALWORLD_HAND_CLOSE_POSES, axis=1) / denominators
    closure = np.clip(closure, 0.0, 1.0).astype(np.float32)
    if closure.shape != (2,) or not np.isfinite(closure).all():
        raise BridgeSafetyError("measured hand closure must be finite with shape (2,)")
    return closure


def quat_mul_wxyz(q1: Any, q2: Any) -> np.ndarray:
    a = normalize_quat_wxyz(q1, "q1")
    b = normalize_quat_wxyz(q2, "q2")
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return normalize_quat_wxyz(
        np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        ),
        "quaternion product",
    )


def quat_conjugate_wxyz(quat: Any) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_heading_wxyz(quat: Any) -> np.ndarray:
    """Return the yaw-only heading of a WXYZ quaternion."""
    q = normalize_quat_wxyz(quat)
    w, x, y, z = q
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)


def rot6d_row_to_quat_wxyz(rot6d: Any) -> np.ndarray:
    """Decode Kimodo's row-stacked first-two-columns 6D rotation."""
    flat = _finite_array(rot6d, (6,), "root rotation 6D")
    col0 = flat[[0, 2, 4]]
    col1 = flat[[1, 3, 5]]
    n0 = float(np.linalg.norm(col0))
    if n0 < 1e-7:
        raise BridgeSafetyError("root rotation 6D first column is degenerate")
    col0 = col0 / n0
    col1 = col1 - np.dot(col0, col1) * col0
    n1 = float(np.linalg.norm(col1))
    if n1 < 1e-7:
        raise BridgeSafetyError("root rotation 6D columns are collinear")
    col1 = col1 / n1
    col2 = np.cross(col0, col1)
    rot = np.stack((col0, col1, col2), axis=1)
    quat_xyzw = SciPyRotation.from_matrix(rot).as_quat().astype(np.float32)
    return normalize_quat_wxyz(quat_xyzw[[3, 0, 1, 2]], "root reference quaternion")


def _build_header(fields: list[dict[str, Any]], version: int = 1) -> bytes:
    header = {"v": int(version), "endian": "le", "count": 1, "fields": fields}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > SONIC_HEADER_SIZE:
        raise ValueError(f"SONIC message header is too large: {len(encoded)}")
    return encoded.ljust(SONIC_HEADER_SIZE, b"\x00")


def pack_pose_v1(
    *,
    joint_pos_isaaclab: Any,
    joint_vel_isaaclab: Any,
    body_quat_wxyz: Any,
    root_position: Any,
    frame_index: int,
    timestamp_monotonic: float | None = None,
    catch_up: bool = True,
    left_hand_joints: Any | None = None,
    right_hand_joints: Any | None = None,
) -> bytes:
    """Pack one SONIC Protocol-v1 pose frame.

    Kimodo's action codec emits the canonical order consumed by SONIC's
    MotionSequence. SONIC applies the hardware mapping downstream. The root
    quaternion is WXYZ and the root position is in the streamed reference
    frame (x-forward, y-left, z-up).
    """
    q = _finite_array(joint_pos_isaaclab, (29,), "joint_pos")
    dq = _finite_array(joint_vel_isaaclab, (29,), "joint_vel")
    quat = normalize_quat_wxyz(body_quat_wxyz, "body_quat")
    root = _finite_array(root_position, (3,), "root_position")
    fields: list[dict[str, Any]] = [
        {"name": "joint_pos", "dtype": "f32", "shape": [1, 29]},
        {"name": "joint_vel", "dtype": "f32", "shape": [1, 29]},
        {"name": "body_quat", "dtype": "f32", "shape": [1, 1, 4]},
        {"name": "root_position", "dtype": "f32", "shape": [1, 3]},
        {"name": "frame_index", "dtype": "i64", "shape": [1]},
        {"name": "catch_up", "dtype": "bool", "shape": [1]},
    ]
    arrays = [
        np.ascontiguousarray(q.reshape(1, 29), dtype=np.float32),
        np.ascontiguousarray(dq.reshape(1, 29), dtype=np.float32),
        np.ascontiguousarray(quat.reshape(1, 1, 4), dtype=np.float32),
        np.ascontiguousarray(root.reshape(1, 3), dtype=np.float32),
        np.asarray([int(frame_index)], dtype=np.int64),
        np.asarray([bool(catch_up)], dtype=np.bool_),
    ]
    if timestamp_monotonic is not None:
        fields.append({"name": "timestamp_monotonic", "dtype": "f64", "shape": [1]})
        arrays.append(np.asarray([float(timestamp_monotonic)], dtype=np.float64))
    for name, value in (("left_hand_joints", left_hand_joints), ("right_hand_joints", right_hand_joints)):
        if value is None:
            continue
        hand = _finite_array(value, (7,), name)
        fields.append({"name": name, "dtype": "f32", "shape": [1, 7]})
        arrays.append(np.ascontiguousarray(hand.reshape(1, 7), dtype=np.float32))
    payload = b"".join(arr.tobytes() for arr in arrays)
    return b"pose" + _build_header(fields, SONIC_PROTOCOL_VERSION) + payload


class SonicPoseWindow:
    """Send overlapping past frames and optional committed future references.

    The SONIC callback keeps only the latest received packet. A single-frame
    sender loses coverage when one packet is overwritten; SONIC then resets
    both playback AND heading. Five overlapping frames tolerate four skipped
    packets without inventing targets or modifying their original timestamps.
    """
    def __init__(self, size: int = 5, lookahead_frames: int = 0, rate: float = 50.0):
        if size < 1:
            raise ValueError("window size must be positive")
        if lookahead_frames < 0 or rate <= 0:
            raise ValueError("lookahead must be non-negative and rate positive")
        self.lookahead_frames = int(lookahead_frames)
        self.rate = float(rate)
        self.frames = deque(maxlen=size)

    def append(self, future_references=(), **pose) -> bytes | None:
        # Reuse single-frame validation and wire encoding before retaining it.
        packet = pack_pose_v1(**pose)
        header = json.loads(packet[4:4 + SONIC_HEADER_SIZE].rstrip(b"\x00"))
        offset = 4 + SONIC_HEADER_SIZE
        arrays = {}
        for field in header["fields"]:
            dtype = {"f32": "<f4", "f64": "<f8", "i64": "<i8", "bool": "?"}[field["dtype"]]
            count = int(np.prod(field["shape"]))
            array = np.frombuffer(packet, dtype=dtype, count=count, offset=offset).reshape(field["shape"])
            arrays[field["name"]] = array.copy()
            offset += array.nbytes
        if self.frames and int(arrays["frame_index"][0]) != int(self.frames[-1]["frame_index"][0]) + 1:
            raise ValueError("SONIC pose window requires consecutive frame indices")
        self.frames.append(arrays)
        if len(self.frames) < self.frames.maxlen:
            return None
        # Only already-admitted references are supplied by the caller. Unknown
        # future is endpoint hold, never extrapolated motion. Keep the packet's
        # end advancing every tick so the native merger does not reset heading.
        sequence = list(self.frames)
        future = list(future_references)[:self.lookahead_frames]
        last = arrays
        for index in range(self.lookahead_frames):
            item = {name: value.copy() for name, value in last.items()}
            if index < len(future):
                ref = future[index]
                item["joint_pos"] = _finite_array(ref["joint_pos"], (29,), "future joints")[None].astype('<f4')
                item["joint_vel"] = _finite_array(ref["joint_vel"], (29,), "future velocity")[None].astype('<f4')
                item["body_quat"] = normalize_quat_wxyz(ref["body_quat"])[None,None].astype('<f4')
                item["root_position"] = _finite_array(ref["root_position"], (3,), "future root")[None].astype('<f4')
            else:
                item["joint_vel"] = np.zeros_like(item["joint_vel"])
            item["frame_index"] = np.asarray([int(arrays["frame_index"][0])+index+1],dtype='<i8')
            if "timestamp_monotonic" in arrays:
                item["timestamp_monotonic"] = arrays["timestamp_monotonic"]+(index+1)/self.rate
            sequence.append(item)
            last = item
        per_frame = {"joint_pos", "joint_vel", "body_quat", "root_position", "frame_index"}
        fields, payload = [], []
        for field in header["fields"]:
            name = field["name"]
            array = np.concatenate([frame[name] for frame in sequence], axis=0) if name in per_frame else arrays[name]
            fields.append(dict(name=name, dtype=field["dtype"], shape=list(array.shape)))
            payload.append(array.tobytes())
        if all("timestamp_monotonic" in frame for frame in sequence):
            stamps = np.concatenate([frame["timestamp_monotonic"] for frame in sequence])
            fields.append(dict(name="reference_timestamp_monotonic", dtype="f64", shape=[len(stamps)]))
            payload.append(stamps.tobytes())
        return b"pose" + _build_header(fields, SONIC_PROTOCOL_VERSION) + b"".join(payload)


def state_msg_to_kimodo_state(
    state_msg: dict[str, Any],
    *,
    initial_root_quat_wxyz: np.ndarray | None,
    previous_state: tuple[np.ndarray, float] | None = None,
    now_monotonic: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Convert a SONIC ``g1_debug`` message to state and raw canonical values.

    Returns ``(state64, q29, dq29, root_quat, timestamp, initial_root_quat)``.
    The C++ publisher provides ``body_q``/``body_dq`` in MuJoCo order and
    ``base_quat`` in WXYZ.
    """
    if "body_q" not in state_msg or "base_quat" not in state_msg:
        raise BridgeSafetyError("g1_debug must contain body_q and base_quat")
    source_timestamp = optional_scalar(
        state_msg.get("state_monotonic_timestamp"),
        "g1_debug.state_monotonic_timestamp",
    )
    current_timestamp = float(
        source_timestamp
        if source_timestamp is not None and source_timestamp > 0.0
        else (now_monotonic if now_monotonic is not None else time.monotonic())
    )
    q_mj = _finite_array(state_msg["body_q"], (29,), "g1_debug.body_q")
    if "body_dq" in state_msg:
        dq_mj = _finite_array(state_msg["body_dq"], (29,), "g1_debug.body_dq")
    elif previous_state is not None:
        prev_q, prev_t = previous_state
        current_t = current_timestamp
        dt = max(current_t - float(prev_t), 1e-4)
        dq_mj = (q_mj - prev_q) / dt
    else:
        dq_mj = np.zeros(29, dtype=np.float32)
    q = q_mj[MUJOCO_TO_ISAACLAB]
    dq = dq_mj[MUJOCO_TO_ISAACLAB]
    root_quat = normalize_quat_wxyz(state_msg["base_quat"], "g1_debug.base_quat")
    if initial_root_quat_wxyz is None:
        initial_root_quat_wxyz = root_quat.copy()
    initial_root_quat_wxyz = normalize_quat_wxyz(initial_root_quat_wxyz, "initial root quaternion")
    relative = quat_mul_wxyz(quat_conjugate_wxyz(quat_heading_wxyz(initial_root_quat_wxyz)), root_quat)
    rel_xyzw = relative[[1, 2, 3, 0]]
    state_rot = SciPyRotation.from_quat(rel_xyzw).as_matrix().astype(np.float32)
    # Match the training/runtime row-stacked 6D convention.
    rot6d = state_rot[:, :2].reshape(6).astype(np.float32)
    state = np.concatenate((rot6d, q, dq), axis=0).astype(np.float32)
    if state.shape != (KIMODO_STATE_DIM,) or not np.isfinite(state).all():
        raise BridgeSafetyError("constructed Kimodo state is invalid")
    timestamp = current_timestamp
    return state, q, dq, root_quat, timestamp, initial_root_quat_wxyz


def reference_to_boundary_action(reference: dict[str, Any]) -> np.ndarray:
    """Encode a held/queued command boundary without reapplying a root delta."""
    action = np.zeros(KIMODO_ACTION_DIM, dtype=np.float32)
    action[2] = _finite_array(reference["root_position"], (3,), "root_position")[2]
    quat = normalize_quat_wxyz(reference["body_quat"], "boundary quaternion")
    rotation = SciPyRotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    action[3:9] = rotation[:, :2].reshape(6)
    action[9:38] = _finite_array(reference["joint_pos"], (29,), "joint_pos")
    hand = reference.get("hand_closure")
    if hand is None:
        hand = reference["hand_binary"]
    action[38:40] = _finite_array(hand, (2,), "hand closure")
    return action


class KimodoActionRuntime:
    """Decode 40D Kimodo actions into SONIC Protocol-v1 reference frames."""

    def __init__(self, limits: SafetyLimits | None = None, max_root_delta_deg: float = 26.0, reference_rate_hz: float = 50.0):
        self.limits = limits or SafetyLimits()
        self.max_root_delta_rad = np.deg2rad(float(max_root_delta_deg)) if max_root_delta_deg > 0 else None
        self.reference_rate_hz = max(float(reference_rate_hz), 1.0)
        self.reset()

    def reset(self) -> None:
        self.reference_root_position = np.zeros(3, dtype=np.float32)
        self.previous_reference_z: float | None = None
        self.previous_joint_q: np.ndarray | None = None
        # SONIC starts a streamed episode from the identity reference frame
        # and applies the measured robot heading on its side.  Seed the
        # quaternion limiter accordingly so the first model frame cannot
        # command an unbounded yaw (for example a 180-degree flip).
        self.previous_reference_quat: np.ndarray | None = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.frame_index = 0

    def prime_joints(self, measured_joint_q: Any) -> None:
        """Seed the jump limiter from the first measured canonical pose."""
        if self.previous_joint_q is None:
            self.previous_joint_q = _finite_array(
                measured_joint_q, (29,), "measured joint_pos"
            )

    def snapshot(self) -> tuple[np.ndarray, float | None, np.ndarray | None, np.ndarray | None, int]:
        """Capture mutable decoder state so a rejected action is transactional."""
        return (
            self.reference_root_position.copy(),
            self.previous_reference_z,
            None if self.previous_joint_q is None else self.previous_joint_q.copy(),
            None if self.previous_reference_quat is None else self.previous_reference_quat.copy(),
            self.frame_index,
        )

    def restore(self, snapshot) -> None:
        """Restore a state captured by :meth:`snapshot`."""
        root, previous_z, previous_joint, previous_quat, frame_index = snapshot
        self.reference_root_position = np.asarray(root, dtype=np.float32).copy()
        self.previous_reference_z = previous_z
        self.previous_joint_q = None if previous_joint is None else np.asarray(previous_joint, dtype=np.float32).copy()
        self.previous_reference_quat = None if previous_quat is None else np.asarray(previous_quat, dtype=np.float32).copy()
        self.frame_index = int(frame_index)

    def decode(self, action: Any, *, current_root_quat_wxyz: Any) -> dict[str, Any]:
        action = _finite_array(action, (KIMODO_ACTION_DIM,), "Kimodo action")
        # Validate the measured quaternion at the boundary.  SONIC applies
        # the initial-heading alignment to the streamed reference quaternion;
        # the bridge must not apply that heading a second time.
        normalize_quat_wxyz(current_root_quat_wxyz, "current root quaternion")
        root_z = float(action[2])
        if not self.limits.min_root_z <= root_z <= self.limits.max_root_z:
            raise BridgeSafetyError(f"action.root_z={root_z:.4f} outside [{self.limits.min_root_z}, {self.limits.max_root_z}]")
        q_ref = rot6d_row_to_quat_wxyz(action[3:9])
        # Enforce the same per-step root safety bound used by the sim runtime.
        if self.previous_reference_quat is not None and self.max_root_delta_rad is not None:
            rel = quat_mul_wxyz(quat_conjugate_wxyz(self.previous_reference_quat), q_ref)
            angle = 2.0 * np.arccos(np.clip(abs(float(rel[0])), -1.0, 1.0))
            if angle > self.max_root_delta_rad:
                # Slerp without importing the larger runtime module.
                dot = float(np.dot(self.previous_reference_quat, q_ref))
                q1 = q_ref if dot >= 0 else -q_ref
                dot = abs(dot)
                t = self.max_root_delta_rad / max(angle, 1e-8)
                if dot > 0.9995:
                    q_ref = normalize_quat_wxyz(self.previous_reference_quat + t * (q1 - self.previous_reference_quat))
                else:
                    theta = np.arccos(np.clip(dot, -1.0, 1.0))
                    q_ref = normalize_quat_wxyz(
                        (np.sin((1.0 - t) * theta) * self.previous_reference_quat + np.sin(t * theta) * q1)
                        / max(np.sin(theta), 1e-8)
                    )
        previous_z = root_z if self.previous_reference_z is None else self.previous_reference_z
        # Action XY is local to the reference root. Convert it to the streamed
        # MuJoCo reference frame before accumulating root_position.
        rot = SciPyRotation.from_quat(q_ref[[1, 2, 3, 0]]).as_matrix().astype(np.float32)
        local_delta = np.zeros(3, dtype=np.float32)
        local_delta[:2] = action[:2]
        dz = root_z - previous_z
        if abs(float(rot[2, 2])) > 1e-6:
            local_delta[2] = (dz - rot[2, 0] * local_delta[0] - rot[2, 1] * local_delta[1]) / rot[2, 2]
        world_delta = rot.dot(local_delta)
        if self.previous_reference_z is not None and abs(root_z - previous_z) > self.limits.max_root_z_step:
            raise BridgeSafetyError("action.root_z has an implausible one-frame jump")
        if np.linalg.norm(world_delta[:2]) > self.limits.max_root_xy_step:
            raise BridgeSafetyError("action.root_p has an implausible one-frame XY jump")
        self.reference_root_position[:2] += world_delta[:2]
        self.reference_root_position[2] = root_z
        if (self.limits.max_root_xy_abs > 0
                and np.linalg.norm(self.reference_root_position[:2]) > self.limits.max_root_xy_abs):
            raise BridgeSafetyError("action.root_p exceeded the configured XY workspace")
        q = action[9:38].copy()
        if np.max(np.abs(q)) > self.limits.max_joint_abs:
            raise BridgeSafetyError("action.joint_pos contains an implausible angle")
        if self.previous_joint_q is not None:
            joint_delta = np.abs(q - self.previous_joint_q)
            max_joint_delta = float(np.max(joint_delta))
            if max_joint_delta > self.limits.max_joint_step:
                joint_index = int(np.argmax(joint_delta))
                raise BridgeSafetyError(
                    "action.joint_pos has an implausible one-frame jump: "
                    f"joint_index={joint_index} "
                    f"previous={float(self.previous_joint_q[joint_index]):.5f} "
                    f"command={float(q[joint_index]):.5f} "
                    f"delta={max_joint_delta:.5f} "
                    f"limit={self.limits.max_joint_step:.5f}"
                )
        dq = np.zeros(29, dtype=np.float32) if self.previous_joint_q is None else (q - self.previous_joint_q) * self.reference_rate_hz
        if np.max(np.abs(dq)) > self.limits.max_joint_velocity:
            raise BridgeSafetyError("action.joint_pos implies an implausible joint velocity")
        hand = np.clip(action[38:40], 0.0, 1.0).astype(np.float32)
        result = {
            "joint_pos": q,
            "joint_vel": dq,
            "body_quat": q_ref,
            "root_position": self.reference_root_position.copy(),
            "hand_closure": hand,
            "hand_binary": hand,
            "frame_index": self.frame_index,
        }
        self.previous_reference_z = root_z
        self.previous_reference_quat = q_ref.copy()
        self.previous_joint_q = q.copy()
        self.frame_index += 1
        return result
