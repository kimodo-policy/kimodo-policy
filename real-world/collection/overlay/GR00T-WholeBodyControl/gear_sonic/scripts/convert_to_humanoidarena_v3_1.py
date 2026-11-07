#!/usr/bin/env python3
"""Convert collected Sonic LeRobot episodes to HumanoidArena V3.1 layout.

The Kimodo HumanoidArena loader expects a LeRobot-style dataset with:

* ``observation.state``: 64 float32 values
* ``action``: 40 float32 values
* video key ``observation.images.front``
* parquet episode metadata under ``meta/episodes/``

This converter maps the richer real-robot collection export into that schema
without modifying the source dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


ARENA_SCHEMA = "unitree_g1_gmt_refpose_v3_1"
ARENA_VERSION = "3.1"
VIDEO_KEY = "observation.images.front"
DEFAULT_ROOT_HEIGHT_M = 0.793
MIN_PLAUSIBLE_ROOT_HEIGHT_M = 0.2
MAX_PLAUSIBLE_ROOT_HEIGHT_M = 1.5
MAX_REFERENCE_ROOT_STEP_M = 0.2
MAX_REFERENCE_ROOT_ROTATION_STEP_DEGREES = 30.0

ARENA_TASKS: dict[str, tuple[int, str, str]] = {
    "HOI_double_desk": (
        0,
        "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
        "Put the hammer from the right table into the basket on the left table.",
    ),
    "HOI_football": (
        1,
        "Isaac-Move-Football-Single-G129-Dex3-Wholebody",
        "Kick the football into the goal.",
    ),
    "HOI_grap_cup": (
        2,
        "Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody",
        "Place the orange drink can on the table into the basket on the tea table.",
    ),
    "HOI_pp_box": (
        3,
        "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        "Move the box from the table onto the shelf.",
    ),
    "HSI_boxing": (
        4,
        "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
        "Strike the green markers on the punching bag.",
    ),
    "HSI_open_door": (
        5,
        "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        "Open the door.",
    ),
    "HSI_sit_sofa": (
        6,
        "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
        "Sit on the sofa.",
    ),
    "HSI_vision_navi": (
        7,
        "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
        "Avoid obstacles and move to the yellow marked area.",
    ),
}

ARENA_CONTENT_PROFILES = {
    "HOI_double_desk": {
        "episode_length": (678, 976.5, 1703),
        "root_delta_q50": 0.00519,
        "root_delta_q99": 0.03801,
        "root_z_q01_q50_q99": (0.83700, 0.88022, 0.90922),
        "hand_closed_fraction": (0.00037, 0.53184),
        "joint_velocity_abs_q99": 5.63125,
    },
    "HOI_football": {
        "episode_length": (252, 482.0, 1578),
        "root_delta_q50": 0.00746,
        "root_delta_q99": 0.03634,
        "root_z_q01_q50_q99": (0.82627, 0.87154, 0.89162),
        "hand_closed_fraction": (0.0, 0.00508),
        "joint_velocity_abs_q99": 4.17337,
    },
    "HOI_grap_cup": {
        "episode_length": (948, 1221.5, 2663),
        "root_delta_q50": 0.01207,
        "root_delta_q99": 0.03937,
        "root_z_q01_q50_q99": (0.84616, 0.87680, 0.89244),
        "hand_closed_fraction": (0.04741, 0.67236),
        "joint_velocity_abs_q99": 4.66971,
    },
    "HOI_pp_box": {
        "episode_length": (380, 638.5, 1917),
        "root_delta_q50": 0.00632,
        "root_delta_q99": 0.04242,
        "root_z_q01_q50_q99": (0.80580, 0.87088, 0.88891),
        "hand_closed_fraction": (0.04681, 0.01800),
        "joint_velocity_abs_q99": 5.14731,
    },
    "HSI_boxing": {
        "episode_length": (271, 480.5, 1564),
        "root_delta_q50": 0.00789,
        "root_delta_q99": 0.04745,
        "root_z_q01_q50_q99": (0.83009, 0.87155, 0.89028),
        "hand_closed_fraction": (0.94266, 0.98840),
        "joint_velocity_abs_q99": 4.59586,
    },
    "HSI_open_door": {
        "episode_length": (470, 773.0, 2069),
        "root_delta_q50": 0.00675,
        "root_delta_q99": 0.03640,
        "root_z_q01_q50_q99": (0.84812, 0.87944, 0.89486),
        "hand_closed_fraction": (0.01800, 0.03212),
        "joint_velocity_abs_q99": 4.51870,
    },
    "HSI_sit_sofa": {
        "episode_length": (543, 687.0, 1087),
        "root_delta_q50": 0.01010,
        "root_delta_q99": 0.04333,
        "root_z_q01_q50_q99": (0.47993, 0.87598, 0.89455),
        "hand_closed_fraction": (0.0, 0.32924),
        "joint_velocity_abs_q99": 4.57405,
    },
    "HSI_vision_navi": {
        "episode_length": (538, 961.5, 3343),
        "root_delta_q50": 0.00786,
        "root_delta_q99": 0.04785,
        "root_z_q01_q50_q99": (0.81165, 0.86674, 0.88920),
        "hand_closed_fraction": (0.65934, 0.94549),
        "joint_velocity_abs_q99": 5.18166,
    },
}

UNITREE_G1_JOINT_NAMES_29 = (
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
)

CANONICAL_G1_JOINT_NAMES_29 = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

STATE_NAMES_64 = (
    *(f"state.root_heading_canonical_rot6d.{i}" for i in range(6)),
    *(f"state.dof_pos.{name}" for name in CANONICAL_G1_JOINT_NAMES_29),
    *(f"state.dof_vel.{name}" for name in CANONICAL_G1_JOINT_NAMES_29),
)

ACTION_NAMES_40 = (
    "action.root_ref_base_local_xy_delta.x",
    "action.root_ref_base_local_xy_delta.y",
    "action.root_z",
    *(f"action.root_ref_rot6d.{i}" for i in range(6)),
    *(f"action.joint_pos.{name}" for name in CANONICAL_G1_JOINT_NAMES_29),
    "action.hand_binary.left",
    "action.hand_binary.right",
)

LEFT_HAND_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
RIGHT_HAND_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix(table: pa.Table, name: str, width: int) -> np.ndarray:
    column = table[name].combine_chunks()
    if column.null_count:
        raise ValueError(f"{name} has {column.null_count} null values")
    if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
        values = column.values.to_numpy(zero_copy_only=False)
    elif width == 1:
        values = column.to_numpy(zero_copy_only=False)
    else:
        raise ValueError(f"{name} must be a list column with width {width}, got {column.type}")
    matrix = values.reshape(len(column), width)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def _reorder_by_name(values: np.ndarray, source_names: list[str], target_names: tuple[str, ...]) -> np.ndarray:
    index = [source_names.index(name) for name in target_names]
    return values[:, index]


def _normalise_quaternions(quaternions: np.ndarray, name: str) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"{name} must have shape (T, 4), got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{name} contains near-zero quaternions")
    return quaternions / norms


def _quat_wxyz_to_matrix(quaternions: np.ndarray, name: str) -> np.ndarray:
    quaternions = _normalise_quaternions(quaternions, name)
    return R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32)


def _validate_reference_root_trajectory(root_p: np.ndarray, root_q: np.ndarray) -> None:
    root_height = root_p[:, 2]
    if np.any(root_height < MIN_PLAUSIBLE_ROOT_HEIGHT_M) or np.any(
        root_height > MAX_PLAUSIBLE_ROOT_HEIGHT_M
    ):
        raise ValueError(
            "Reference root height is outside the plausible G1 range "
            f"[{MIN_PLAUSIBLE_ROOT_HEIGHT_M}, {MAX_PLAUSIBLE_ROOT_HEIGHT_M}] m: "
            f"min={root_height.min():.3f}, max={root_height.max():.3f}"
        )
    if root_p.shape[0] < 2:
        return

    position_steps = np.linalg.norm(np.diff(root_p, axis=0), axis=1)
    bad_position_steps = np.flatnonzero(position_steps > MAX_REFERENCE_ROOT_STEP_M)
    if bad_position_steps.size:
        raise ValueError(
            "Reference root position jumps exceed "
            f"{MAX_REFERENCE_ROOT_STEP_M} m/frame at transitions "
            f"{bad_position_steps[:10].tolist()}"
        )

    normalized_q = _normalise_quaternions(root_q, "action root quaternion")
    rotations = R.from_quat(normalized_q[:, [1, 2, 3, 0]])
    rotation_steps = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    bad_rotation_steps = np.flatnonzero(
        rotation_steps > MAX_REFERENCE_ROOT_ROTATION_STEP_DEGREES
    )
    if bad_rotation_steps.size:
        raise ValueError(
            "Reference root rotation jumps exceed "
            f"{MAX_REFERENCE_ROOT_ROTATION_STEP_DEGREES} deg/frame at transitions "
            f"{bad_rotation_steps[:10].tolist()}"
        )


def _rot6d_row_from_matrix(rot_mats: np.ndarray) -> np.ndarray:
    return rot_mats[:, :, :2].reshape(rot_mats.shape[0], 6).astype(np.float32)


def _matrix_from_rot6d_row(rot6d: np.ndarray) -> np.ndarray:
    flat = np.asarray(rot6d, dtype=np.float32).reshape(-1, 6)
    column0 = flat[:, [0, 2, 4]]
    column1 = flat[:, [1, 3, 5]]
    column0_norm = np.linalg.norm(column0, axis=1, keepdims=True)
    if np.any(column0_norm < 1e-8):
        raise ValueError("rot6d contains near-zero first columns")
    column0 = column0 / column0_norm
    column1 = column1 - (column0 * column1).sum(axis=1, keepdims=True) * column0
    column1_norm = np.linalg.norm(column1, axis=1, keepdims=True)
    if np.any(column1_norm < 1e-8):
        raise ValueError("rot6d contains near-zero second columns")
    column1 = column1 / column1_norm
    column2 = np.cross(column0, column1)
    return np.stack((column0, column1, column2), axis=-1).reshape(rot6d.shape[:-1] + (3, 3))


def _integrate_action_root_positions(action: np.ndarray) -> np.ndarray:
    """Reconstruct HumanoidArena root positions using the Kimodo decoder semantics."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 40:
        raise ValueError(f"Expected HumanoidArena action shape (T, 40), got {action.shape}")
    root_rot = _matrix_from_rot6d_row(action[:, 3:9])
    positions = np.zeros((action.shape[0], 3), dtype=np.float32)
    positions[:, 2] = action[:, 2]
    for frame_index in range(1, action.shape[0]):
        local_delta = np.zeros(3, dtype=np.float32)
        local_delta[:2] = action[frame_index, :2]
        rotation = root_rot[frame_index]
        delta_z = action[frame_index, 2] - action[frame_index - 1, 2]
        if abs(float(rotation[2, 2])) > 1e-6:
            local_delta[2] = (
                delta_z - rotation[2, 0] * local_delta[0] - rotation[2, 1] * local_delta[1]
            ) / rotation[2, 2]
        positions[frame_index, :2] = positions[frame_index - 1, :2] + (rotation @ local_delta)[:2]
    return positions


def _joint_velocity(joint_q: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros_like(joint_q, dtype=np.float32)
    if joint_q.shape[0] > 1:
        velocity[1:] = (joint_q[1:] - joint_q[:-1]) * float(fps)
    return velocity


def _hand_binary_from_dex3_targets(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """Convert 7-DoF Dex3 target joints to HumanoidArena left/right binary hands."""
    output = np.zeros((left_hand.shape[0], 2), dtype=np.float32)
    for side, hand_values in enumerate((left_hand, right_hand)):
        hand_values = np.asarray(hand_values, dtype=np.float32)
        if hand_values.ndim != 2 or hand_values.shape[1] != 7:
            raise ValueError(f"Dex3 hand target must have shape (T, 7), got {hand_values.shape}")

        flexion = np.abs(hand_values[:, 1:7])
        low = np.quantile(flexion, 0.10, axis=0)
        high = np.quantile(flexion, 0.90, axis=0)
        span = high - low
        active = span >= 0.03
        if active.sum() < 2:
            continue

        score = ((flexion[:, active] - low[active]) / span[active]).clip(0.0, 1.0).mean(axis=1)
        closed = False
        for frame_index, value in enumerate(score):
            if closed:
                closed = value > 0.40
            else:
                closed = value >= 0.60
            output[frame_index, side] = float(closed)
    return output


def _fixed_size_list_array(matrix: np.ndarray, value_type: pa.DataType) -> pa.FixedSizeListArray:
    flat = pa.array(np.asarray(matrix).reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, matrix.shape[1])


def _stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _source_video_key(info: dict) -> str:
    for key, feature in info.get("features", {}).items():
        if key.startswith("observation.images.") and feature.get("dtype") == "video":
            return key
    raise ValueError("Source dataset has no video feature")


def _format_source_path(pattern: str, *, video_key: str, episode_index: int, chunks_size: int) -> Path:
    episode_chunk = episode_index // chunks_size
    return Path(
        pattern.format(
            video_key=video_key,
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
    )


def _copy_or_link_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "skip":
        return
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def _map_root_positions(root_p: np.ndarray, axis_order: str, root_z_mode: str, root_height: float) -> np.ndarray:
    if sorted(axis_order) != ["x", "y", "z"] or len(axis_order) != 3:
        raise ValueError("--root-position-axis-order must be a permutation of xyz")
    source_axis = {"x": 0, "y": 1, "z": 2}
    mapped = np.stack([root_p[:, source_axis[axis]] for axis in axis_order], axis=1).astype(np.float32)
    if root_z_mode == "constant":
        mapped[:, 2] = float(root_height)
    elif root_z_mode == "offset":
        mapped[:, 2] = mapped[:, 2] + float(root_height)
    elif root_z_mode == "relative":
        pass
    else:
        raise ValueError(f"Unsupported root z mode: {root_z_mode}")
    return mapped


def _source_root_positions_for_action(
    table: pa.Table,
    *,
    root_position_axis_order: str,
    root_z_mode: str,
    root_height: float,
) -> np.ndarray:
    root_p = _matrix(table, "action.root_p", 3).astype(np.float32)
    root_p = _map_root_positions(root_p, root_position_axis_order, root_z_mode, root_height)
    if "action.root_z" in table.column_names:
        root_p[:, 2] = _matrix(table, "action.root_z", 1).astype(np.float32)[:, 0]
    return root_p


def _build_state(
    table: pa.Table,
    info: dict,
    fps: float,
    *,
    strict_source_semantics: bool,
) -> np.ndarray:
    joint_names = list(info["features"]["observation.joint_q"]["names"])
    joint_q = _matrix(table, "observation.joint_q", 29).astype(np.float32)
    joint_q = _reorder_by_name(joint_q, joint_names, CANONICAL_G1_JOINT_NAMES_29)
    if "observation.joint_dq" in table.column_names:
        joint_dq = _matrix(table, "observation.joint_dq", 29).astype(np.float32)
        joint_dq_names = list(
            info["features"]["observation.joint_dq"].get("names", joint_names)
        )
        joint_dq = _reorder_by_name(
            joint_dq, joint_dq_names, CANONICAL_G1_JOINT_NAMES_29
        )
    elif strict_source_semantics:
        raise ValueError(
            "Source is missing observation.joint_dq. HumanoidArena state requires measured "
            "joint velocity; finite differences are allowed only with legacy compatibility."
        )
    else:
        joint_dq = _joint_velocity(joint_q, fps)
    if "observation.root_q_relative" in table.column_names:
        root_q = _matrix(table, "observation.root_q_relative", 4)
    else:
        root_q = _matrix(table, "observation.root_orientation", 4)
    root_rot6d = _rot6d_row_from_matrix(_quat_wxyz_to_matrix(root_q, "observation root quaternion"))
    state = np.concatenate((root_rot6d, joint_q, joint_dq), axis=1).astype(np.float32)
    if state.shape[1] != 64:
        raise RuntimeError(f"Internal error: expected state width 64, got {state.shape[1]}")
    return state


def _build_action(
    table: pa.Table,
    info: dict,
    *,
    root_position_axis_order: str,
    root_z_mode: str,
    root_height: float,
    strict_source_semantics: bool,
) -> np.ndarray:
    joint_names = list(info["features"]["action.joint_q"]["names"])
    joint_q = _matrix(table, "action.joint_q", 29).astype(np.float32)
    joint_q = _reorder_by_name(joint_q, joint_names, CANONICAL_G1_JOINT_NAMES_29)
    if strict_source_semantics and "action.root_z" not in table.column_names:
        raise ValueError(
            "Source is missing action.root_z. Ref-pose V3.1 requires the absolute reference "
            "root height instead of a silently injected constant."
        )
    if strict_source_semantics and "action.root_target_valid" not in table.column_names:
        raise ValueError(
            "Source is missing action.root_target_valid; target dropouts cannot be detected."
        )
    if "action.root_target_valid" in table.column_names:
        target_valid = _matrix(table, "action.root_target_valid", 1)[:, 0]
        invalid_count = int(np.count_nonzero(target_valid < 0.5))
        if invalid_count:
            raise ValueError(
                f"Episode contains {invalid_count} frames without a valid reference root target"
            )
    if strict_source_semantics and "action.root_target_discontinuous" not in table.column_names:
        raise ValueError(
            "Source is missing action.root_target_discontinuous; tracking resets cannot be detected."
        )
    if "action.root_target_discontinuous" in table.column_names:
        discontinuous = _matrix(table, "action.root_target_discontinuous", 1)[:, 0]
        reset_frames = np.flatnonzero(discontinuous >= 0.5)
        if reset_frames.size:
            raise ValueError(
                "Episode contains reference-root tracking resets at frames "
                f"{reset_frames[:10].tolist()}"
            )

    root_p = _source_root_positions_for_action(
        table,
        root_position_axis_order=root_position_axis_order,
        root_z_mode=root_z_mode,
        root_height=root_height,
    )
    root_q = _matrix(table, "action.root_q", 4)
    _validate_reference_root_trajectory(root_p, root_q)
    root_rot = _quat_wxyz_to_matrix(root_q, "action root quaternion")
    root_rot6d = _rot6d_row_from_matrix(root_rot)

    deltas = np.zeros_like(root_p, dtype=np.float32)
    if root_p.shape[0] > 1:
        deltas[1:] = root_p[1:] - root_p[:-1]
    local_deltas = np.einsum("tji,tj->ti", root_rot, deltas).astype(np.float32)

    if "action.hand_binary" in table.column_names:
        hand_binary = _matrix(table, "action.hand_binary", 2).astype(np.float32)
        if not np.all(np.isclose(hand_binary, 0.0) | np.isclose(hand_binary, 1.0)):
            raise ValueError("action.hand_binary must contain only 0/1 commands")
        hand_binary = np.rint(hand_binary).astype(np.float32)
    elif strict_source_semantics:
        raise ValueError(
            "Source is missing action.hand_binary. Per-episode quantile inference is not "
            "stable enough for HumanoidArena-compatible supervision."
        )
    else:
        state_names = list(info["features"]["action.wbc"]["names"])
        action_wbc = _matrix(table, "action.wbc", 43).astype(np.float32)
        left_hand = _reorder_by_name(action_wbc, state_names, LEFT_HAND_NAMES)
        right_hand = _reorder_by_name(action_wbc, state_names, RIGHT_HAND_NAMES)
        hand_binary = _hand_binary_from_dex3_targets(left_hand, right_hand)

    action = np.zeros((joint_q.shape[0], 40), dtype=np.float32)
    action[:, :2] = local_deltas[:, :2]
    action[:, 2] = root_p[:, 2]
    action[:, 3:9] = root_rot6d
    action[:, 9:38] = joint_q
    action[:, 38:40] = hand_binary
    return action


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _identity_quaternion_error(quaternion_wxyz: np.ndarray) -> float:
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quaternion_wxyz = _normalise_quaternions(quaternion_wxyz.reshape(1, 4), "first root quaternion")[0]
    return float(min(np.linalg.norm(quaternion_wxyz - identity), np.linalg.norm(quaternion_wxyz + identity)))


def _source_episode_paths(source_root: Path, info: dict) -> list[Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    source_video_key = _source_video_key(info)
    episodes = [
        json.loads(line)
        for line in (source_root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        source_root
        / _format_source_path(
            info["data_path"],
            video_key=source_video_key,
            episode_index=int(source_episode["episode_index"]),
            chunks_size=chunks_size,
        )
        for source_episode in episodes
    ]


def _source_episode_columns(source_data_path: Path, *, include_indices: bool) -> list[str]:
    columns = [
        "observation.joint_q",
        "observation.root_q_relative",
        "action.joint_q",
        "action.root_p",
        "action.root_q",
        "action.wbc",
    ]
    if include_indices:
        columns.extend(["timestamp", "frame_index"])
    available = set(pq.ParquetFile(source_data_path).schema_arrow.names)
    for optional_column in (
        "observation.joint_dq",
        "action.root_z",
        "action.root_target_valid",
        "action.root_target_discontinuous",
        "action.hand_binary",
    ):
        if optional_column in available:
            columns.append(optional_column)
    return columns


def validate_conversion(
    dataset_paths: list[Path],
    arena_root: Path,
    *,
    root_position_axis_order: str = "xyz",
    root_z_mode: str = "constant",
    root_height: float = DEFAULT_ROOT_HEIGHT_M,
    tolerance: float = 5e-5,
    strict_source_semantics: bool = True,
) -> dict:
    """Validate field-level semantics between source Sonic data and converted Arena data."""
    max_errors = {
        "observation_root_rot6d": 0.0,
        "observation_joint_q": 0.0,
        "observation_joint_dq": 0.0,
        "action_root_xy_position": 0.0,
        "action_root_z": 0.0,
        "action_root_rotation_matrix": 0.0,
        "action_joint_q": 0.0,
        "action_hand_binary": 0.0,
        "source_first_action_root_p_norm": 0.0,
    }
    diagnostics = {"source_first_action_root_q_identity": 0.0}
    total_frames = 0
    output_episode_index = 0
    hand_values: set[float] = set()

    for source_root in dataset_paths:
        source_root = source_root.expanduser().resolve()
        info = _load_json(source_root / "meta/info.json")
        fps = float(info["fps"])
        for source_data_path in _source_episode_paths(source_root, info):
            output_data_path = arena_root / "data/chunk-000" / f"file-{output_episode_index:03d}.parquet"
            if not output_data_path.is_file():
                raise FileNotFoundError(output_data_path)

            source_table = pq.read_table(
                source_data_path,
                columns=_source_episode_columns(source_data_path, include_indices=False),
            )
            output_table = pq.read_table(output_data_path, columns=["observation.state", "action"])
            output_state = _matrix(output_table, "observation.state", 64).astype(np.float32)
            output_action = _matrix(output_table, "action", 40).astype(np.float32)
            expected_state = _build_state(
                source_table,
                info,
                fps,
                strict_source_semantics=strict_source_semantics,
            )
            expected_action = _build_action(
                source_table,
                info,
                root_position_axis_order=root_position_axis_order,
                root_z_mode=root_z_mode,
                root_height=root_height,
                strict_source_semantics=strict_source_semantics,
            )

            source_action_root_p = _matrix(source_table, "action.root_p", 3).astype(np.float32)
            source_action_root_q = _matrix(source_table, "action.root_q", 4).astype(np.float32)
            mapped_root_p = _source_root_positions_for_action(
                source_table,
                root_position_axis_order=root_position_axis_order,
                root_z_mode=root_z_mode,
                root_height=root_height,
            )
            reconstructed_root_p = _integrate_action_root_positions(output_action)
            source_action_root_rot = _quat_wxyz_to_matrix(source_action_root_q, "action.root_q")
            output_action_root_rot = _matrix_from_rot6d_row(output_action[:, 3:9])

            max_errors["observation_root_rot6d"] = max(
                max_errors["observation_root_rot6d"],
                _max_abs_error(output_state[:, :6], expected_state[:, :6]),
            )
            max_errors["observation_joint_q"] = max(
                max_errors["observation_joint_q"],
                _max_abs_error(output_state[:, 6:35], expected_state[:, 6:35]),
            )
            max_errors["observation_joint_dq"] = max(
                max_errors["observation_joint_dq"],
                _max_abs_error(output_state[:, 35:64], expected_state[:, 35:64]),
            )
            max_errors["action_root_xy_position"] = max(
                max_errors["action_root_xy_position"],
                _max_abs_error(reconstructed_root_p[:, :2], mapped_root_p[:, :2]),
            )
            max_errors["action_root_z"] = max(
                max_errors["action_root_z"],
                _max_abs_error(reconstructed_root_p[:, 2], mapped_root_p[:, 2]),
            )
            max_errors["action_root_rotation_matrix"] = max(
                max_errors["action_root_rotation_matrix"],
                _max_abs_error(output_action_root_rot, source_action_root_rot),
            )
            max_errors["action_joint_q"] = max(
                max_errors["action_joint_q"],
                _max_abs_error(output_action[:, 9:38], expected_action[:, 9:38]),
            )
            max_errors["action_hand_binary"] = max(
                max_errors["action_hand_binary"],
                _max_abs_error(output_action[:, 38:40], expected_action[:, 38:40]),
            )
            max_errors["source_first_action_root_p_norm"] = max(
                max_errors["source_first_action_root_p_norm"],
                float(np.linalg.norm(source_action_root_p[0])),
            )
            diagnostics["source_first_action_root_q_identity"] = max(
                diagnostics["source_first_action_root_q_identity"],
                _identity_quaternion_error(source_action_root_q[0]),
            )
            hand_values.update(float(value) for value in np.unique(output_action[:, 38:40]))
            total_frames += output_action.shape[0]
            output_episode_index += 1

    invalid_hand_values = [value for value in sorted(hand_values) if value not in (0.0, 1.0)]
    if invalid_hand_values:
        raise ValueError(f"Converted hand labels are not binary: {invalid_hand_values}")

    failing = {
        key: value
        for key, value in max_errors.items()
        if value > tolerance
    }
    if failing:
        raise ValueError(
            "Converted HumanoidArena content is not semantically aligned within "
            f"tolerance={tolerance}: {failing}"
        )
    return {
        "episodes": output_episode_index,
        "frames": total_frames,
        "tolerance": tolerance,
        "max_errors": max_errors,
        "diagnostics": diagnostics,
        "hand_values": sorted(hand_values),
    }


def _features(video_info: dict) -> dict:
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": int(video_info.get("video.height", 480)),
                "video.width": int(video_info.get("video.width", 640)),
                "video.codec": str(video_info.get("video.codec", "h264")),
                "video.pix_fmt": str(video_info.get("video.pix_fmt", "yuv420p")),
                "video.is_depth_map": False,
                "video.fps": int(video_info.get("video.fps", 50)),
                "video.channels": 3,
                "has_audio": bool(video_info.get("has_audio", False)),
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [64],
            "names": list(STATE_NAMES_64),
        },
        "action": {
            "dtype": "float32",
            "shape": [40],
            "names": list(ACTION_NAMES_40),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _write_tasks(path: Path, selected_task_name: str, selected_prompt: str | None) -> None:
    rows = []
    for task_name, (task_index, _task_id, default_prompt) in ARENA_TASKS.items():
        prompt = selected_prompt if task_name == selected_task_name and selected_prompt else default_prompt
        rows.append({"task_index": task_index, "task": prompt})
    table = pa.table(
        {
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
            "task": pa.array([row["task"] for row in rows], type=pa.large_string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_info(path: Path, total_episodes: int, total_frames: int, video_info: dict) -> None:
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_refpose_v3_1",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(ARENA_TASKS),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 50,
        "splits": {"train": f"0:{int(total_episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _features(video_info),
        "vla_protocol": {
            "schema": ARENA_SCHEMA,
            "version": ARENA_VERSION,
            "action_dim": 40,
            "action_layout": "root_xy_delta_z_rot6d_joints29_hands2",
            "rotation_6d_layout": "row",
            "action_semantics": "reference_pose_not_robot_current_residual",
            "root_xy_delta_frame": "current_reference_base_frame",
            "root_rotation_frame": "episode_reference_frame",
            "fps": 50.0,
            "control_dt": 0.02,
            "dt_source": "converted_from_sonic_lerobot",
            "backend_source": "sonic",
            "twist2_reference_source": None,
            "sonic_reference_source": "converted_action_root_p_root_q_joint_q",
        },
    }
    path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def _write_merge_manifest(
    path: Path,
    *,
    output_root: Path,
    task_name: str,
    task_prompt: str | None,
    video_mode: str,
    total_episodes: int,
    total_frames: int,
) -> None:
    selected_task_index, _selected_task_id, selected_default_prompt = ARENA_TASKS[task_name]
    selected_prompt = task_prompt or selected_default_prompt
    manifest = {
        "repo_id": "local/sonic_8_refpose_v3_1",
        "root": str(output_root),
        "video_mode": video_mode,
        "total_sources": 1,
        "total_tasks": len(ARENA_TASKS),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "sources": [
            {
                "path": str(output_root),
                "task_name": task_name,
                "dataset_name": "sonic_refpose_v3_1",
                "total_episodes": int(total_episodes),
                "total_frames": int(total_frames),
                "task_index_map": {"0": selected_task_index},
            }
        ],
        "tasks": [
            {
                "task_index": task_index,
                "task": selected_prompt if task_index == selected_task_index else prompt,
            }
            for task_index, _task_id, prompt in ARENA_TASKS.values()
        ],
    }
    path.write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")


def _content_alignment_report(
    *,
    task_name: str,
    episode_lengths: list[int],
    state: np.ndarray,
    action: np.ndarray,
) -> dict:
    profile = ARENA_CONTENT_PROFILES[task_name]
    lengths = np.asarray(episode_lengths, dtype=np.float32)
    root_delta_norm = np.linalg.norm(action[:, :2], axis=1)
    root_z = action[:, 2]
    hand_closed_fraction = action[:, 38:40].mean(axis=0)
    joint_velocity_abs_q99 = float(np.quantile(np.abs(state[:, 35:64]), 0.99))
    observed = {
        "episodes": int(lengths.size),
        "episode_length": {
            "min": int(lengths.min()),
            "q50": float(np.quantile(lengths, 0.50)),
            "max": int(lengths.max()),
        },
        "root_delta_norm": {
            "q50": float(np.quantile(root_delta_norm, 0.50)),
            "q99": float(np.quantile(root_delta_norm, 0.99)),
            "max": float(root_delta_norm.max()),
        },
        "root_z": {
            "q01": float(np.quantile(root_z, 0.01)),
            "q50": float(np.quantile(root_z, 0.50)),
            "q99": float(np.quantile(root_z, 0.99)),
            "std": float(root_z.std()),
        },
        "hand_closed_fraction": hand_closed_fraction.tolist(),
        "joint_velocity_abs_q99": joint_velocity_abs_q99,
    }

    warnings = []
    expected_min, expected_median, expected_max = profile["episode_length"]
    if observed["episode_length"]["min"] < expected_min * 0.5:
        warnings.append(
            "Some episodes are much shorter than the official task distribution; "
            "record complete task attempts instead of motion fragments."
        )
    if observed["episode_length"]["max"] > expected_max * 1.5:
        warnings.append(
            "Some episodes are much longer than the official task distribution; split retries "
            "and idle periods into separate discarded episodes."
        )
    if not 0.5 * expected_median <= observed["episode_length"]["q50"] <= 1.8 * expected_median:
        warnings.append("Median episode duration differs strongly from the selected Arena task.")

    expected_delta_q50 = profile["root_delta_q50"]
    expected_delta_q99 = profile["root_delta_q99"]
    if observed["root_delta_norm"]["q50"] < expected_delta_q50 * 0.25:
        warnings.append(
            "Typical reference root motion is much smaller than Arena; remove long idle "
            "segments and verify streamed root translation."
        )
    if observed["root_delta_norm"]["q99"] < expected_delta_q99 * 0.20:
        warnings.append(
            "Reference root motion is nearly static compared with the selected Arena task; "
            "check PICO root translation and avoid recording long idle periods."
        )
    if observed["root_delta_norm"]["q99"] > expected_delta_q99 * 3.0:
        warnings.append(
            "Reference root motion is much more abrupt than Arena; check coordinate jumps, "
            "tracking resets, and episode boundaries."
        )

    expected_z_q01, expected_z_median, expected_z_q99 = profile["root_z_q01_q50_q99"]
    if abs(observed["root_z"]["q50"] - expected_z_median) > 0.15:
        warnings.append("Median root height is more than 15 cm from the selected Arena task.")
    expected_z_span = expected_z_q99 - expected_z_q01
    observed_z_span = observed["root_z"]["q99"] - observed["root_z"]["q01"]
    if observed["root_z"]["std"] < 0.002:
        warnings.append(
            "Root height is effectively constant; new training data must record action.root_z "
            "from the live reference target."
        )
    if expected_z_span > 0.08 and observed_z_span < expected_z_span * 0.25:
        warnings.append(
            "Root height variation is too small for this task; verify that crouching/sitting "
            "height comes from the streamed reference instead of a constant."
        )

    for side, expected_fraction, observed_fraction in zip(
        ("left", "right"),
        profile["hand_closed_fraction"],
        hand_closed_fraction,
    ):
        if expected_fraction >= 0.20 and observed_fraction < 0.01:
            warnings.append(
                f"{side} hand never closes although the official task frequently uses it."
            )
        if expected_fraction <= 0.05 and observed_fraction > 0.80:
            warnings.append(
                f"{side} hand is almost always closed unlike the official task distribution."
            )
        if abs(float(observed_fraction) - expected_fraction) > 0.35:
            warnings.append(
                f"{side} hand closed-frame ratio differs strongly from the official task."
            )

    expected_velocity_q99 = profile["joint_velocity_abs_q99"]
    if joint_velocity_abs_q99 < expected_velocity_q99 * 0.35:
        warnings.append(
            "Measured joint velocity is unusually small; confirm observation.joint_dq is live."
        )
    if joint_velocity_abs_q99 > expected_velocity_q99 * 4.0:
        warnings.append(
            "Measured joint velocity has unusually large outliers; inspect state timing and resets."
        )
    if lengths.size < 20:
        warnings.append(
            "Fewer than 20 episodes gives an unstable content comparison; the official set has "
            "100 episodes per task."
        )

    return {
        "reference": {
            "dataset": "WilliamWang16/HumanoidArena_dataset_v3_1",
            "task_name": task_name,
            "profile": profile,
        },
        "observed": observed,
        "warnings": warnings,
        "status": "warning" if warnings else "aligned",
    }


def convert(args: argparse.Namespace) -> dict:
    selected_task_index, selected_task_id, _default_prompt = ARENA_TASKS[args.task_name]
    output_root = args.output_root.expanduser().resolve()
    arena_root = output_root / "HumanoidArena_merged_datasets_v3_1" / "sonic_8_refpose_v3_1"
    if arena_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {arena_root}. Use --overwrite to replace files.")
    if arena_root.exists() and args.overwrite:
        shutil.rmtree(arena_root)

    data_dir = arena_root / "data/chunk-000"
    video_dir = arena_root / "videos" / VIDEO_KEY / "chunk-000"
    meta_episodes_dir = arena_root / "meta/episodes/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_episodes_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    all_state = []
    all_action = []
    all_timestamp = []
    all_frame_index = []
    all_episode_index = []
    all_index = []
    all_task_index = []
    episode_lengths = []
    global_index = 0
    output_episode_index = 0
    video_info = None
    root_z_min = []
    root_z_max = []

    for source_root in args.dataset_path:
        source_root = source_root.expanduser().resolve()
        info = _load_json(source_root / "meta/info.json")
        fps = float(info["fps"])
        if abs(fps - 50.0) > 1e-6:
            raise ValueError(f"{source_root} has fps={fps}; this converter currently expects 50 FPS")
        chunks_size = int(info.get("chunks_size", 1000))
        source_video_key = _source_video_key(info)
        if video_info is None:
            video_info = info["features"][source_video_key].get("info", {})
        episodes = [
            json.loads(line)
            for line in (source_root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for source_episode in episodes:
            source_episode_index = int(source_episode["episode_index"])
            source_data_path = source_root / _format_source_path(
                info["data_path"],
                video_key=source_video_key,
                episode_index=source_episode_index,
                chunks_size=chunks_size,
            )
            source_video_path = source_root / _format_source_path(
                info["video_path"],
                video_key=source_video_key,
                episode_index=source_episode_index,
                chunks_size=chunks_size,
            )
            if not source_data_path.is_file():
                raise FileNotFoundError(source_data_path)
            if args.video_mode != "skip" and not source_video_path.is_file():
                raise FileNotFoundError(source_video_path)

            table = pq.read_table(
                source_data_path,
                columns=_source_episode_columns(source_data_path, include_indices=True),
            )
            length = table.num_rows
            if int(source_episode["length"]) != length:
                raise ValueError(
                    f"{source_data_path} has {length} rows but metadata says {source_episode['length']}"
                )
            state = _build_state(
                table,
                info,
                fps,
                strict_source_semantics=args.strict_source_semantics,
            )
            action = _build_action(
                table,
                info,
                root_position_axis_order=args.root_position_axis_order,
                root_z_mode=args.root_z_mode,
                root_height=args.root_height,
                strict_source_semantics=args.strict_source_semantics,
            )
            timestamp = np.arange(length, dtype=np.float32) / np.float32(fps)
            frame_index = np.arange(length, dtype=np.int64)
            episode_index = np.full(length, output_episode_index, dtype=np.int64)
            index = np.arange(global_index, global_index + length, dtype=np.int64)
            task_index = np.full(length, selected_task_index, dtype=np.int64)

            data_table = pa.table(
                {
                    "observation.state": _fixed_size_list_array(state, pa.float32()),
                    "action": _fixed_size_list_array(action, pa.float32()),
                    "timestamp": pa.array(timestamp, type=pa.float32()),
                    "frame_index": pa.array(frame_index, type=pa.int64()),
                    "episode_index": pa.array(episode_index, type=pa.int64()),
                    "index": pa.array(index, type=pa.int64()),
                    "task_index": pa.array(task_index, type=pa.int64()),
                }
            )
            output_data_path = data_dir / f"file-{output_episode_index:03d}.parquet"
            pq.write_table(data_table, output_data_path)

            output_video_path = video_dir / f"file-{output_episode_index:03d}.mp4"
            _copy_or_link_video(source_video_path, output_video_path, args.video_mode)

            episode_rows.append(
                {
                    "episode_index": output_episode_index,
                    "tasks": [selected_task_id],
                    "length": length,
                    "data/chunk_index": 0,
                    "data/file_index": output_episode_index,
                    "dataset_from_index": int(global_index),
                    "dataset_to_index": int(global_index + length),
                    f"videos/{VIDEO_KEY}/chunk_index": 0,
                    f"videos/{VIDEO_KEY}/file_index": output_episode_index,
                    f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                    f"videos/{VIDEO_KEY}/to_timestamp": float(length / fps),
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            all_state.append(state)
            all_action.append(action)
            all_timestamp.append(timestamp)
            all_frame_index.append(frame_index)
            all_episode_index.append(episode_index)
            all_index.append(index)
            all_task_index.append(task_index)
            episode_lengths.append(length)
            root_z_min.append(float(action[:, 2].min()))
            root_z_max.append(float(action[:, 2].max()))
            global_index += length
            output_episode_index += 1

    if video_info is None:
        video_info = {}
    episodes_table = pa.table(
        {
            "episode_index": pa.array([row["episode_index"] for row in episode_rows], type=pa.int64()),
            "tasks": pa.array([row["tasks"] for row in episode_rows], type=pa.list_(pa.string())),
            "length": pa.array([row["length"] for row in episode_rows], type=pa.int64()),
            "data/chunk_index": pa.array([row["data/chunk_index"] for row in episode_rows], type=pa.int64()),
            "data/file_index": pa.array([row["data/file_index"] for row in episode_rows], type=pa.int64()),
            "dataset_from_index": pa.array([row["dataset_from_index"] for row in episode_rows], type=pa.int64()),
            "dataset_to_index": pa.array([row["dataset_to_index"] for row in episode_rows], type=pa.int64()),
            f"videos/{VIDEO_KEY}/chunk_index": pa.array(
                [row[f"videos/{VIDEO_KEY}/chunk_index"] for row in episode_rows], type=pa.int64()
            ),
            f"videos/{VIDEO_KEY}/file_index": pa.array(
                [row[f"videos/{VIDEO_KEY}/file_index"] for row in episode_rows], type=pa.int64()
            ),
            f"videos/{VIDEO_KEY}/from_timestamp": pa.array(
                [row[f"videos/{VIDEO_KEY}/from_timestamp"] for row in episode_rows], type=pa.float64()
            ),
            f"videos/{VIDEO_KEY}/to_timestamp": pa.array(
                [row[f"videos/{VIDEO_KEY}/to_timestamp"] for row in episode_rows], type=pa.float64()
            ),
            "meta/episodes/chunk_index": pa.array(
                [row["meta/episodes/chunk_index"] for row in episode_rows], type=pa.int64()
            ),
            "meta/episodes/file_index": pa.array(
                [row["meta/episodes/file_index"] for row in episode_rows], type=pa.int64()
            ),
        }
    )
    pq.write_table(episodes_table, meta_episodes_dir / "file-000.parquet")
    _write_tasks(arena_root / "meta/tasks.parquet", args.task_name, args.task_prompt)
    _write_info(arena_root / "meta/info.json", output_episode_index, global_index, video_info)
    _write_merge_manifest(
        arena_root / "merge_manifest.json",
        output_root=arena_root,
        task_name=args.task_name,
        task_prompt=args.task_prompt,
        video_mode=args.video_mode,
        total_episodes=output_episode_index,
        total_frames=global_index,
    )

    stats_payload = {
        "observation.state": _stats(np.concatenate(all_state, axis=0)),
        "action": _stats(np.concatenate(all_action, axis=0)),
        "timestamp": _stats(np.concatenate(all_timestamp, axis=0)),
        "frame_index": _stats(np.concatenate(all_frame_index, axis=0)),
        "episode_index": _stats(np.concatenate(all_episode_index, axis=0)),
        "index": _stats(np.concatenate(all_index, axis=0)),
        "task_index": _stats(np.concatenate(all_task_index, axis=0)),
    }
    (arena_root / "meta/stats.json").write_text(
        json.dumps(stats_payload, indent=4) + "\n", encoding="utf-8"
    )
    content_alignment = _content_alignment_report(
        task_name=args.task_name,
        episode_lengths=episode_lengths,
        state=np.concatenate(all_state, axis=0),
        action=np.concatenate(all_action, axis=0),
    )
    (arena_root / "content_alignment.json").write_text(
        json.dumps(content_alignment, indent=4) + "\n", encoding="utf-8"
    )

    summary = {
        "arena_root": str(arena_root),
        "episodes": output_episode_index,
        "frames": global_index,
        "root_z_min": min(root_z_min) if root_z_min else None,
        "root_z_max": max(root_z_max) if root_z_max else None,
        "content_alignment": content_alignment,
    }
    if not getattr(args, "skip_validation", False):
        summary["validation"] = validate_conversion(
            list(args.dataset_path),
            arena_root,
            root_position_axis_order=args.root_position_axis_order,
            root_z_mode=args.root_z_mode,
            root_height=args.root_height,
            strict_source_semantics=args.strict_source_semantics,
        )
    return summary


def convert_datasets(
    dataset_paths: list[Path],
    output_root: Path,
    *,
    task_name: str,
    task_prompt: str | None = None,
    root_height: float = DEFAULT_ROOT_HEIGHT_M,
    root_z_mode: str = "constant",
    root_position_axis_order: str = "xyz",
    video_mode: str = "copy",
    overwrite: bool = False,
    skip_validation: bool = False,
    strict_source_semantics: bool = True,
) -> dict:
    args = argparse.Namespace(
        dataset_path=dataset_paths,
        output_root=output_root,
        task_name=task_name,
        task_prompt=task_prompt,
        root_height=root_height,
        root_z_mode=root_z_mode,
        root_position_axis_order=root_position_axis_order,
        video_mode=video_mode,
        overwrite=overwrite,
        skip_validation=skip_validation,
        strict_source_semantics=strict_source_semantics,
    )
    return convert(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source datasets such as outputs/g1_real_20260809_23_00.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that will become HUMANOID_ARENA_ROOT for Kimodo training.",
    )
    parser.add_argument(
        "--task-name",
        choices=sorted(ARENA_TASKS),
        required=True,
        help="Known HumanoidArena task bucket used for hardcoded Kimodo loader mapping.",
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help="Optional natural-language prompt written into meta/tasks.parquet for the selected task.",
    )
    parser.add_argument(
        "--root-height",
        type=float,
        default=DEFAULT_ROOT_HEIGHT_M,
        help="Absolute root height used by --root-z-mode constant/offset.",
    )
    parser.add_argument(
        "--root-z-mode",
        choices=("constant", "offset", "relative"),
        default="constant",
        help=(
            "How to turn episode-relative action.root_p z into HumanoidArena action.root_z. "
            "constant is safest for real G1 logs without absolute target height."
        ),
    )
    parser.add_argument(
        "--root-position-axis-order",
        default="xyz",
        help=(
            "Source action.root_p axes to use as HumanoidArena xyz, e.g. xyz or xzy. "
            "Use this if the streamed root translation uses a different up axis."
        ),
    )
    parser.add_argument(
        "--video-mode",
        choices=("copy", "symlink", "skip"),
        default="copy",
        help="How to materialize videos in the converted dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing HumanoidArena_merged_datasets_v3_1/sonic_8_refpose_v3_1 output.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip source-vs-converted semantic validation after writing the output.",
    )
    parser.add_argument(
        "--allow-legacy-source-semantics",
        action="store_true",
        help=(
            "Allow old datasets without measured joint_dq, root_z, root validity, or direct "
            "hand binary labels. Not recommended for training-quality collection."
        ),
    )
    args = parser.parse_args()
    args.strict_source_semantics = not args.allow_legacy_source_semantics
    return args


def main() -> None:
    summary = convert(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
