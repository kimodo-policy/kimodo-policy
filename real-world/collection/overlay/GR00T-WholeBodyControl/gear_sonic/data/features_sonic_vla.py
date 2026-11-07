"""
Dataset configuration for the Sonic VLA pipeline.

Provides feature/modality config dicts and a convenience function to
instantiate the G1 RobotModel needed for FK and joint configuration
assembly during data collection.

Joint names, counts, and group indices are derived at runtime from the
``RobotModel`` (via its ``supplemental_info``).
"""

from __future__ import annotations

from typing import Literal

from gear_sonic.data.g1_reference import G1_REFERENCE_JOINT_NAMES
from gear_sonic.data.robot_model import RobotModel

EGO_VIEW_HEIGHT: int = 480
EGO_VIEW_WIDTH: int = 640
WRIST_VIEW_HEIGHT: int = 480
WRIST_VIEW_WIDTH: int = 640
FPS: int = 30


_JOINT_GROUPS_FOR_STATE: list[str] = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]


def _get_joint_group_slices(robot_model: RobotModel) -> dict[str, dict[str, int]]:
    """Derive ``{group_name: {"start": ..., "end": ...}}`` from the robot model."""
    slices: dict[str, dict[str, int]] = {}
    for group in _JOINT_GROUPS_FOR_STATE:
        indices = sorted(robot_model.get_joint_group_indices(group))
        slices[group] = {"start": indices[0], "end": indices[-1] + 1}
    return slices


def get_modality_config_sonic_vla(robot_model: RobotModel) -> dict:
    """Return the modality config for the Sonic VLA dataset.

    Produces the exact content of meta/modality.json.
    """
    group_slices = _get_joint_group_slices(robot_model)

    return {
        "state": {
            **group_slices,
            "joint_q": {
                "start": 0,
                "end": 29,
                "original_key": "observation.joint_q",
            },
            "joint_dq": {
                "start": 0,
                "end": 29,
                "original_key": "observation.joint_dq",
            },
            "hand_binary": {
                "start": 0,
                "end": 2,
                "original_key": "observation.hand_binary",
            },
            "left_wrist_pos": {
                "start": 0,
                "end": 3,
                "original_key": "observation.eef_state",
            },
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {
                "start": 7,
                "end": 10,
                "original_key": "observation.eef_state",
            },
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "root_orientation": {
                "start": 0,
                "end": 4,
                "original_key": "observation.root_orientation",
                "rotation_type": "quaternion",
            },
            "root_q_relative": {
                "start": 0,
                "end": 4,
                "original_key": "observation.root_q_relative",
                "rotation_type": "quaternion",
            },
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
            "cpp_rotation_offset": {
                "start": 0,
                "end": 4,
                "original_key": "observation.cpp_rotation_offset",
                "rotation_type": "quaternion",
            },
            "init_base_quat": {
                "start": 0,
                "end": 4,
                "original_key": "observation.init_base_quat",
                "rotation_type": "quaternion",
            },
        },
        "action": {
            "joint_q": {
                "start": 0,
                "end": 29,
                "original_key": "action.joint_q",
            },
            "root_p": {
                "start": 0,
                "end": 3,
                "original_key": "action.root_p",
            },
            "root_q": {
                "start": 0,
                "end": 4,
                "original_key": "action.root_q",
                "rotation_type": "quaternion",
            },
            "root_z": {
                "start": 0,
                "end": 1,
                "original_key": "action.root_z",
            },
            "root_target_valid": {
                "start": 0,
                "end": 1,
                "original_key": "action.root_target_valid",
            },
            "root_target_discontinuous": {
                "start": 0,
                "end": 1,
                "original_key": "action.root_target_discontinuous",
            },
            "hand_binary": {
                "start": 0,
                "end": 2,
                "original_key": "action.hand_binary",
            },
            "hand_q": {
                "start": 0,
                "end": 14,
                "original_key": "action.hand_q",
            },
            "delta_heading": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.delta_heading",
            },
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "smpl_joints": {
                "start": 0,
                "end": 72,
                "original_key": "teleop.smpl_joints",
            },
            "smpl_pose": {
                "start": 0,
                "end": 63,
                "original_key": "teleop.smpl_pose",
            },
            "body_quat_w": {
                "start": 0,
                "end": 4,
                "original_key": "teleop.body_quat_w",
                "rotation_type": "quaternion",
            },
            "target_body_orientation": {
                "start": 0,
                "end": 6,
                "original_key": "teleop.target_body_orientation",
                "rotation_type": "rotation_6d",
            },
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.right_hand_joints",
            },
            "left_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.left_wrist_joints",
            },
            "right_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.right_wrist_joints",
            },
            "stream_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.stream_mode",
            },
            "planner_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_mode",
            },
            "planner_movement": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_movement",
            },
            "planner_facing": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_facing",
            },
            "planner_speed": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_speed",
            },
            "planner_height": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_height",
            },
            "vr_3pt_position": {
                "start": 0,
                "end": 9,
                "original_key": "teleop.vr_3pt_position",
            },
            "vr_3pt_orientation": {
                "start": 0,
                "end": 18,
                "original_key": "teleop.vr_3pt_orientation",
                "rotation_type": "rotation_6d",
            },
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def get_features_sonic_vla(robot_model: RobotModel) -> dict:
    """Return the dataset features for the Sonic VLA dataset.

    The returned dict populates the "features" key of meta/info.json.
    """
    joint_names = robot_model.joint_names
    num_joints = robot_model.num_joints
    body_joint_names = [
        joint_names[index] for index in robot_model.get_body_actuated_joint_indices()
    ]
    if len(body_joint_names) != 29:
        raise ValueError(
            f"Sonic VLA requires 29 body joints, got {len(body_joint_names)}"
        )
    if tuple(body_joint_names) != G1_REFERENCE_JOINT_NAMES:
        raise ValueError(
            "RobotModel body joint order does not match GMR qpos[7:] canonical order"
        )

    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "observation.joint_q": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "observation.joint_dq": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "observation.hand_binary": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_closed", "right_closed"],
        },
        "observation.eef_state": {
            "dtype": "float64",
            "shape": (14,),
            "names": [
                "left_wrist_pos",
                "left_wrist_abs_quat",
                "right_wrist_pos",
                "right_wrist_abs_quat",
            ],
        },
        "action.wbc": {
            "dtype": "float64",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "action.hand_q": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(
                name
                for side in ("left", "right")
                for finger, count in (("thumb", 3), ("index", 2), ("middle", 2))
                for name in (
                    f"{side}_hand_{finger}_{joint_index}_joint"
                    for joint_index in range(count)
                )
            ),
        },
        "action.joint_q": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "control.joint_q_target": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "audit.gmr_joint_q_raw": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "audit.gmr_root_p_raw": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["root_px", "root_py", "root_pz"],
        },
        "audit.gmr_root_q_raw": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["root_qw", "root_qx", "root_qy", "root_qz"],
        },
        "audit.g1_joint_q_measured_raw": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "audit.g1_joint_dq_measured_raw": {
            "dtype": "float64",
            "shape": (29,),
            "names": body_joint_names,
        },
        "audit.g1_root_q_measured_raw": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.root_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.root_q_relative": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["root_qw", "root_qx", "root_qy", "root_qz"],
        },
        "action.root_p": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["root_px", "root_py", "root_pz"],
        },
        "action.root_q": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["root_qw", "root_qx", "root_qy", "root_qz"],
        },
        "action.root_z": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["root_z"],
        },
        "action.root_target_valid": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["root_target_valid"],
        },
        "action.root_target_discontinuous": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["root_target_discontinuous"],
        },
        "action.hand_binary": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_closed", "right_closed"],
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "observation.cpp_rotation_offset": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["rot_offset_qw", "rot_offset_qx", "rot_offset_qy", "rot_offset_qz"],
        },
        "observation.init_base_quat": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["init_base_qw", "init_base_qx", "init_base_qy", "init_base_qz"],
        },
        "teleop.delta_heading": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["delta_heading"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": (64,),
            "names": "motion_token",
        },
        "teleop.smpl_joints": {
            "dtype": "float32",
            "shape": (72,),
            "names": "smpl_joints",
        },
        "teleop.smpl_pose": {
            "dtype": "float32",
            "shape": (63,),
            "names": "smpl_pose",
        },
        "teleop.body_quat_w": {
            "dtype": "float32",
            "shape": (4,),
            "names": "body_quat_w",
        },
        "teleop.target_body_orientation": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "target_body_r00",
                "target_body_r10",
                "target_body_r01",
                "target_body_r11",
                "target_body_r02",
                "target_body_r12",
            ],
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "right_hand_joints",
        },
        "teleop.smpl_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["smpl_frame_index"],
        },
        "teleop.left_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        },
        "teleop.right_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
        },
        "teleop.stream_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["stream_mode"],
        },
        "teleop.planner_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["locomotion_mode"],
        },
        "teleop.planner_movement": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["movement_x", "movement_y", "movement_z"],
        },
        "teleop.planner_facing": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["facing_x", "facing_y", "facing_z"],
        },
        "teleop.planner_speed": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["speed"],
        },
        "teleop.planner_height": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["height"],
        },
        "teleop.vr_3pt_position": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "lwrist_x", "lwrist_y", "lwrist_z",
                "rwrist_x", "rwrist_y", "rwrist_z",
                "neck_x", "neck_y", "neck_z",
            ],
        },
        "teleop.vr_3pt_orientation": {
            "dtype": "float32",
            "shape": (18,),
            "names": [
                "lwrist_r00", "lwrist_r10", "lwrist_r01", "lwrist_r11", "lwrist_r02", "lwrist_r12",
                "rwrist_r00", "rwrist_r10", "rwrist_r01", "rwrist_r11", "rwrist_r02", "rwrist_r12",
                "neck_r00", "neck_r10", "neck_r01", "neck_r11", "neck_r02", "neck_r12",
            ],
        },
        "sync.state_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["state_sequence"],
        },
        "sync.state_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["state_monotonic_timestamp"],
        },
        "sync.reference_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["reference_sequence"],
        },
        "sync.reference_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["reference_monotonic_timestamp"],
        },
        "sync.control_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["control_sequence"],
        },
        "sync.control_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["control_monotonic_timestamp"],
        },
        "sync.video_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["video_sequence"],
        },
        "sync.video_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["video_monotonic_timestamp"],
        },
        "sync.video_capture_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["video_capture_sequence"],
        },
        "sync.video_capture_source_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["video_capture_source_monotonic_timestamp"],
        },
        "sync.camera_resampled": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["camera_resampled"],
        },
        "sync.camera_source_gap_frames": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_source_gap_frames"],
        },
        "sync.camera_source_skipped_captures": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_source_skipped_captures"],
        },
        "sync.smpl_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["smpl_sequence"],
        },
        "sync.smpl_monotonic_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["smpl_monotonic_timestamp"],
        },
        "sync.state_video_receive_skew_ms": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["state_video_receive_skew_ms"],
        },
        "sync.smpl_state_receive_skew_ms": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["smpl_state_receive_skew_ms"],
        },
        "sync.reference_video_receive_skew_ms": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["reference_video_receive_skew_ms"],
        },
        "sync.control_video_receive_skew_ms": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["control_video_receive_skew_ms"],
        },
        "sync.state_source_monotonic_timestamp": {
            "dtype": "float64", "shape": (1,), "names": ["state_source_monotonic_timestamp"],
        },
        "sync.reference_source_monotonic_timestamp": {
            "dtype": "float64", "shape": (1,), "names": ["reference_source_monotonic_timestamp"],
        },
        "sync.video_source_monotonic_timestamp": {
            "dtype": "float64", "shape": (1,), "names": ["video_source_monotonic_timestamp"],
        },
        "sync.smpl_source_monotonic_timestamp": {
            "dtype": "float64", "shape": (1,), "names": ["smpl_source_monotonic_timestamp"],
        },
    }


def get_wrist_camera_features() -> dict:
    """Features for optional wrist cameras (added when ``record_wrist_cameras`` is enabled)."""
    return {
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": [WRIST_VIEW_HEIGHT, WRIST_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [WRIST_VIEW_HEIGHT, WRIST_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
    }


def get_wrist_camera_modality_config() -> dict:
    """Modality config entries for optional wrist cameras."""
    return {
        "video": {
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
    }


def get_g1_robot_model(
    waist_location: Literal[
        "lower_body", "upper_body", "lower_and_upper_body"
    ] = "lower_and_upper_body",
    high_elbow_pose: bool = False,
):
    """Instantiate the G1 + ThreeFinger RobotModel for Sonic VLA."""
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

    return instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=high_elbow_pose,
    )
