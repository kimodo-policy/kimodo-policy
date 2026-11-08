# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
gripper state
"""      

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def get_robot_girl_joint_names() -> list[str]:
    return [
        "right_hand_Joint1_1",
        "left_hand_Joint1_1",
    ]

# global variable to cache the DDS instance
_gripper_dds = None
_dds_initialized = False

def _get_gripper_dds_instance():
    """get the DDS instance, delay initialization"""
    global _gripper_dds, _dds_initialized
    
    if not _dds_initialized or _gripper_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            
            _gripper_dds = dds_manager.get_object("dex1")
            print("[Observations] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _gripper_dds:
                        dds_manager.unregister_object("dex1")
                        print("[gripper_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[gripper_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations] Failed to get DDS instances: {e}")
            _gripper_dds = None
        
        _dds_initialized = True
    
    return _gripper_dds

def initialize_gripper_dds():
    """explicitly initialize the DDS communication
    
    this function can be called manually to initialize the DDS communication,
    instead of relying on delayed initialization
    """
    return _get_gripper_dds_instance()


def get_robot_gipper_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
    """
    # get the gripper joint states
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel  
    joint_torque = env.scene["robot"].data.applied_torque
    
    # get the gripper joint indices (last 2 joints)
    # gripper_joint_names = get_robot_girl_joint_names()
    # all_joint_names = env.scene["robot"].data.joint_names
    # gripper_joint_indices = [all_joint_names.index(name) for name in gripper_joint_names if name in all_joint_names]
    # print(f"gripper_joint_indices: {gripper_joint_indices}")
    gripper_joint_indices = [31, 29]
    if len(gripper_joint_indices) >= 2:
        # extract the gripper joint states in the specified order
        gripper_positions = joint_pos[:, gripper_joint_indices]
        gripper_velocities = joint_vel[:, gripper_joint_indices]  
        gripper_torques = joint_torque[:, gripper_joint_indices]
        
        # publish to DDS (only publish the data of the first environment)
        if enable_dds and len(gripper_positions) > 0:
            try:

                gripper_dds = _get_gripper_dds_instance()
                if gripper_dds:
                    pos = gripper_positions[0].cpu().numpy()
                    vel = gripper_velocities[0].cpu().numpy()
                    torque = gripper_torques[0].cpu().numpy()
                    right_pos = pos[:1]
                    left_pos = pos[1:]
                    right_vel = vel[:1]
                    left_vel = vel[1:]
                    right_torque = torque[:1]
                    left_torque = torque[1:]
                    # write the gripper state to shared memory
                    gripper_dds.write_gripper_state(left_pos, left_vel, left_torque, right_pos, right_vel, right_torque)
            except Exception as e:
                print(f"[gripper_state] Failed to write to shared memory: {e}")
        
        return gripper_positions
    else:
        # if the gripper joints are not found, return a zero tensor
        # print(f"[gripper_state] Warning: no gripper joints found, expected: {gripper_joint_names}, available: {all_joint_names}")
        print(f"[gripper_state] Warning: no gripper joints found")
        return torch.zeros((joint_pos.shape[0], 2))

