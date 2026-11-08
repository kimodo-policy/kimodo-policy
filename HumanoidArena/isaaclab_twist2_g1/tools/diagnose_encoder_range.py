#!/usr/bin/env python3
"""
Diagnose encoder input fields outside their expected ranges.
"""

import numpy as np

# Encoder input ranges extracted from logs
encoder_input_min = -31.945160
encoder_input_max = 29.832436

print("=" * 80)
print("Encoder Input Range Diagnostics")
print("=" * 80)
print(f"\nCurrent range: [{encoder_input_min:.6f}, {encoder_input_max:.6f}]")
print(f"Expected range: [-10.0, 10.0]")
print(f"\n❌ Input out of range: approximately {max(abs(encoder_input_min), abs(encoder_input_max)) / 10:.1f}x the expected magnitude")

print("\n" + "=" * 80)
print("Expected ranges by field")
print("=" * 80)

fields = [
    ("encoder_mode", 4, "[-1, 1]", "Mode flags"),
    ("motion_joint_pos_step5", 290, "[-3.14, 3.14]", "Joint positions (radians)"),
    ("motion_joint_vel_step5", 290, "[-10, 10]", "Joint velocities (rad/s)"),
    ("motion_root_z_step5", 10, "[0, 2]", "Root height (m)"),
    ("motion_root_z", 1, "[0, 2]", "Root height (m)"),
    ("motion_anchor_orient", 6, "[-1, 1]", "6D rotation"),
    ("motion_anchor_orient_step5", 60, "[-1, 1]", "6D rotation history"),
    ("motion_joint_pos_lowerbody", 120, "[-3.14, 3.14]", "Lower-body joint positions"),
    ("motion_joint_vel_lowerbody", 120, "[-10, 10]", "Lower-body joint velocities"),
    ("vr_3pt_pos", 9, "[-2, 2]", "VR target positions"),
    ("vr_3pt_orn", 12, "[-1, 1]", "VR target rotations"),
    ("smpl_joints", 720, "[-1, 1]", "SMPL joint positions in local coordinates"),
    ("smpl_anchor_orient", 60, "[-1, 1]", "SMPL anchor rotations"),
    ("motion_wrist_pos", 60, "[-3.14, 3.14]", "Wrist joint positions"),
]

print("\nField                           Size    Expected range     Description")
print("-" * 80)
for name, dims, range_str, desc in fields:
    print(f"{name:30s} {dims:4d}    {range_str:15s}    {desc}")

print("\n" + "=" * 80)
print("Likely causes")
print("=" * 80)

print("\n1. Excessive joint velocity (motion_joint_vel_step5)")
print("   Cause: unstable simulation or robot oscillation")
print("   Remedy: check PD parameters or clip velocities to [-10, 10]")

print("\n2. Incorrect SMPL units (smpl_joints)")
print("   Cause: SMPL data may be in mm instead of m")
print("   Remedy: check SMPL data units in pico_server")

print("\n3. Uninitialized history buffers")
print("   Cause: history buffers contain NaN or extremely large values")
print("   Remedy: check initialization in on_env_reset")

print("\n" + "=" * 80)
print("Next steps")
print("=" * 80)
print("\nSearch the IsaacLab logs for:")
print("  [SONIC][ENCODER_INPUT] motion_pos_step5=[...] motion_vel_step5=[...]")
print("\nThen identify the field whose range is abnormal.")
