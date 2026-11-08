#!/usr/bin/env python3
"""
Add detailed debug logging to action_provider_sonic.py.

This script inserts print statements at key points to trace data changes through the pipeline.

Usage:
1. Back up the original file: cp action_provider_sonic.py action_provider_sonic.py.bak
2. Run this script: python add_debug_logs.py
3. Start IsaacLab and swing your hands rhythmically in VR.
4. Inspect the logs to identify the stage where data stops changing rhythmically.
5. Restore the original file: mv action_provider_sonic.py.bak action_provider_sonic.py
"""

import os
import sys

# Debug code snippets to insert
DEBUG_SNIPPETS = {
    # After receiving smpl_joints in _fetch_zmq_pose()
    "after_smpl_joints_received": '''
        # [DEBUG] Check changes in SMPL joints.
        if hasattr(self, '_prev_smpl_joints_frame'):
            smpl_diff = np.abs(frame - self._prev_smpl_joints_frame).max()
            print(f"[DEBUG_ZMQ] Maximum SMPL joint change: {smpl_diff:.6f}")
        self._prev_smpl_joints_frame = frame.copy()
''',

    # After building the encoder input in _run_gear_sonic()
    "after_encoder_input_built": '''
        # [DEBUG] Check changes in encoder input.
        if hasattr(self, '_prev_encoder_input'):
            enc_diff = np.abs(encoder_input - self._prev_encoder_input).max()
            print(f"[DEBUG_ENC] Maximum encoder input change: {enc_diff:.6f}")
            smpl_part = encoder_input[0, -780:-720]  # A subset of smpl_joints.
            smpl_part_diff = np.abs(smpl_part - self._prev_encoder_input[0, -780:-720]).max()
            print(f"[DEBUG_ENC] Maximum SMPL component change: {smpl_part_diff:.6f}")
        self._prev_encoder_input = encoder_input.copy()
''',

    # After encoder inference
    "after_encoder_inference": '''
        # [DEBUG] Check changes in the latent representation.
        if hasattr(self, '_prev_latent'):
            latent_diff = np.abs(latent - self._prev_latent).max()
            print(f"[DEBUG_LATENT] Maximum latent change: {latent_diff:.6f}")
        self._prev_latent = latent.copy()
''',

    # After decoder inference
    "after_decoder_inference": '''
        # [DEBUG] Check changes in actions.
        if hasattr(self, '_prev_action_raw'):
            action_diff = np.abs(action_raw - self._prev_action_raw).max()
            print(f"[DEBUG_ACTION] Maximum action change: {action_diff:.6f}")
            print(f"[DEBUG_ACTION] First 5 action dimensions: {action_raw[0, :5]}")
        self._prev_action_raw = action_raw.copy()

        # [DEBUG] Check the robot state in the decoder input.
        print(f"[DEBUG_ROBOT] First 5 joint positions: {self._robot_joint_pos_hist[-1, :5]}")
        print(f"[DEBUG_ROBOT] First 5 joint velocities: {self._robot_joint_vel_hist[-1, :5]}")
''',

    # Before the final output
    "before_final_output": '''
        # [DEBUG] Check changes in the final targets.
        if hasattr(self, '_prev_sonic_targets'):
            targets_diff = np.abs(sonic_targets - self._prev_sonic_targets).max()
            print(f"[DEBUG_TARGETS] Maximum target change: {targets_diff:.6f}")
            print(f"[DEBUG_TARGETS] First 5 target dimensions: {sonic_targets[:5]}")
        self._prev_sonic_targets = sonic_targets.copy()
''',
}

def add_debug_logs():
    """Add debug logging to action_provider_sonic.py."""

    file_path = "../action_provider/action_provider_sonic.py"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print(f"   Run this script from the tools/ directory.")
        return

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find insertion points and add debug code
    insertions = []

    # 1. After receiving smpl_joints (around line 638)
    for i, line in enumerate(lines):
        if 'self._smpl_joints_buf[-1] = frame' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_smpl_joints_received"]))
            print(f"✓ Found insertion point 1: line {i+1} (after receiving SMPL joints)")
            break

    # 2. After building the encoder input (around line 874)
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] Encoder input shape: {encoder_input.shape}' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_encoder_input_built"]))
            print(f"✓ Found insertion point 2: line {i+1} (after building encoder input)")
            break

    # 3. After encoder inference(around line 930)
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] ✓ Encoder output latent shape: {latent.shape}")' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_encoder_inference"]))
            print(f"✓ Found insertion point 3: line {i+1} (after encoder inference)")
            break

    # 4. After decoder inference(around line 960)
    for i, line in enumerate(lines):
        if 'print(f"[SONIC] ✓ Decoder output action shape: {action_raw.shape}")' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["after_decoder_inference"]))
            print(f"✓ Found insertion point 4: line {i+1} (after decoder inference)")
            break

    # 5. Before the final output(around line 1000)
    for i, line in enumerate(lines):
        if 'sonic_targets = SONIC_DEFAULT_POS + action_scaled' in line:
            insertions.append((i+1, DEBUG_SNIPPETS["before_final_output"]))
            print(f"✓ Found insertion point 5: line {i+1} (before final output)")
            break

    if len(insertions) < 5:
        print(f"\n❌ Found only {len(insertions)}/5 insertion points")
        print(f"   The file may have changed; inspect it before continuing.")
        return

    # Insert in descending line-number order to avoid shifting later locations
    insertions.sort(reverse=True)

    for line_num, code in insertions:
        lines.insert(line_num, code + '\n')

    # Write the file
    output_path = file_path + ".debug"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\n✅ Debug version generated: {output_path}")
    print(f"\nUsage:")
    print(f"1. Back up the original file: cp {file_path} {file_path}.bak")
    print(f"2. Replace the file: mv {output_path} {file_path}")
    print(f"3. Start IsaacLab and swing your hands in VR (left and right at about 1 Hz for 5 seconds).")
    print(f"4. Inspect the [DEBUG_*] log output.")
    print(f"5. Restore the original file: mv {file_path}.bak {file_path}")
    print(f"\nKey metrics:")
    print(f"- [DEBUG_ZMQ] Maximum SMPL joint change: expected >0.01")
    print(f"- [DEBUG_ENC] Maximum encoder input change: expected >0.01")
    print(f"- [DEBUG_LATENT] Maximum latent change: expected >0.01")
    print(f"- [DEBUG_ACTION] Maximum action change: expected >0.01")
    print(f"- [DEBUG_TARGETS] Maximum target change: expected >0.01")
    print(f"\nA change below 0.001 at any stage suggests a problem at that stage.")

if __name__ == "__main__":
    add_debug_logs()
