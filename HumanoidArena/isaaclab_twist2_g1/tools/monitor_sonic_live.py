#!/usr/bin/env python3
"""
SONIC live data monitoring utility.

Usage:
1. Start pico_server.
2. Run: python monitor_sonic_live.py
3. Move in VR and observe changes in the data.

Displays:
- SMPL joint variation amplitudes.
- Body orientation changes.
- Joint position changes.
- Data reception frequency.
"""

import json
import sys
import time
import numpy as np
import zmq
from collections import deque

_HEADER_SIZE = 1280

def parse_zmq_pose(raw: bytes):
    """Parse a ZMQ pose message."""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        _DTYPE_MAP = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4), "i64": (np.int64, 8),
            "u8": (np.uint8, 1), "bool": (np.bool_, 1),
        }
        result, offset = {}, 0
        for f in header.get("fields", []):
            np_dtype, itemsize = _DTYPE_MAP.get(f["dtype"], (np.float32, 4))
            n = int(np.prod(f["shape"]))
            arr = np.frombuffer(payload[offset: offset + n * itemsize],
                                dtype=np_dtype).reshape(f["shape"])
            result[f["name"]] = arr
            offset += n * itemsize
        return result
    except Exception as e:
        return None

def monitor():
    """Monitor the data stream in real time."""
    ZMQ_HOST = "localhost"
    ZMQ_PORT = 5556

    print("=" * 80)
    print("SONIC Live Data Monitor")
    print("=" * 80)
    print(f"Connecting to: {ZMQ_HOST}:{ZMQ_PORT}")
    print("Press Ctrl+C to stop.\n")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    # History is used to calculate changes
    smpl_joints_history = deque(maxlen=10)
    body_quat_history = deque(maxlen=10)
    joint_pos_history = deque(maxlen=10)

    frame_count = 0
    last_report_time = time.time()
    fps_counter = 0

    print("Waiting for data...\n")

    try:
        while True:
            try:
                raw = socket.recv()
                fps_counter += 1
            except zmq.error.Again:
                continue

            data = parse_zmq_pose(raw)
            if data is None:
                continue

            frame_count += 1

            # Extract data
            smpl_joints = data.get("smpl_joints", np.zeros((1, 24, 3)))[-1]  # (24, 3)
            body_quat = data.get("body_quat_w", np.zeros((1, 4)))[-1]  # (4,)
            joint_pos = data.get("joint_pos", np.zeros((1, 29)))[-1]  # (29,)

            # Add data to history
            smpl_joints_history.append(smpl_joints)
            body_quat_history.append(body_quat)
            joint_pos_history.append(joint_pos)

            # Report once per second
            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - last_report_time
                fps = fps_counter / elapsed

                # Calculate the magnitude of change
                if len(smpl_joints_history) >= 2:
                    smpl_diff = np.abs(smpl_joints_history[-1] - smpl_joints_history[0]).max()
                    quat_diff = np.abs(body_quat_history[-1] - body_quat_history[0]).max()
                    joint_diff = np.abs(joint_pos_history[-1] - joint_pos_history[0]).max()
                else:
                    smpl_diff = quat_diff = joint_diff = 0.0

                # Calculate the data range
                smpl_range = f"[{smpl_joints.min():.3f}, {smpl_joints.max():.3f}]"
                joint_range = f"[{joint_pos.min():.3f}, {joint_pos.max():.3f}]"

                # Clear the screen and print
                print("\033[2J\033[H")  # ANSI screen clear
                print("=" * 80)
                print(f"SONIC Live Monitor | Frame: {frame_count} | FPS: {fps:.1f}")
                print("=" * 80)
                print(f"\n📊 Statistics (last 1 second):")
                print(f"  SMPL joint variation: {smpl_diff:.4f}  Range: {smpl_range}")
                print(f"  Body quaternion variation: {quat_diff:.4f}  Current: {body_quat}")
                print(f"  Joint position variation: {joint_diff:.4f}  Range: {joint_range}")

                # Status indicator
                print(f"\n🎯 Status:")
                if smpl_diff < 0.001:
                    print(f"  ⚠️  SMPL data barely changes: move in VR!")
                elif smpl_diff < 0.01:
                    print(f"  ⚡ Small changes in SMPL data")
                else:
                    print(f"  ✅ SMPL data changes normally")

                if joint_diff < 0.001:
                    print(f"  ⚠️  Joint positions barely change")
                elif joint_diff < 0.1:
                    print(f"  ⚡ Small changes in joint positions")
                else:
                    print(f"  ✅ Joint positions change normally")

                # Monitor key joints (right shoulder, elbow, and wrist)
                print(f"\n🤖 Key joint positions (right arm):")
                print(f"  Right shoulder pitch [12]: {joint_pos[12]:+.3f}")
                print(f"  Right shoulder roll  [16]: {joint_pos[16]:+.3f}")
                print(f"  Right shoulder yaw   [20]: {joint_pos[20]:+.3f}")
                print(f"  Right elbow          [22]: {joint_pos[22]:+.3f}")
                print(f"  Right wrist roll     [24]: {joint_pos[24]:+.3f}")

                print(f"\n💡 Tips:")
                print(f"  - Raise your right hand in VR and watch the right shoulder/elbow values.")
                print(f"  - Unchanging values indicate a data stream problem.")
                print(f"  - Press Ctrl+C to stop monitoring.")

                last_report_time = current_time
                fps_counter = 0

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")

if __name__ == "__main__":
    monitor()
