#!/usr/bin/env python3
"""
Detect periodic changes in the SONIC data stream.

Usage:
1. Start pico_server.
2. Run: python detect_periodic_motion.py
3. Swing your hands rhythmically in VR (for example, left and right at about 1 Hz).
4. The script analyzes periodicity to identify where the pattern is lost.

Output:
- Periodicity analysis of Pico data.
- Variation amplitudes for each key data field.
- FFT spectrum analysis to detect the dominant frequency.
"""

import json
import sys
import time
import numpy as np
import zmq
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Do not display a window
import matplotlib.pyplot as plt

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

def analyze_periodicity(signal, fps=50):
    """Analyze signal periodicity."""
    if len(signal) < 50:
        return None, None

    # Remove the mean
    signal = signal - np.mean(signal)

    # FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1.0/fps)

    # Use positive frequencies only
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = np.abs(fft[:len(fft)//2])

    # Find the dominant frequency, excluding the DC component
    if len(positive_freqs) > 1:
        main_freq_idx = np.argmax(positive_fft[1:]) + 1
        main_freq = positive_freqs[main_freq_idx]
        main_power = positive_fft[main_freq_idx]
        return main_freq, main_power
    return None, None

def detect_periodic_motion():
    """Detect periodic motion."""
    ZMQ_HOST = "localhost"
    ZMQ_PORT = 5556
    BUFFER_SIZE = 200  # 4 seconds of data (50 Hz)

    print("=" * 80)
    print("SONIC Periodic Motion Detection")
    print("=" * 80)
    print(f"Connecting to: {ZMQ_HOST}:{ZMQ_PORT}")
    print("\nSwing your hands rhythmically in VR (left and right at about 1 Hz for 5 seconds).")
    print("Press Ctrl+C to stop.\n")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    # Data buffers
    smpl_joints_history = deque(maxlen=BUFFER_SIZE)
    body_quat_history = deque(maxlen=BUFFER_SIZE)
    joint_pos_history = deque(maxlen=BUFFER_SIZE)

    # Extract history for selected joints related to the right hand
    right_shoulder_pitch_history = deque(maxlen=BUFFER_SIZE)  # joint_pos[12]
    right_elbow_history = deque(maxlen=BUFFER_SIZE)  # joint_pos[22]

    # SMPL right-wrist position history
    smpl_right_wrist_x_history = deque(maxlen=BUFFER_SIZE)  # smpl_joints[20, 0]

    frame_count = 0
    start_time = time.time()

    print("Collecting data...\n")

    try:
        while True:
            try:
                raw = socket.recv()
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

            # Extract key joints
            right_shoulder_pitch_history.append(joint_pos[12])
            right_elbow_history.append(joint_pos[22])
            smpl_right_wrist_x_history.append(smpl_joints[20, 0])  # Right-wrist X coordinate

            # Report once per second
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Collected {frame_count} frames ({elapsed:.1f} seconds)")

                # Analyze periodic motion when enough data is available
                if len(smpl_right_wrist_x_history) >= 100:
                    print("\n" + "=" * 80)
                    print("Periodicity analysis (last 2 seconds)")
                    print("=" * 80)

                    # Analyze the SMPL right-wrist X coordinate
                    signal = np.array(list(smpl_right_wrist_x_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 SMPL right wrist X:")
                    print(f"   Standard deviation: {std:.4f}")
                    if freq is not None:
                        print(f"   Dominant frequency: {freq:.2f} Hz")
                        print(f"   Power: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ Periodic motion detected at approximately {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  Unexpected frequency (expected 0.5–2 Hz)")
                    else:
                        print(f"   ❌ No clear periodicity detected")

                    # Analyze the right shoulder pitch in joint_pos
                    signal = np.array(list(right_shoulder_pitch_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 G1 right shoulder pitch (joint_pos[12]):")
                    print(f"   Standard deviation: {std:.4f}")
                    if freq is not None:
                        print(f"   Dominant frequency: {freq:.2f} Hz")
                        print(f"   Power: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ Periodic motion detected at approximately {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  Unexpected frequency (expected 0.5–2 Hz)")
                    else:
                        print(f"   ❌ No clear periodicity detected")

                    # Analyze the right elbow in joint_pos
                    signal = np.array(list(right_elbow_history))
                    freq, power = analyze_periodicity(signal)
                    std = np.std(signal)

                    print(f"\n📊 G1 right elbow (joint_pos[22]):")
                    print(f"   Standard deviation: {std:.4f}")
                    if freq is not None:
                        print(f"   Dominant frequency: {freq:.2f} Hz")
                        print(f"   Power: {power:.2f}")
                        if freq > 0.5 and freq < 2.0:
                            print(f"   ✅ Periodic motion detected at approximately {freq:.2f} Hz")
                        else:
                            print(f"   ⚠️  Unexpected frequency (expected 0.5–2 Hz)")
                    else:
                        print(f"   ❌ No clear periodicity detected")

                    print("\n" + "=" * 80)
                    print("Continuing data collection...\n")

    except KeyboardInterrupt:
        print("\n\nData collection stopped")

        # Final analysis
        if len(smpl_right_wrist_x_history) >= 50:
            print("\n" + "=" * 80)
            print("Final Analysis Report")
            print("=" * 80)

            # Plot the time-series data
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))

            # SMPL right-wrist X
            signal1 = np.array(list(smpl_right_wrist_x_history))
            axes[0].plot(signal1)
            axes[0].set_title('SMPL Right Wrist X Coordinate')
            axes[0].set_ylabel('Position (m)')
            axes[0].grid(True)

            # G1 right-shoulder pitch
            signal2 = np.array(list(right_shoulder_pitch_history))
            axes[1].plot(signal2)
            axes[1].set_title('G1 Right Shoulder Pitch (joint_pos[12])')
            axes[1].set_ylabel('Angle (rad)')
            axes[1].grid(True)

            # G1 right elbow
            signal3 = np.array(list(right_elbow_history))
            axes[2].plot(signal3)
            axes[2].set_title('G1 Right Elbow (joint_pos[22])')
            axes[2].set_ylabel('Angle (rad)')
            axes[2].set_xlabel('Frame')
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig('periodic_motion_analysis.png', dpi=150)
            print("\n📈 Time-series plot saved to: periodic_motion_analysis.png")

            # Numerical analysis
            print(f"\n📊 Statistics:")
            print(f"   SMPL right wrist X: range [{signal1.min():.3f}, {signal1.max():.3f}], std={np.std(signal1):.4f}")
            print(f"   G1 right shoulder pitch: range [{signal2.min():.3f}, {signal2.max():.3f}], std={np.std(signal2):.4f}")
            print(f"   G1 right elbow: range [{signal3.min():.3f}, {signal3.max():.3f}], std={np.std(signal3):.4f}")

            # Evaluate
            print(f"\n🔍 Diagnosis:")
            smpl_has_variation = np.std(signal1) > 0.01
            g1_shoulder_has_variation = np.std(signal2) > 0.01
            g1_elbow_has_variation = np.std(signal3) > 0.01

            if smpl_has_variation:
                print(f"   ✅ SMPL data changes significantly")
            else:
                print(f"   ❌ SMPL data barely changes: check Pico tracking")

            if g1_shoulder_has_variation:
                print(f"   ✅ G1 joint positions change significantly")
            else:
                print(f"   ❌ G1 joint positions barely change: check SMPL-to-G1 conversion")

            if smpl_has_variation and not g1_shoulder_has_variation:
                print(f"\n⚠️  Key finding: SMPL data changes normally, but G1 joint positions remain constant.")
                print(f"   Check joint position calculations in pico_server (lines 1286–1344)")
                print(f"   and whether joint_pos data reaches IsaacLab correctly.")

if __name__ == "__main__":
    detect_periodic_motion()
