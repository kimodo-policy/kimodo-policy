#!/usr/bin/env python3
"""
Joint position tracking statistics (text only, no GUI).
Track errors for 29 joints in real time to help tune PD parameters.
"""

import numpy as np
from collections import deque
import time


class JointPositionTracker:
    """Track statistics for target and current joint positions."""

    def __init__(self, num_joints=29, window_size=200):
        """
        Args:
            num_joints: Number of joints.
            window_size: Sliding window size for the moving average.
        """
        self.num_joints = num_joints
        self.window_size = window_size

        # Data buffers
        self.errors = [deque(maxlen=window_size) for _ in range(num_joints)]
        self.timestamps = deque(maxlen=window_size)

        # Joint names (G1 29 DOF)
        self.joint_names = [
            # Legs (12)
            "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
            # Waist (3)
            "waist_yaw", "waist_roll", "waist_pitch",
            # Arms (14)
            "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
            "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
            "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
            "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw",
        ]

        # Statistics
        self.max_error = np.zeros(num_joints)
        self.mean_error = np.zeros(num_joints)
        self.current_error = np.zeros(num_joints)

        # Latest target and current positions
        self.last_target = None
        self.last_current = None

        # Update count
        self.update_count = 0

    def update_data(self, target_pos, current_pos, timestamp=None):
        """
        Update the data.

        Args:
            target_pos: Target positions, shape (num_joints,).
            current_pos: Current positions, shape (num_joints,).
            timestamp: Optional timestamp.
        """
        if timestamp is None:
            timestamp = self.update_count

        self.timestamps.append(timestamp)
        self.last_target = target_pos.copy()
        self.last_current = current_pos.copy()

        for i in range(self.num_joints):
            error = abs(target_pos[i] - current_pos[i])
            self.errors[i].append(error)
            self.current_error[i] = error

            # Update statistics
            self.max_error[i] = max(self.max_error[i], error)
            if len(self.errors[i]) > 0:
                self.mean_error[i] = np.mean(list(self.errors[i]))

        self.update_count += 1

    def get_statistics(self):
        """Get statistics."""
        stats = {}
        for i in range(self.num_joints):
            joint_name = self.joint_names[i] if i < len(self.joint_names) else f"Joint_{i}"
            stats[joint_name] = {
                'current_error': self.current_error[i],
                'max_error': self.max_error[i],
                'mean_error': self.mean_error[i],
            }
        return stats

    def print_statistics(self, top_n=10):
        """Print statistics for the top N joints with the largest errors."""
        print("\n" + "="*100)
        print(f"Joint Tracking Statistics (Updates: {self.update_count}, Window: {len(self.timestamps)})")
        print("="*100)

        # Sort by current error
        stats = self.get_statistics()
        sorted_joints = sorted(stats.items(), key=lambda x: x[1]['current_error'], reverse=True)

        print(f"{'Joint Name':<25} | {'Current Error':>12} | {'Mean Error':>12} | {'Max Error':>12}")
        print("-"*100)

        # Show the N joints with the largest errors
        for joint_name, stat in sorted_joints[:top_n]:
            print(f"{joint_name:<25} | {stat['current_error']:12.6f} | {stat['mean_error']:12.6f} | {stat['max_error']:12.6f}")

        if len(sorted_joints) > top_n:
            print(f"... and {len(sorted_joints) - top_n} more joints")

        # Overall statistics
        all_current_errors = [s['current_error'] for s in stats.values()]
        all_mean_errors = [s['mean_error'] for s in stats.values()]
        all_max_errors = [s['max_error'] for s in stats.values()]

        print("-"*100)
        print(f"{'OVERALL':<25} | {np.mean(all_current_errors):12.6f} | {np.mean(all_mean_errors):12.6f} | {np.max(all_max_errors):12.6f}")
        print("="*100 + "\n")

    def print_compact_summary(self):
        """Print a compact summary."""
        if self.update_count == 0:
            return

        stats = self.get_statistics()
        all_current_errors = [s['current_error'] for s in stats.values()]
        all_mean_errors = [s['mean_error'] for s in stats.values()]

        # Find the 3 joints with the largest errors
        sorted_joints = sorted(stats.items(), key=lambda x: x[1]['current_error'], reverse=True)
        top3 = sorted_joints[:3]

        print(f"[JointTracker] Updates: {self.update_count:6d} | "
              f"Avg Error: {np.mean(all_current_errors):.4f} | "
              f"Top3: {top3[0][0]}({top3[0][1]['current_error']:.4f}), "
              f"{top3[1][0]}({top3[1][1]['current_error']:.4f}), "
              f"{top3[2][0]}({top3[2][1]['current_error']:.4f})")


# Usage example
if __name__ == "__main__":
    import time

    # Create the tracker
    tracker = JointPositionTracker(num_joints=29, window_size=200)

    print("Simulating joint tracking...")
    print("Press Ctrl+C to stop")

    try:
        t = 0
        while True:
            # Generate simulated data
            target = np.sin(t * 0.1 + np.arange(29) * 0.2)
            current = target + np.random.randn(29) * 0.05  # Add noise

            tracker.update_data(target, current, timestamp=t)

            t += 1
            time.sleep(0.02)  # 50Hz

            # Print a compact summary every 50 steps
            if t % 50 == 0:
                tracker.print_compact_summary()

            # Print detailed statistics every 200 steps
            if t % 200 == 0:
                tracker.print_statistics(top_n=10)

    except KeyboardInterrupt:
        print("\nStopping...")
        tracker.print_statistics(top_n=29)
