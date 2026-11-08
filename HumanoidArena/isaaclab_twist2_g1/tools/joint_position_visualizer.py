#!/usr/bin/env python3
"""
Joint position visualization utility.
Display target and current positions of 29 joints in real time to help tune PD parameters.
"""

import numpy as np
import os
import sys

# Fix Qt plugin path conflict with OpenCV
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.rcParams['toolbar'] = 'None'  # Disable toolbar to avoid icon resize bug
# Use TkAgg backend (works without Qt)
try:
    matplotlib.use('TkAgg')
    print("[JointViz] Using TkAgg backend for matplotlib")
except Exception as e:
    print(f"[JointViz] Warning: Could not set TkAgg backend: {e}")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading


class JointPositionVisualizer:
    """Display target and current joint positions in a sliding window."""

    def __init__(self, num_joints=29, window_size=200, update_interval=50):
        """
        Args:
            num_joints: Number of joints.
            window_size: Sliding window size in time steps.
            update_interval: Update interval in milliseconds.
        """
        self.num_joints = num_joints
        self.window_size = window_size
        self.update_interval = update_interval

        # Data buffers
        self.target_positions = [deque(maxlen=window_size) for _ in range(num_joints)]
        self.current_positions = [deque(maxlen=window_size) for _ in range(num_joints)]
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

        # Indices of the currently displayed joints
        self.current_joint_idx = 0
        self.joints_per_plot = 6  # Display 6 joints per plot

        # Thread lock
        self.lock = threading.Lock()

        # Statistics
        self.max_error = np.zeros(num_joints)
        self.mean_error = np.zeros(num_joints)

        # Initialize the figure
        self._init_plot()

    def _init_plot(self):
        """Initialize the matplotlib figure."""
        # Set minimum font size to avoid ppem error
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 8

        # Create figure with explicit DPI
        self.fig, self.axes = plt.subplots(3, 2, figsize=(16, 10), dpi=100)
        self.fig.suptitle('Joint Position Tracking (Press Left/Right to navigate)', fontsize=14)
        self.axes = self.axes.flatten()

        # Initialize each subplot
        self.lines_target = []
        self.lines_current = []
        self.lines_error = []

        for i in range(self.joints_per_plot):
            ax = self.axes[i]
            line_target, = ax.plot([], [], 'b-', label='Target', linewidth=1.5)
            line_current, = ax.plot([], [], 'r-', label='Current', linewidth=1.5)
            line_error, = ax.plot([], [], 'g-', label='Error', linewidth=1, alpha=0.7)

            self.lines_target.append(line_target)
            self.lines_current.append(line_current)
            self.lines_error.append(line_error)

            ax.set_xlabel('Time Steps', fontsize=10)
            ax.set_ylabel('Position (rad)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Set initial limits to avoid zero-size axes
            ax.set_xlim(0, 10)
            ax.set_ylim(-1, 1)

        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        plt.tight_layout(pad=2.0)

    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'right':
            self.current_joint_idx = min(self.current_joint_idx + self.joints_per_plot,
                                         self.num_joints - self.joints_per_plot)
        elif event.key == 'left':
            self.current_joint_idx = max(0, self.current_joint_idx - self.joints_per_plot)

    def update_data(self, target_pos, current_pos, timestamp=None):
        """
        Update the data.

        Args:
            target_pos: Target positions, shape (num_joints,).
            current_pos: Current positions, shape (num_joints,).
            timestamp: Optional timestamp.
        """
        with self.lock:
            if timestamp is None:
                timestamp = len(self.timestamps)

            self.timestamps.append(timestamp)

            for i in range(self.num_joints):
                self.target_positions[i].append(target_pos[i])
                self.current_positions[i].append(current_pos[i])
                error = target_pos[i] - current_pos[i]
                self.errors[i].append(error)

                # Update statistics
                self.max_error[i] = max(self.max_error[i], abs(error))
                if len(self.errors[i]) > 0:
                    self.mean_error[i] = np.mean(np.abs(list(self.errors[i])))

    def _update_plot(self, frame):
        """Update the plot (animation callback)."""
        with self.lock:
            if len(self.timestamps) == 0:
                return self.lines_target + self.lines_current + self.lines_error

            times = np.array(self.timestamps)

            for i in range(self.joints_per_plot):
                joint_idx = self.current_joint_idx + i

                if joint_idx >= self.num_joints:
                    # Hide unused subplots
                    self.axes[i].set_visible(False)
                    continue

                self.axes[i].set_visible(True)

                # Get data
                target = np.array(self.target_positions[joint_idx])
                current = np.array(self.current_positions[joint_idx])
                error = np.array(self.errors[joint_idx])

                # Update the lines
                self.lines_target[i].set_data(times, target)
                self.lines_current[i].set_data(times, current)
                self.lines_error[i].set_data(times, error)

                # Update the title, including statistics
                joint_name = self.joint_names[joint_idx] if joint_idx < len(self.joint_names) else f"Joint_{joint_idx}"
                title = f"{joint_name}\nMax Err: {self.max_error[joint_idx]:.4f} | Mean Err: {self.mean_error[joint_idx]:.4f}"
                self.axes[i].set_title(title, fontsize=10)

                # Automatically adjust the y-axis range
                if len(target) > 0:
                    all_data = np.concatenate([target, current, error])
                    y_min, y_max = all_data.min(), all_data.max()
                    margin = (y_max - y_min) * 0.1 + 0.01
                    self.axes[i].set_ylim(y_min - margin, y_max + margin)

                # Automatically adjust the x-axis range
                if len(times) > 0:
                    self.axes[i].set_xlim(times[0], times[-1])

        return self.lines_target + self.lines_current + self.lines_error

    def start(self):
        """Start visualization (blocking)."""
        self.anim = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )
        plt.show()

    def start_non_blocking(self):
        """Start visualization (nonblocking)."""
        self.anim = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )
        plt.ion()
        plt.show()

    def get_statistics(self):
        """Get statistics."""
        stats = {}
        for i in range(self.num_joints):
            joint_name = self.joint_names[i] if i < len(self.joint_names) else f"Joint_{i}"
            stats[joint_name] = {
                'max_error': self.max_error[i],
                'mean_error': self.mean_error[i],
            }
        return stats

    def print_statistics(self):
        """Print statistics."""
        print("\n" + "="*80)
        print("Joint Tracking Statistics")
        print("="*80)
        stats = self.get_statistics()
        for joint_name, stat in stats.items():
            print(f"{joint_name:20s} | Max Error: {stat['max_error']:8.4f} | Mean Error: {stat['mean_error']:8.4f}")
        print("="*80 + "\n")


# Usage example
if __name__ == "__main__":
    import time

    # Create the visualizer
    viz = JointPositionVisualizer(num_joints=29, window_size=200)

    # Start non-blocking visualization
    viz.start_non_blocking()

    # Simulate data updates
    print("Simulating joint tracking...")
    print("Press Ctrl+C to stop")

    try:
        t = 0
        while True:
            # Generate simulated data
            target = np.sin(t * 0.1 + np.arange(29) * 0.2)
            current = target + np.random.randn(29) * 0.05  # Add noise

            viz.update_data(target, current, timestamp=t)

            t += 1
            time.sleep(0.02)  # 50Hz

            # Print statistics every 100 steps
            if t % 100 == 0:
                viz.print_statistics()

    except KeyboardInterrupt:
        print("\nStopping...")
        viz.print_statistics()
        plt.close()
