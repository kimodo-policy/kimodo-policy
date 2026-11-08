#!/usr/bin/env python3
"""
Joint position visualization integration module.
Can be integrated into any simulation script.
"""

import torch
import numpy as np
import sys
import os

# Fix Qt plugin conflict with OpenCV
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

print("[JointVizIntegration] Importing visualizer...")
try:
    from tools.joint_position_visualizer import JointPositionVisualizer
    print("[JointVizIntegration] Visualizer imported successfully")
except Exception as e:
    print(f"[JointVizIntegration] ERROR: Failed to import visualizer: {e}")
    print(f"[JointVizIntegration] sys.path: {sys.path[:3]}")
    raise


class JointVisualizerIntegration:
    """Visualizer wrapper for integration into a simulation loop."""

    def __init__(self, robot, env_id=0, window_size=200, update_frequency=2):
        """
        Args:
            robot: Isaac Lab robot object.
            env_id: ID of the environment to visualize.
            window_size: Sliding window size.
            update_frequency: Update every N steps.
        """
        print(f"[JointVizIntegration] Initializing with robot: {type(robot)}")
        print(f"[JointVizIntegration] Robot num_joints: {robot.num_joints}")

        self.robot = robot
        self.env_id = env_id
        self.update_frequency = update_frequency
        self.step_count = 0

        # Create the visualizer on the main thread
        print(f"[JointVizIntegration] Creating visualizer in main thread...")
        try:
            self.visualizer = JointPositionVisualizer(
                num_joints=robot.num_joints,
                window_size=window_size,
                update_interval=50
            )
            print(f"[JointVizIntegration] Visualizer created successfully")
        except Exception as e:
            print(f"[JointVizIntegration] ERROR creating visualizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Update joint names
        if hasattr(robot.data, 'joint_names'):
            self.visualizer.joint_names = robot.data.joint_names
            print(f"[JointVizIntegration] Updated joint names: {len(robot.data.joint_names)} joints")

        # Start non-blocking mode without a thread; update on the main thread
        print(f"[JointVizIntegration] Starting visualizer in non-blocking mode...")
        try:
            self.visualizer.start_non_blocking()
            print(f"[JointVizIntegration] Visualizer started successfully")
        except Exception as e:
            print(f"[JointVizIntegration] ERROR starting visualizer: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"\n{'='*80}")
        print("Joint Position Visualizer Started")
        print(f"{'='*80}")
        print(f"Monitoring {robot.num_joints} joints from environment {env_id}")
        print("Controls:")
        print("  - Left/Right Arrow: Navigate between joint groups")
        print("  - Close window or Ctrl+C to stop")
        print(f"{'='*80}\n")

    def update(self):
        """Call this method from the simulation loop."""
        self.step_count += 1

        if self.step_count % self.update_frequency == 0:
            # Get target and current positions
            target_pos = self.robot.data.joint_pos_target[self.env_id].cpu().numpy()
            current_pos = self.robot.data.joint_pos[self.env_id].cpu().numpy()

            self.visualizer.update_data(target_pos, current_pos, timestamp=self.step_count)

    def print_statistics(self):
        """Print statistics."""
        self.visualizer.print_statistics()

    def get_statistics(self):
        """Get statistics."""
        return self.visualizer.get_statistics()


# Usage example:
# Add the following code to sim_main.py:
#
# from tools.joint_visualizer_integration import JointVisualizerIntegration
#
# # After creating the environment:
# if args.visualize_joints:
#     joint_viz = JointVisualizerIntegration(env.scene["robot"], env_id=0)
#
# # In the main loop:
# while simulation_app.is_running():
#     obs, reward, terminated, truncated, info = env.step(actions)
#
#     if args.visualize_joints:
#         joint_viz.update()
#
#     # Print statistics every 1000 steps
#     if args.visualize_joints and step_count % 1000 == 0:
#         joint_viz.print_statistics()
