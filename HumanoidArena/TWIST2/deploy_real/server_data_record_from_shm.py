#!/usr/bin/env python3

"""
Enhanced data collection script for TWIST2+IsaacLab.
Reads directly from IsaacLab shared memory (no network needed).

Collects data from Redis and IsaacLab simulation, including:
- Front camera: RGB + Depth (first-person view)
- World camera: RGB + Depth (third-person view)
- SMPLX motion data (human body poses)
- Body and hand state
- Body and hand action

Usage:
    python server_data_record_from_shm.py \
        --task_name my_demo_task \
        --frequency 30
"""

import argparse
import os
import sys
import json
import time
import redis
import cv2
import numpy as np
from multiprocessing import shared_memory
import threading
from rich import print

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.episode_writer import EpisodeWriter
from robot_control.speaker import Speaker
from datetime import datetime

# Import IsaacLab's shared memory reader
sys.path.insert(0, "/home/yixiao/Users/yixiao/temp/isaaclab_twist2_g1")
from image_server.shared_memory_utils import MultiImageReader


class IsaacLabVisionClient:
    """Read multiple camera images directly from IsaacLab shared memory."""

    def __init__(self, shm_name="isaac_multi_image_shm"):
        """
        Args:
            shm_name: IsaacLab shared-memory name
        """
        self.shm_name = shm_name
        self.reader = None
        self.running = True

        # Image shapes (will be determined from shared memory)
        self.image_shape = (480, 640, 3)  # Default, will update
        self.depth_shape = (480, 640)

        # Initialize reader
        try:
            self.reader = MultiImageReader(shm_name=shm_name)
            print(f"[IsaacLabVisionClient] ✅ Connected to IsaacLab shared memory: {shm_name}")
        except FileNotFoundError:
            print(f"[IsaacLabVisionClient] ❌ Error: Shared memory '{shm_name}' not found!")
            print(f"[IsaacLabVisionClient] Please make sure IsaacLab is running.")
            raise
        except Exception as e:
            print(f"[IsaacLabVisionClient] ❌ Error opening shared memory: {e}")
            raise

        # Test read to get image dimensions
        test_images = self.reader.read_images()
        if test_images and 'head' in test_images:
            head_img = test_images['head']
            self.image_shape = head_img.shape
            self.depth_shape = head_img.shape[:2]
            print(f"[IsaacLabVisionClient] Image shape: {self.image_shape}")

        self.world_camera_available = False
        if test_images and 'world' in test_images:
            self.world_camera_available = True
            print(f"[IsaacLabVisionClient] ✅ World camera available")
        else:
            print(f"[IsaacLabVisionClient] ⚠️  World camera not available")

    def get_images(self):
        """
        Read all camera images and metric depth data.

        Returns:
            dict: {
                'front': {'rgb': np.array, 'depth': np.array},
                'world': {'rgb': np.array, 'depth': np.array} or None
            }
        """
        if self.reader is None:
            return None

        try:
            # Read all images and depth maps from shared memory
            images = self.reader.read_images()

            if images is None or 'head' not in images:
                return None

            result = {}

            # Front camera (head)
            front_rgb = images['head']
            # IsaacLab uses BGR, so convert the image to RGB
            front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_BGR2RGB)

            # Get real depth data from IsaacLab (distance_to_image_plane)
            front_depth = images.get('head_depth', None)
            if front_depth is None:
                print("[IsaacLabVisionClient] Warning: No depth data for front camera, using placeholder")
                front_depth = np.zeros((front_rgb.shape[0], front_rgb.shape[1]), dtype=np.float32)

            # Resize to the target dimensions (360, 640)
            if front_rgb.shape[0] != 360 or front_rgb.shape[1] != 640:
                front_rgb = cv2.resize(front_rgb, (640, 360))
                front_depth = cv2.resize(front_depth, (640, 360))

            result['front'] = {
                'rgb': front_rgb,
                'depth': front_depth
            }

            # World camera (if available)
            if 'world' in images:
                world_rgb = images['world']
                world_rgb = cv2.cvtColor(world_rgb, cv2.COLOR_BGR2RGB)

                # Get real depth data
                world_depth = images.get('world_depth', None)
                if world_depth is None:
                    print("[IsaacLabVisionClient] Warning: No depth data for world camera, using placeholder")
                    world_depth = np.zeros((world_rgb.shape[0], world_rgb.shape[1]), dtype=np.float32)

                # Resize the image
                if world_rgb.shape[0] != 360 or world_rgb.shape[1] != 640:
                    world_rgb = cv2.resize(world_rgb, (640, 360))
                    world_depth = cv2.resize(world_depth, (640, 360))

                result['world'] = {
                    'rgb': world_rgb,
                    'depth': world_depth
                }

            return result

        except Exception as e:
            print(f"[IsaacLabVisionClient] Error reading images: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """Close the connection."""
        self.running = False
        if self.reader:
            self.reader.close()
        print("[IsaacLabVisionClient] Closed")


def main(args):

    # Connect to Redis with connection pool for better performance
    try:
        redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        redis_client.ping()
        print("✅ Connected to Redis")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return

    # Initialize vision client (directly from shared memory)
    try:
        vision_manager = IsaacLabVisionClient()
        print("✅ Connected to IsaacLab shared memory")
    except Exception as e:
        print(f"❌ Failed to connect to IsaacLab shared memory: {e}")
        print("Please make sure IsaacLab is running!")
        return

    # Initialize speaker
    try:
        speaker = Speaker()
        speaker_available = True
        print("✅ Speaker initialized (audio will play on local machine)")
    except Exception as e:
        speaker_available = False
        print(f"⚠️  Speaker initialization failed: {e}")
        print("   Continuing without audio feedback (visual feedback only)")

    # Create task directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = args.task_name if args.task_name else f"demo_{timestamp}"
    task_dir = os.path.join(args.data_folder, task_name)
    os.makedirs(task_dir, exist_ok=True)
    print(f"💾 Data will be saved to: {task_dir}")

    # Initialize episode writer
    # Support multiple cameras with RGB and depth: front_rgb, world_rgb, front_depth, world_depth
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=(360, 640, 3),  # Single camera shape
        data_keys=['front_rgb', 'world_rgb', 'front_depth', 'world_depth']  # RGB + Depth
    )

    recorder.text_desc(
        "Multi-camera recording with SMPLX data\n"
        f"Task: {task_name}\n"
        f"Frequency: {args.frequency} Hz\n"
        f"Cameras: Front (first-person) saved to front_rgb/, "
        f"World (third-person) saved to world_rgb/ (if available)\n"
    )

    print("\n" + "="*60)
    print("🎮 CONTROLS:")
    print("  Left Controller 'key_two': Start/Stop recording episode")
    print("  Left Controller 'axis_click': Exit program")
    print("="*60 + "\n")

    # Main loop variables
    recording = False
    step_count = 0
    control_dt = 1.0 / args.frequency
    prev_button_pressed = False
    running = True

    # Always show display window (like original script)
    image_show = True

    print("\n" + "="*60)
    print("📹 Camera Display Window: ENABLED")
    print("="*60)

    try:
        while running:
            start_time = time.time()

            # Check controller state
            try:
                # Try new key name first
                controller_raw = redis_client.get("controller_data")
                if not controller_raw:
                    # Fallback to old key name
                    controller_raw = redis_client.get("teleop_controller_unitree_g1_with_hands")

                button_pressed = False
                axis_click_pressed = False

                if controller_raw:
                    controller_state = json.loads(controller_raw)

                    # Handle nested structure (controller_data format)
                    if "LeftController" in controller_state:
                        left_controller = controller_state["LeftController"]
                        button_pressed = left_controller.get("key_two", False)
                        axis_click_pressed = left_controller.get("axis_click", False)
                    else:
                        # Handle flat structure (old format)
                        button_pressed = controller_state.get("key_two", False)
                        axis_click_pressed = controller_state.get("axis_click", False)

                    if axis_click_pressed:
                        print("\n🛑 Exit button pressed, stopping...")
                        running = False
                        break

                    # Toggle recording on button press (edge detection)
                    if button_pressed and not prev_button_pressed:
                        print(f"[Controller] ✅ Button press detected! Toggling recording...")
                        recording = not recording

                        if recording:
                            # Try to create episode with retry
                            max_retries = 50  # Wait up to 5 seconds (50 * 0.1s)
                            retry_count = 0
                            while retry_count < max_retries:
                                if recorder.create_episode():
                                    if speaker_available:
                                        try:
                                            speaker.speak("recording started.")
                                        except Exception as e:
                                            print(f"[Warning] Speaker error: {e}")
                                    step_count = 0
                                    print("🔴 Episode recording started...")
                                    break
                                else:
                                    # Recorder is still busy saving previous episode
                                    retry_count += 1
                                    if retry_count == 1:
                                        print("⏳ Waiting for previous episode to finish saving...")
                                    time.sleep(0.1)

                            if retry_count >= max_retries:
                                print("❌ Failed to create episode after waiting. Please try again.")
                                recording = False
                        else:
                            recorder.save_episode()
                            print("💾 Episode save initiated (processing in background)...")
                            if speaker_available:
                                try:
                                    speaker.speak("episode saved.")
                                except Exception as e:
                                    print(f"[Warning] Speaker error: {e}")

                    # Update previous button state
                    prev_button_pressed = button_pressed
                # else: controller_raw is None, just continue to display

            except Exception as e:
                print(f"[Warning] Error reading controller data: {e}")
                import traceback
                traceback.print_exc()
                # Don't sleep here, just continue to display

            if recording:
                # Create data dictionary
                data_dict = {'idx': step_count}

                # 1. Get vision data from IsaacLab shared memory
                try:
                    all_images = vision_manager.get_images()

                    if all_images is None:
                        print("[Warning] No images available")
                        time.sleep(0.01)
                        continue

                    # Front camera (always required) - save RGB and depth separately
                    front_rgb = all_images['front']['rgb']
                    front_depth = all_images['front']['depth']
                    # Convert RGB to BGR for storage (OpenCV format)
                    data_dict["front_rgb"] = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
                    # Save depth as float32 numpy array
                    data_dict["front_depth"] = front_depth

                    # World camera (optional) - save RGB and depth separately if available
                    if 'world' in all_images:
                        world_rgb = all_images['world']['rgb']
                        world_depth = all_images['world']['depth']
                        data_dict["world_rgb"] = cv2.cvtColor(world_rgb, cv2.COLOR_RGB2BGR)
                        data_dict["world_depth"] = world_depth

                        # Get joint keypoints for world camera from Redis
                        try:
                            keypoints_raw = redis_client.get("world_camera_joint_keypoints")
                            if keypoints_raw is not None:
                                keypoints_2d = json.loads(keypoints_raw)
                                data_dict["world_camera_joint_keypoints"] = keypoints_2d
                                print(f"[DEBUG] world_camera_joint_keypoints: {len(keypoints_2d)} joints")
                            else:
                                data_dict["world_camera_joint_keypoints"] = None
                                print(f"[DEBUG] world_camera_joint_keypoints: None")
                        except Exception as e:
                            print(f"[Warning] Failed to get joint keypoints: {e}")
                            data_dict["world_camera_joint_keypoints"] = None
                    else:
                        # World camera not available, set to None
                        # EpisodeWriter will handle None gracefully
                        data_dict["world_rgb"] = None
                        data_dict["world_depth"] = None
                        data_dict["world_camera_joint_keypoints"] = None

                    data_dict["t_img"] = int(time.time() * 1000)  # timestamp in ms

                except Exception as e:
                    print(f"[Warning] Error getting vision data: {e}")
                    continue

                # 2. Get SMPLX data from Redis
                try:
                    smplx_data_raw = redis_client.get("smplx_data_unitree_g1_with_hands")
                    if smplx_data_raw is not None:
                        data_dict["smplx_data"] = json.loads(smplx_data_raw)
                    else:
                        data_dict["smplx_data"] = None
                except Exception as e:
                    print(f"[Warning] Error reading SMPLX data: {e}")
                    data_dict["smplx_data"] = None

                # 3. Get state and action data from Redis (pipeline for efficiency)
                redis_keys = [
                    "state_body_unitree_g1_with_hands",
                    "state_hand_left_unitree_g1_with_hands",
                    "state_hand_right_unitree_g1_with_hands",
                    "state_neck_unitree_g1_with_hands",
                    "t_state",

                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                    "t_action",
                ]

                data_dict_keys = [
                    "state_body",
                    "state_hand_left",
                    "state_hand_right",
                    "state_neck",
                    "t_state",

                    "action_body",
                    "action_hand_left",
                    "action_hand_right",
                    "action_neck",
                    "t_action",
                ]

                try:
                    # Use Redis pipeline to batch all GET operations
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()

                    # Process results with error handling
                    for i, (result, dict_key) in enumerate(zip(redis_results, data_dict_keys)):
                        if result is not None:
                            try:
                                data_dict[dict_key] = json.loads(result)
                            except json.JSONDecodeError:
                                data_dict[dict_key] = None
                        else:
                            data_dict[dict_key] = None

                except Exception as e:
                    print(f"[Error] Redis pipeline operation: {e}")
                    continue

                # Add to recorder
                try:
                    recorder.add_item(data_dict)
                except Exception as e:
                    print(f"[Error] Failed to add item to recorder: {e}")
                    continue

                # Display concatenated view (front + world if available, with depth)
                if image_show:
                    try:
                        # Get front camera image and depth (always available)
                        front_rgb = all_images['front']['rgb']
                        front_depth = all_images['front']['depth']

                        # Normalize depth to 0-255 for visualization (using fixed range 0-5m like D435i)
                        # Store real depth stats for display
                        front_depth_min = front_depth.min()
                        front_depth_max = front_depth.max()
                        front_depth_mean = front_depth.mean()

                        # Clip to 0-5m range and normalize to 0-255
                        front_depth_normalized = np.clip(front_depth, 0, 5.0)
                        front_depth_normalized = (front_depth_normalized / 5.0 * 255).astype(np.uint8)
                        front_depth_colored = cv2.applyColorMap(front_depth_normalized, cv2.COLORMAP_JET)
                        front_depth_colored = cv2.cvtColor(front_depth_colored, cv2.COLOR_BGR2RGB)

                        # Check if world camera is available
                        if 'world' in all_images:
                            world_rgb = all_images['world']['rgb']
                            world_depth = all_images['world']['depth']

                            # Ensure same height
                            if front_rgb.shape[0] != world_rgb.shape[0]:
                                world_rgb = cv2.resize(
                                    world_rgb,
                                    (world_rgb.shape[1], front_rgb.shape[0])
                                )
                            if front_depth_colored.shape[0] != world_depth.shape[0]:
                                world_depth = cv2.resize(
                                    world_depth,
                                    (world_depth.shape[1], front_depth_colored.shape[0])
                                )

                            # Normalize world depth
                            world_depth_normalized = world_depth.copy()
                            if world_depth_normalized.max() > 0:
                                world_depth_normalized = (world_depth_normalized / world_depth_normalized.max() * 255).astype(np.uint8)
                            else:
                                world_depth_normalized = np.zeros_like(world_depth, dtype=np.uint8)
                            world_depth_colored = cv2.applyColorMap(world_depth_normalized, cv2.COLORMAP_JET)
                            world_depth_colored = cv2.cvtColor(world_depth_colored, cv2.COLOR_BGR2RGB)

                            # Create a copy for display with keypoints
                            world_rgb_display = world_rgb.copy()

                            # Draw keypoints on display copy (not on saved image)
                            keypoints = data_dict.get("world_camera_joint_keypoints", None)
                            if keypoints is not None:
                                num_visible = 0
                                for kp in keypoints:
                                    if kp is not None:  # Skip out-of-view joints
                                        u, v = int(kp[0]), int(kp[1])
                                        # Draw red circle for each joint (RGB format: 255,0,0 = red)
                                        cv2.circle(world_rgb_display, (u, v), radius=2, color=(255, 0, 0), thickness=-1)
                                        num_visible += 1
                                print(f"[DEBUG] Recording: Drew {num_visible}/{len(keypoints)} visible joints on world camera")
                            else:
                                print(f"[DEBUG] Recording: No keypoints available")

                            # Create 2x2 grid: [RGB row] / [Depth row]
                            rgb_row = np.concatenate([front_rgb, world_rgb_display], axis=1)
                            depth_row = np.concatenate([front_depth_colored, world_depth_colored], axis=1)
                            display_image = np.concatenate([rgb_row, depth_row], axis=0)
                        else:
                            # Only front camera available: 2x1 grid
                            display_image = np.concatenate([front_rgb, front_depth_colored], axis=0)

                        # Convert RGB to BGR for OpenCV display
                        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                        # Camera labels for RGB images (top row)
                        cv2.putText(
                            display_image,
                            "Front RGB",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

                        # Depth labels (bottom row)
                        cv2.putText(
                            display_image,
                            "Front Depth",
                            (10, 390),  # 360 + 30
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        # Only show world camera labels if it exists
                        if 'world' in all_images:
                            cv2.putText(
                                display_image,
                                "World RGB",
                                (650, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            cv2.putText(
                                display_image,
                                "World Depth",
                                (650, 390),  # 360 + 30
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                        # Recording status bar
                        status_y = display_image.shape[0] - 50
                        cv2.rectangle(
                            display_image,
                            (0, status_y),
                            (display_image.shape[1], display_image.shape[0]),
                            (0, 0, 0),
                            -1
                        )
                        cv2.rectangle(
                            display_image,
                            (0, status_y),
                            (display_image.shape[1], display_image.shape[0]),
                            (0, 0, 255),
                            2
                        )

                        # Status text
                        status_text = f"🔴 RECORDING - Episode: {recorder.episode_id + 1}, Frame: {step_count}, FPS: {1/(time.time()-start_time):.1f}"
                        cv2.putText(
                            display_image,
                            status_text,
                            (10, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )

                        # Control hints
                        cv2.putText(
                            display_image,
                            "Press 'key_two' to stop | 'axis_click' to exit",
                            (display_image.shape[1] - 500, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        window_name = "TWIST2 Data Recording - Multi-Camera View"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, display_image)
                        cv2.waitKey(1)

                    except Exception as e:
                        print(f"[Warning] Display error: {e}")

                step_count += 1

                # Print status every 30 frames
                if step_count % 30 == 0:
                    print(f"📊 Recording... Step: {step_count}, FPS: {1/(time.time()-start_time):.1f}")

                # Maintain frequency
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                # Not recording, just display
                if image_show:
                    try:
                        all_images = vision_manager.get_images()

                        if all_images is None:
                            print("[Debug] get_images() returned None, waiting for data...")
                            time.sleep(0.1)
                            continue

                        if not all_images:
                            print("[Debug] get_images() returned empty dict, waiting for data...")
                            time.sleep(0.1)
                            continue

                        if 'front' not in all_images:
                            print("[Debug] No 'front' camera in images, waiting...")
                            time.sleep(0.1)
                            continue

                        front_rgb = all_images['front']['rgb']
                        front_depth = all_images['front']['depth']

                        # Normalize depth to 0-255 for visualization (using colormap)
                        front_depth_normalized = front_depth.copy()
                        if front_depth_normalized.max() > 0:
                            front_depth_normalized = (front_depth_normalized / front_depth_normalized.max() * 255).astype(np.uint8)
                        else:
                            front_depth_normalized = np.zeros_like(front_depth, dtype=np.uint8)
                        front_depth_colored = cv2.applyColorMap(front_depth_normalized, cv2.COLORMAP_JET)
                        front_depth_colored = cv2.cvtColor(front_depth_colored, cv2.COLOR_BGR2RGB)

                        # Check if world camera is available
                        if 'world' in all_images:
                            world_rgb = all_images['world']['rgb']
                            world_depth = all_images['world']['depth']

                            if front_rgb.shape[0] != world_rgb.shape[0]:
                                world_rgb = cv2.resize(
                                    world_rgb,
                                    (world_rgb.shape[1], front_rgb.shape[0])
                                )
                            if front_depth_colored.shape[0] != world_depth.shape[0]:
                                world_depth = cv2.resize(
                                    world_depth,
                                    (world_depth.shape[1], front_depth_colored.shape[0])
                                )

                            # Normalize world depth
                            world_depth_normalized = world_depth.copy()
                            if world_depth_normalized.max() > 0:
                                world_depth_normalized = (world_depth_normalized / world_depth_normalized.max() * 255).astype(np.uint8)
                            else:
                                world_depth_normalized = np.zeros_like(world_depth, dtype=np.uint8)
                            world_depth_colored = cv2.applyColorMap(world_depth_normalized, cv2.COLORMAP_JET)
                            world_depth_colored = cv2.cvtColor(world_depth_colored, cv2.COLOR_BGR2RGB)

                            # Create a copy for display with keypoints
                            world_rgb_display = world_rgb.copy()

                            # Draw keypoints on display copy (fetch from Redis in idle mode)
                            try:
                                keypoints_raw = redis_client.get("world_camera_joint_keypoints")
                                if keypoints_raw is not None:
                                    keypoints = json.loads(keypoints_raw)
                                    num_visible = 0
                                    for kp in keypoints:
                                        if kp is not None:  # Skip out-of-view joints
                                            u, v = int(kp[0]), int(kp[1])
                                            # Draw red circle for each joint (RGB format: 255,0,0 = red)
                                            cv2.circle(world_rgb_display, (u, v), radius=2, color=(255, 0, 0), thickness=-1)
                                            num_visible += 1
                                    print(f"[DEBUG] Idle: Drew {num_visible}/{len(keypoints)} visible joints on world camera")
                                else:
                                    print(f"[DEBUG] Idle: No keypoints in Redis")
                            except Exception as e:
                                print(f"[Warning] Idle: Failed to draw keypoints: {e}")

                            # Create 2x2 grid: [RGB row] / [Depth row]
                            rgb_row = np.concatenate([front_rgb, world_rgb_display], axis=1)
                            depth_row = np.concatenate([front_depth_colored, world_depth_colored], axis=1)
                            display_image = np.concatenate([rgb_row, depth_row], axis=0)
                        else:
                            # Only front camera available: 2x1 grid
                            display_image = np.concatenate([front_rgb, front_depth_colored], axis=0)

                        # Convert RGB to BGR for OpenCV display
                        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                        # Camera labels for RGB images (top row)
                        cv2.putText(
                            display_image,
                            "Front RGB",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

                        # Depth labels (bottom row)
                        cv2.putText(
                            display_image,
                            "Front Depth",
                            (10, 390),  # 360 + 30
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        # Only show world camera labels if it exists
                        if 'world' in all_images:
                            cv2.putText(
                                display_image,
                                "World RGB",
                                (650, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            cv2.putText(
                                display_image,
                                "World Depth",
                                (650, 390),  # 360 + 30
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                        # Idle status bar (green)
                        status_y = display_image.shape[0] - 50
                        cv2.rectangle(
                            display_image,
                            (0, status_y),
                            (display_image.shape[1], display_image.shape[0]),
                            (0, 0, 0),
                            -1
                        )
                        cv2.rectangle(
                            display_image,
                            (0, status_y),
                            (display_image.shape[1], display_image.shape[0]),
                            (0, 255, 0),
                            2
                        )

                        # Status text
                        status_text = f"⏸️  IDLE - Total Episodes: {recorder.episode_id + 1}"
                        cv2.putText(
                            display_image,
                            status_text,
                            (10, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

                        # Control hints
                        cv2.putText(
                            display_image,
                            "Press 'key_two' to start recording | 'axis_click' to exit",
                            (display_image.shape[1] - 600, status_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        window_name = "TWIST2 Data Recording - Multi-Camera View"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, display_image.shape[1], display_image.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, display_image)
                        cv2.waitKey(1)

                    except Exception as e:
                        print(f"[Warning] Display error: {e}")
                else:
                    # For non-display mode, just sleep
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⌨️  Received Ctrl+C, exiting...")
        running = False
    finally:
        print("\n🧹 Cleaning up...")

        cv2.destroyAllWindows()

        # Close vision manager
        vision_manager.close()

        # Close recorder
        recorder.close()

        # Close Redis
        redis_client.close()

        print(f"\n✅ Done! Recorded {recorder.episode_id + 1} episodes to {task_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record multi-camera data from IsaacLab (shared memory)")

    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Task name for organizing data"
    )

    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="Recording frequency (Hz)"
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default="twist2_demonstration_smplx",
        help="Root folder for saving data"
    )

    args = parser.parse_args()

    main(args)
