#!/usr/bin/env python3

"""
Enhanced data collection script for TWIST2+IsaacLab.

Collects data from Redis and IsaacLab simulation, including:
- Front camera: RGB + Depth (first-person view)
- World camera: RGB + Depth (third-person view)
- SMPLX motion data (human body poses)
- Body and hand state
- Body and hand action

Usage:
    python server_data_record_with_third_smplx_qpos.py \
        --robot_ip 192.168.123.164 \
        --task_name my_demo_task \
        --frequency 30
"""

import argparse
import os
import json
import time
import redis
import cv2
import numpy as np
from multiprocessing import shared_memory, Array, Lock
import threading
from data_utils.episode_writer import EpisodeWriter
from data_utils.vision_client import VisionClient
from rich import print
from robot_control.speaker import Speaker
from datetime import datetime


class MultiCameraVisionClient:
    """Manage visual clients for multiple cameras."""

    def __init__(self, robot_ip, num_cameras=2):
        """
        Args:
            robot_ip: Robot IP address
            num_cameras: Number of cameras (default: 2, front + world)
        """
        self.robot_ip = robot_ip
        self.num_cameras = num_cameras
        self.running = True

        # Front camera (first-person view).
        self.front_rgb_shape = (360, 640, 3)
        self.front_depth_shape = (360, 640)

        # World camera (third-person view) - optional.
        self.world_rgb_shape = (360, 640, 3)
        self.world_depth_shape = (360, 640)

        # Create shared memory for front camera (always required)
        print("[MultiCameraVisionClient] Creating shared memory for front camera...")
        self.front_rgb_shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(self.front_rgb_shape) * np.uint8().itemsize)
        )
        self.front_rgb_array = np.ndarray(
            self.front_rgb_shape,
            dtype=np.uint8,
            buffer=self.front_rgb_shm.buf
        )

        self.front_depth_shm = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(self.front_depth_shape) * np.float32().itemsize)
        )
        self.front_depth_array = np.ndarray(
            self.front_depth_shape,
            dtype=np.float32,
            buffer=self.front_depth_shm.buf
        )

        # Try to create shared memory for world camera (optional)
        self.world_camera_available = False
        self.world_rgb_shm = None
        self.world_depth_shm = None
        self.world_rgb_array = None
        self.world_depth_array = None
        self.world_client = None
        self.world_thread = None

        try:
            print("[MultiCameraVisionClient] Creating shared memory for world camera...")
            self.world_rgb_shm = shared_memory.SharedMemory(
                create=True,
                size=int(np.prod(self.world_rgb_shape) * np.uint8().itemsize)
            )
            self.world_rgb_array = np.ndarray(
                self.world_rgb_shape,
                dtype=np.uint8,
                buffer=self.world_rgb_shm.buf
            )

            self.world_depth_shm = shared_memory.SharedMemory(
                create=True,
                size=int(np.prod(self.world_depth_shape) * np.float32().itemsize)
            )
            self.world_depth_array = np.ndarray(
                self.world_depth_shape,
                dtype=np.float32,
                buffer=self.world_depth_shm.buf
            )
            print("[MultiCameraVisionClient] World camera shared memory created successfully")
        except Exception as e:
            print(f"[MultiCameraVisionClient] Warning: Could not create world camera shared memory: {e}")
            print("[MultiCameraVisionClient] Will try to connect anyway...")

        print(f"[MultiCameraVisionClient] Shared memory initialized:")
        print(f"  Front RGB:   {self.front_rgb_shm.name}")
        print(f"  Front Depth: {self.front_depth_shm.name}")
        if self.world_rgb_shm is not None:
            print(f"  World RGB:   {self.world_rgb_shm.name}")
            print(f"  World Depth: {self.world_depth_shm.name}")

        # Create vision clients
        # Port 5555: Front camera (always required)
        self.front_client = VisionClient(
            server_address=robot_ip,
            port=5555,
            img_shape=self.front_rgb_shape,
            img_shm_name=self.front_rgb_shm.name,
            depth_shape=self.front_depth_shape,
            depth_shm_name=self.front_depth_shm.name,
            image_show=False,
            depth_show=False,
            unit_test=False
        )

    # Port 5556: optional world camera (third-person view).
        if self.world_rgb_shm is not None:
            try:
                print("[MultiCameraVisionClient] Attempting to connect to world camera on port 5556...")
                self.world_client = VisionClient(
                    server_address=robot_ip,
                    port=5556,
                    img_shape=self.world_rgb_shape,
                    img_shm_name=self.world_rgb_shm.name,
                    depth_shape=self.world_depth_shape,
                    depth_shm_name=self.world_depth_shm.name,
                    image_show=False,
                    depth_show=False,
                    unit_test=False
                )
                self.world_camera_available = True
                print("[MultiCameraVisionClient] ✅ World camera connection initialized")
            except Exception as e:
                print(f"[MultiCameraVisionClient] ⚠️ Warning: Could not connect to world camera: {e}")
                print("[MultiCameraVisionClient] Continuing with front camera only...")
                self.world_camera_available = False

        # Start threads
        self.front_thread = threading.Thread(
            target=self.front_client.receive_process,
            daemon=True
        )
        self.front_thread.start()

        if self.world_camera_available and self.world_client is not None:
            self.world_thread = threading.Thread(
                target=self.world_client.receive_process,
                daemon=True
            )
            self.world_thread.start()
            print("[MultiCameraVisionClient] Vision threads started (front + world)")
        else:
            print("[MultiCameraVisionClient] Vision thread started (front only)")

        # Give threads time to initialize
        time.sleep(0.5)

    def get_front_images(self):
        """Get the first-person camera image."""
        return {
            'rgb': self.front_rgb_array.copy(),
            'depth': self.front_depth_array.copy()
        }

    def get_world_images(self):
        """Get the third-person camera image (None if unavailable)."""
        if not self.world_camera_available or self.world_rgb_array is None:
            return None
        return {
            'rgb': self.world_rgb_array.copy(),
            'depth': self.world_depth_array.copy()
        }

    def get_all_images(self):
        """Get images from all cameras."""
        result = {
            'front': self.get_front_images()
        }

        # Only include world camera if available
        world_images = self.get_world_images()
        if world_images is not None:
            result['world'] = world_images

        return result

    def cleanup(self):
        """Clean up shared memory."""
        try:
            # Unlink and close front camera shared memory
            self.front_rgb_shm.unlink()
            self.front_rgb_shm.close()
            self.front_depth_shm.unlink()
            self.front_depth_shm.close()

            # Unlink and close world camera shared memory (if exists)
            if self.world_rgb_shm is not None:
                self.world_rgb_shm.unlink()
                self.world_rgb_shm.close()
            if self.world_depth_shm is not None:
                self.world_depth_shm.unlink()
                self.world_depth_shm.close()

            print("[MultiCameraVisionClient] Shared memory cleaned up")
        except Exception as e:
            print(f"[MultiCameraVisionClient] Error during cleanup: {e}")


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
        # Test connection
        redis_client.ping()
        print(f"✅ Connected to Redis at localhost:6379, DB=0")
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")
        return

    # Initialize multi-camera vision client
    print("\n" + "="*60)
    print("Initializing multi-camera vision client...")
    print("="*60)

    vision_manager = MultiCameraVisionClient(
        robot_ip=args.robot_ip,
        num_cameras=2
    )

    # Give some time for vision threads to connect
    time.sleep(1.0)

    # Create recorder with enhanced data keys
    recording = False
    save_data_keys = [
        'front_rgb',        # First-person RGB image.
        'front_depth',      # First-person depth image.
        'world_rgb',        # Third-person RGB image.
        'world_depth',      # Third-person depth image.
        'smplx_data',       # SMPL-X body pose.
    ]

    task_dir = os.path.join(args.data_folder, args.task_name)

    # Note: image_shape is used for display, but we store each camera separately
    display_shape = (360, 640*2, 3)  # Concatenated view for display

    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=display_shape,
        data_keys=save_data_keys
    )

    recorder.text_desc(
        goal="Perform humanoid teleoperation task",
        desc="TWIST2+IsaacLab teleoperation with multi-camera and SMPLX data recording.",
        steps="Record multi-view RGB-D and SMPLX motion data for imitation learning."
    )

    control_dt = 1 / args.frequency
    step_count = 0
    running = True

    print("\n" + "="*60)
    print(f"Recording Configuration:")
    print(f"  Task directory: {task_dir}")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Data keys: {save_data_keys}")
    print("="*60)

    speaker = Speaker()

    # Initialize button state tracking
    prev_button_pressed = False

    # Display window
    image_show = True

    try:
        while running:
            start_time = time.time()

            # Handle controller input
            try:
                controller_data_raw = redis_client.get(f"controller_data")
                if controller_data_raw is None:
                    print("[Warning] No controller data from Redis")
                    time.sleep(0.1)
                    continue

                controller_data = json.loads(controller_data_raw)
                button_pressed = controller_data['LeftController']['key_two']
                quit_key = controller_data['LeftController']['axis_click']

                if quit_key:
                    running = False
                    speaker.speak("Recording stopped.")
                    print("\n🛑 Quitting...")
                    break

                # Detect button press (rising edge detection)
                if button_pressed and not prev_button_pressed:
                    print("🔘 Button pressed")
                    recording = not recording
                    if recording:
                        speaker.speak("episode recording started.")
                        if not recorder.create_episode():
                            recording = False
                        step_count = 0
                        print("🔴 Episode recording started...")
                    else:
                        recorder.save_episode()
                        speaker.speak("episode saved.")
                        print("💾 Episode saved!")

                # Update previous button state
                prev_button_pressed = button_pressed

            except Exception as e:
                print(f"[Warning] Error reading controller data: {e}")
                time.sleep(0.1)
                continue

            if recording:
                # Create data dictionary
                data_dict = {'idx': step_count}

                # 1. Get vision data from all cameras
                try:
                    all_images = vision_manager.get_all_images()

                    # Front camera (first-person view) - always required
                    data_dict["front_rgb"] = all_images['front']['rgb']
                    data_dict["front_depth"] = all_images['front']['depth']

                    # World camera (third-person view) - optional.
                    if 'world' in all_images:
                        data_dict["world_rgb"] = all_images['world']['rgb']
                        data_dict["world_depth"] = all_images['world']['depth']
                    else:
                        # World camera not available, set to None
                        data_dict["world_rgb"] = None
                        data_dict["world_depth"] = None

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
                        print("[Warning] No SMPLX data from Redis")
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
                                print(f"[Warning] Failed to decode JSON for key {redis_keys[i]}")
                                data_dict[dict_key] = None
                        else:
                            # Don't print warning for every missing key to reduce spam
                            data_dict[dict_key] = None

                except Exception as e:
                    print(f"[Error] Redis pipeline operation: {e}")
                    continue

                # Write data to recorder
                try:
                    recorder.add_item(data_dict)
                except Exception as e:
                    print(f"[Error] Failed to add item to recorder: {e}")
                    continue

                # Display concatenated view (front + world if available)
                if image_show:
                    try:
                        # Get front camera image (always available)
                        front_rgb = all_images['front']['rgb']

                        # Check if world camera is available
                        if 'world' in all_images:
                            world_rgb = all_images['world']['rgb']

                            # Ensure same height
                            if front_rgb.shape[0] != world_rgb.shape[0]:
                                world_rgb = cv2.resize(
                                    world_rgb,
                                    (world_rgb.shape[1], front_rgb.shape[0])
                                )

                            display_image = np.concatenate([front_rgb, world_rgb], axis=1)
                        else:
                            # Only front camera available
                            display_image = front_rgb

                        # Add text overlay
                        display_image = display_image.copy()
                        cv2.putText(
                            display_image,
                            f"Front Camera",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # Only show world camera label if it exists
                        if 'world' in all_images:
                            cv2.putText(
                                display_image,
                                f"World Camera",
                                (650, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                        cv2.putText(
                            display_image,
                            f"Recording... Step: {step_count}",
                            (10, 340),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

                        window_name = "Multi-Camera Recording (Press controller button to stop)"
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
                        all_images = vision_manager.get_all_images()
                        front_rgb = all_images['front']['rgb']

                        # Check if world camera is available
                        if 'world' in all_images:
                            world_rgb = all_images['world']['rgb']

                            if front_rgb.shape[0] != world_rgb.shape[0]:
                                world_rgb = cv2.resize(
                                    world_rgb,
                                    (world_rgb.shape[1], front_rgb.shape[0])
                                )

                            display_image = np.concatenate([front_rgb, world_rgb], axis=1)
                        else:
                            # Only front camera available
                            display_image = front_rgb

                        display_image = display_image.copy()

                        cv2.putText(
                            display_image,
                            f"Front Camera",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # Only show world camera label if it exists
                        if 'world' in all_images:
                            cv2.putText(
                                display_image,
                                f"World Camera",
                                (650, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                        cv2.putText(
                            display_image,
                            f"Idle - Press controller button to start recording",
                            (10, 340),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2
                        )

                        window_name = "Multi-Camera Recording (Press controller button to start)"
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
        print(f"\n✅ Done! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        # Cleanup
        vision_manager.cleanup()
        recorder.close()
        cv2.destroyAllWindows()

        print("🏁 Exiting the recording...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Record multi-view RGB-D + SMPLX data from TWIST2+IsaacLab"
    )

    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser.add_argument(
        "--data_folder",
        default="/home/ANT.AMAZON.COM/yanjieze/projects/TWIST2/TWIST2-clean/deploy_real/twist2_demonstration_smplx",
        help="Data folder for recordings"
    )
    parser.add_argument(
        "--task_name",
        default=f"smplx_multiview_{cur_time}",
        help="Task name (used as subfolder)"
    )
    parser.add_argument(
        "--frequency",
        default=30,
        type=int,
        help="Recording frequency (Hz)"
    )
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=["unitree_g1"],
        help="Robot name"
    )
    parser.add_argument(
        "--robot_ip",
        default="192.168.123.164",
        help="Robot IP address for vision server"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("TWIST2+IsaacLab Enhanced Data Recorder")
    print("="*70)
    print(f"Recording multi-view RGB-D + SMPLX motion data")
    print(f"Task: {args.task_name}")
    print(f"Frequency: {args.frequency} Hz")
    print(f"Robot IP: {args.robot_ip}")
    print("="*70 + "\n")

    main(args)
