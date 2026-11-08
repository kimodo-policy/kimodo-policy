#!/usr/bin/env python3

"""
EpisodeReader usage examples

Demonstrates how to use EpisodeReader to:
1. Read episode data
2. Generate videos
3. Visualize joint keypoints
4. Access the recorded data fields
"""

import sys
import os

# Add the project path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_reader import EpisodeReader


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*60)
    print("Example 1: Basic usage")
    print("="*60)

    # Load episode data.
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # Print detailed information.
    reader.print_info()

    # Get the total number of frames.
    print(f"Total frames: {len(reader)}")

    # Get the first frame.
    frame_0 = reader.get_frame(0)
    print(f"\nFirst-frame data keys: {list(frame_0.keys())}")


def example_create_videos():
    """Video generation example."""
    print("\n" + "="*60)
    print("Example 2: Generate videos")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a video from the front camera.
    if reader.has_front_cam:
        print("\n1. Generating the front-camera video...")
        reader.create_video(
            output_path=f"{output_dir}/front_camera.mp4",
            camera="front",
            fps=30
        )

    # Generate a video from the world camera.
    if reader.has_world_cam:
        print("\n2. Generating the world-camera video...")
        reader.create_video(
            output_path=f"{output_dir}/world_camera.mp4",
            camera="world",
            fps=30
        )


def example_visualize_keypoints():
    """Joint keypoint visualization example."""
    print("\n" + "="*60)
    print("Example 3: Visualize joint keypoints")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Visualize joints on the world-camera view.
    if reader.has_world_cam:
        print("\nGenerating the world-camera video with joint keypoints...")
        reader.visualize_keypoints_on_world_cam(
            output_path=f"{output_dir}/world_with_keypoints.mp4",
            keypoint_radius=3,
            keypoint_color=(255, 0, 0),  # Red (RGB).
            show_frame_number=True,
            fps=30
        )


def example_access_data():
    """Example of accessing different data fields."""
    print("\n" + "="*60)
    print("Example 4: Access data")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    frame_idx = 100

    # Get images.
    print(f"\n1. Get images for frame {frame_idx}")
    if reader.has_front_cam:
        front_img = reader.get_image(frame_idx, "front")
        print(f"   Front-camera image shape: {front_img.shape}")

    if reader.has_world_cam:
        world_img = reader.get_image(frame_idx, "world")
        print(f"   World-camera image shape: {world_img.shape}")

    # Get joints.
    print(f"\n2. Get joint keypoints")
    keypoints = reader.get_keypoints(frame_idx)
    if keypoints:
        visible_count = sum(1 for kp in keypoints if kp is not None)
        print(f"   Total keypoints: {len(keypoints)}")
        print(f"   Visible keypoints: {visible_count}")
        if keypoints[0] is not None:
            print(f"   First keypoint position: {keypoints[0]}")

    # Get SMPL-X data.
    print(f"\n3. Get SMPLX data")
    smplx_data = reader.get_smplx_data(frame_idx)
    if smplx_data:
        print(f"   SMPLX data keys: {list(smplx_data.keys())}")

    # Get states and actions.
    print(f"\n4. Get states and actions")
    state_body = reader.get_state_body(frame_idx)
    action_body = reader.get_action_body(frame_idx)
    if state_body:
        print(f"   Body state dimensions: {len(state_body)}")
    if action_body:
        print(f"   Body action dimensions: {len(action_body)}")

    state_hand_left = reader.get_state_hand(frame_idx, "left")
    state_hand_right = reader.get_state_hand(frame_idx, "right")
    if state_hand_left:
        print(f"   Left-hand state dimensions: {len(state_hand_left)}")
    if state_hand_right:
        print(f"   Right-hand state dimensions: {len(state_hand_right)}")


def example_partial_video():
    """Generate a video for part of an episode."""
    print("\n" + "="*60)
    print("Example 5: Generate a video for a selected frame range")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Generate a video from only the first 300 frames.
    start_frame = 0
    end_frame = 300

    print(f"\nGenerating video for frames {start_frame}-{end_frame}...")

    if reader.has_world_cam:
        reader.visualize_keypoints_on_world_cam(
            output_path=f"{output_dir}/world_partial_{start_frame}_{end_frame}.mp4",
            start_frame=start_frame,
            end_frame=end_frame,
            fps=30
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EpisodeReader usage examples")
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        choices=["all", "basic", "videos", "keypoints", "data", "partial"],
        help="Select the example to run"
    )

    args = parser.parse_args()

    try:
        if args.example == "all":
            example_basic_usage()
            example_create_videos()
            example_visualize_keypoints()
            example_access_data()
            example_partial_video()
        elif args.example == "basic":
            example_basic_usage()
        elif args.example == "videos":
            example_create_videos()
        elif args.example == "keypoints":
            example_visualize_keypoints()
        elif args.example == "data":
            example_access_data()
        elif args.example == "partial":
            example_partial_video()

        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
