#!/usr/bin/env python3
"""
Test SMPLX and qpos visualization

Use these tests to inspect data correctness.
"""

import sys
import os

# Add the project path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_reader import EpisodeReader


def test_smplx_visualization():
    """Test SMPLX visualization."""
    print("\n" + "="*60)
    print("Test 1: SMPLX skeleton visualization")
    print("="*60)

    # Load episode data.
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # Visualize the SMPL-X skeleton at frame 100.
    print("\nVisualizing the SMPLX skeleton at frame 100...")
    reader.visualize_smplx(
        idx=100,
        save_path="test_output/smplx_frame_100.png",
        show=False  # Save without displaying a window.
    )

    # Visualize frame 500.
    print("Visualizing the SMPLX skeleton at frame 500...")
    reader.visualize_smplx(
        idx=500,
        save_path="test_output/smplx_frame_500.png",
        show=False
    )

    print("\n✅ SMPLX visualization test complete")


def test_qpos_visualization():
    """Test qpos visualization."""
    print("\n" + "="*60)
    print("Test 2: Joint angle (qpos) visualization")
    print("="*60)

    # Load episode data.
    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # Visualize qpos for the first 300 frames.
    print("\nVisualizing joint angles for the first 300 frames...")
    reader.visualize_qpos(
        start_frame=0,
        end_frame=300,
        save_path="test_output/qpos_0_300.png",
        show=False
    )

    # Visualize a 300-frame segment from the middle.
    print("Visualizing joint angles for the middle 300 frames...")
    reader.visualize_qpos(
        start_frame=500,
        end_frame=800,
        save_path="test_output/qpos_500_800.png",
        show=False
    )

    print("\n✅ Qpos visualization test complete")


def test_data_inspection():
    """Inspect basic dataset information."""
    print("\n" + "="*60)
    print("Test 3: Data inspection")
    print("="*60)

    episode_path = "/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001"
    reader = EpisodeReader(episode_path)

    # Print detailed information.
    reader.print_info()

    # Check SMPL-X data.
    print("\nChecking SMPLX data...")
    smplx = reader.get_smplx_data(100)
    if smplx:
        print(f"  ✓ SMPLX joints: {len(smplx)}")
        print(f"  ✓ Joint names: {list(smplx.keys())[:5]}...")
    else:
        print("  ✗ No SMPLX data")

    # Check qpos data.
    print("\nChecking qpos data...")
    state_body = reader.get_state_body(100)
    if state_body:
        print(f"  ✓ Body joints: {len(state_body)}")
        print(f"  ✓ Value range: [{min(state_body):.3f}, {max(state_body):.3f}]")
    else:
        print("  ✗ No body state data")

    state_hand_left = reader.get_state_hand(100, 'left')
    if state_hand_left:
        print(f"  ✓ Left-hand joints: {len(state_hand_left)}")
    else:
        print("  ✗ No left-hand data")

    state_hand_right = reader.get_state_hand(100, 'right')
    if state_hand_right:
        print(f"  ✓ Right-hand joints: {len(state_hand_right)}")
    else:
        print("  ✗ No right-hand data")

    print("\n✅ Data inspection complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SMPLX and qpos visualization")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "smplx", "qpos", "inspect"],
        help="Select the test to run"
    )

    args = parser.parse_args()

    # Create the output directory.
    os.makedirs("test_output", exist_ok=True)

    try:
        if args.test == "all":
            test_smplx_visualization()
            test_qpos_visualization()
            test_data_inspection()
        elif args.test == "smplx":
            test_smplx_visualization()
        elif args.test == "qpos":
            test_qpos_visualization()
        elif args.test == "inspect":
            test_data_inspection()

        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        print("\nView the generated images:")
        print("  - test_output/smplx_frame_*.png")
        print("  - test_output/qpos_*.png")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
