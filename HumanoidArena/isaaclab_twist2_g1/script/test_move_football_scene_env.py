#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Move-football G1 29DOF Dex3 wholebody scene visualization script.

Based on `test_visual_zones_env.py`, this quickly checks whether the
`move_football_g1_29dof_dex3_wholebody` scene can be constructed and rendered.

Usage (from the project root):
    cd isaaclab_twist2_g1

    # Test startup only (without visual rendering)
    python scripts/test_move_football_scene_env.py --headless --device cuda

    # Enable rendering (requires a display; omit --headless)
    python scripts/test_move_football_scene_env.py --device cuda

    # Set a custom step count (for example, 500)
    python scripts/test_move_football_scene_env.py --device cuda --num_steps 500

    # Run without a step limit until Ctrl+C
    python scripts/test_move_football_scene_env.py --device cuda --no_limit
"""

import argparse
import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

# Parse arguments
parser = argparse.ArgumentParser(
    description="Visualize move_football_g1_29dof_dex3_wholebody scene"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help="Number of simulation steps (ignored if --no_limit)",
)
parser.add_argument(
    "--no_limit",
    action="store_true",
    help="Run until Ctrl+C, no step limit",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments",
)

# Add IsaacLab launcher args (includes --device, --headless, etc.)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import IsaacLab modules
from isaaclab.envs import ManagerBasedRLEnv

# Import env cfg for the football task
from tasks.g1_tasks.move_football_g1_29dof_dex3_wholebody.move_football_g1_29dof_dex3_hw_env_cfg import (
    MoveFootballG129Dex3WholebodyEnvCfg,
)
from tools.grass_ground_material import apply_grass_pbr_to_ground
from tools.pitch_lines import create_simple_debug_lines
from tools.football_physics_material import apply_football_physics_material


def main():
    """Visualize move_football_g1_29dof_dex3_wholebody scene."""

    print("=" * 60)
    print("Move-Football G1-29DOF Dex3 Wholebody Scene Visualization")
    print("=" * 60)

    # Create environment configuration
    print("\n[1] Creating environment configuration...")
    try:
        env_cfg = MoveFootballG129Dex3WholebodyEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        # Sync AppLauncher --device with sim to avoid using fabric CPU instead of fabric GPU
        sim_device = getattr(args, "device", "cuda")
        if sim_device == "cuda":
            sim_device = "cuda:0"
        env_cfg.sim.device = sim_device
        print("   ✓ Configuration created")
        print(f"   - Num envs: {env_cfg.scene.num_envs}")
        print(f"   - Sim device: {env_cfg.sim.device} (fabric will use the GPU)")
        print(f"   - Decimation: {env_cfg.decimation}")
        print(f"   - dt: {env_cfg.sim.dt}")
    except Exception as e:
        print(f"   ✗ Failed to create configuration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Create environment
    print("\n[2] Creating environment...")
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("   ✓ Environment created")
        print("   [foot_collision] skipped live collision rebuild before reset to avoid invalidating PhysX tensor views")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Check scene contents
    print("\n[3] Checking scene contents...")
    try:
        scene_keys = list(env.scene.keys())
        print(f"   Scene contains {len(scene_keys)} elements:")
        for key in scene_keys:
            print(f"      - {key}")

        # Try listing a few key elements if present
        interesting = [
            k
            for k in scene_keys
            if any(
                substr in k.lower()
                for substr in ["robot", "object", "football", "table", "camera"]
            )
        ]
        print(f"\n   Interesting elements (robot/ball/table/camera): {len(interesting)}")
        for elem in interesting:
            print(f"      - {elem}")

        print("   ✓ Scene contents verified")
    except Exception as e:
        print(f"   ✗ Failed to check scene: {e}")
        import traceback

        traceback.print_exc()

    # Reset and run simulation (use only sim.step to inspect the scene and physics)
    print("\n[4] Running simulation (scene visualization)...")
    try:
        env.reset()
        print("   ✓ Environment reset")
        foot_log_path = env_cfg.log_foot_collision_status()
        print(f"   [foot_collision] validation logged to: {foot_log_path}")
        # Apply the grass PBR material after reset
        try:
            apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(150.0, 150.0))
        except Exception as e:
            print(f"   [grass] Texture unavailable or skipped: {e}")
        try:
            apply_football_physics_material(restitution=0.75)
        except Exception as e:
            print(f"   [football_physics] Skipped: {e}")
        # Simplified test markings: a 7.32 m line in front of the goal and a 2 m circle around the robot
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            create_simple_debug_lines(
                stage,
                axis_mode="z_up",
                line_color=(127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0),
            )
        except Exception as e:
            print(f"   [pitch_lines] Field markings unavailable or skipped: {e}")
        if args.no_limit:
            print("   Running until Ctrl+C (no step limit)...")

        step = 0
        try:
            while True:
                # Drive the simulation directly and render the current football scene
                env.sim.step(render=True)

                if args.no_limit:
                    if step % 100 == 0:
                        print(f"   Step {step}: running... (Ctrl+C to exit)")
                elif step % 20 == 0:
                    print(
                        f"   Step {step:4d}: move-football scene simulation running..."
                    )

                step += 1
                if not args.no_limit and step >= args.num_steps:
                    break
        except KeyboardInterrupt:
            if args.no_limit:
                print("\n   Stopped by user (Ctrl+C).")
            else:
                raise

        if not args.no_limit:
            print(f"   ✓ Completed {args.num_steps} simulation steps")
    except KeyboardInterrupt:
        print("\n   Stopped by user (Ctrl+C).")
        return 0
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Cleanup
    print("\n[5] Cleaning up...")
    try:
        env.close()
        print("   ✓ Environment closed")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Move-Football Scene Visualization: PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()

    sys.exit(exit_code)
