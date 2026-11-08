# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Command-line utility for converting URDF to USD.

URDF (Unified Robot Description Format) is the XML format used by ROS to describe robot components.
For more information, see: http://wiki.ros.org/urdf

This script uses the Isaac Sim URDF importer extension (``isaacsim.asset.importer.urdf``) to convert URDF assets to USD.
This convenience script runs from the command line. For details, see the URDF importer extension documentation:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html

Arguments:
  input               Path to the input URDF file.
  output              Path to the output USD file.

Optional arguments:
  -h, --help          Show help.
  --merge-joints      Merge links connected by fixed joints (default: False).
  --fix-base          Fix the base at the import position (default: False).
  --joint-stiffness   Joint drive stiffness (default: 100.0).
  --joint-damping     Joint drive damping (default: 1.0).
  --joint-target-type Joint drive control type (default: "position").
"""

"""Launch the Isaac Sim simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Add command-line argument parsing
parser = argparse.ArgumentParser(description="Convert URDF to USD")
# parser.add_argument("input", type=str, default="/home/unitree/newDisk/URDF/urdf-to-usd/g1withdex1/g1_29dof_with_dex1_rev_1_0.urdf", help="Path to the input URDF file")
# parser.add_argument("output", type=str, default="/home/unitree/Code/isaaclab_demo/usd/g1_body29_hand14.usd", help="Path to the output USD file")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Merge links connected by fixed joints",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base at the import position")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="Joint drive stiffness",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="Joint drive damping",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="Joint drive control type",
)

# Add AppLauncher command-line arguments
AppLauncher.add_app_launcher_args(parser)
# Parse command-line arguments
args_cli = parser.parse_args()

# Launch the Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Main implementation follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():
    # Check whether the input file path is valid
    urdf_path = "/home/unitree/newDisk/URDF/urdf-to-usd/g1withinspire_hand/g1_29dof_with_inpire_rev_1_0.urdf" #args_cli.input
    print(urdf_path)
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    
    # Create the output file path
    dest_path = "/home/unitree/newDisk/URDF/wholevody_with_inspire/g1_29dof_with_inspire_rev_1_0.usd"
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create the URDF importer configuration
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    # Print configuration information
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer configuration:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create the URDF importer and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    # Print output information
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    # Check whether the GUI needs an update:
    # Get the settings interface
    carb_settings_iface = carb.settings.get_settings()
    # Read the flag indicating whether the local GUI is enabled
    local_gui = carb_settings_iface.get("/app/window/enabled")
    # Read the flag indicating whether the live GUI is enabled
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Run the simulated scene if the GUI is enabled
    if local_gui or livestream_gui:
        # Open the USD scene
        stage_utils.open_stage(urdf_converter.usd_path)
        # Reinitialize the simulation
        app = omni.kit.app.get_app_interface()
        # Run the simulation
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                # Run a simulation step
                app.update()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulation application
    simulation_app.close()
