#!/usr/bin/env python3
"""Run the composed camera server with a single RealSense ego camera."""

from __future__ import annotations

import argparse

from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    config = ComposedCameraConfig(
        ego_view_camera="realsense",
        fps=args.fps,
        port=args.port,
    )
    camera = ComposedCameraSensor(config)
    try:
        camera.run_server()
    finally:
        camera.close()


if __name__ == "__main__":
    main()
