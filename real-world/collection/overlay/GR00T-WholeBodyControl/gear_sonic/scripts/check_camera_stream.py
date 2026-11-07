#!/usr/bin/env python3
"""Read-only, bounded validation of the camera stream used by G1 collection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor  # noqa: E402


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return result


def check_stream(args: argparse.Namespace) -> None:
    client = ComposedCameraClientSensor(server_ip=args.host, port=args.port)
    startup_deadline = time.monotonic() + args.startup_timeout
    first_received = last_received = None
    last_sequence = last_source_time = None
    count = 0
    max_gap = 0.0
    shape = None
    print(f"Checking {args.host}:{args.port} for {args.duration:g}s of live frames")
    try:
        while True:
            now = time.monotonic()
            if first_received is not None:
                if now - last_received > 1.0:
                    raise ValueError("camera stream stalled for more than 1 second")
                if now - first_received >= args.duration:
                    break
            elif now >= startup_deadline:
                raise ValueError(f"no camera frames within {args.startup_timeout:g}s")

            message = client.read(blocking=False)
            if not message or not message.get("is_new", False):
                time.sleep(0.001)
                continue

            frame = message.get("images", {}).get("ego_view")
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
                or min(frame.shape[:2]) <= 0
            ):
                raise ValueError("ego_view must be a decoded uint8 HxWx3 image")

            sequence = message.get("sequence")
            source_time = message.get("server_monotonic_timestamp")
            received = message.get("receive_monotonic_timestamp")
            image_time = message.get("timestamps", {}).get("ego_view")
            if not isinstance(sequence, (int, np.integer)) or sequence < 0:
                raise ValueError("missing or invalid camera sequence")
            for name, value in (
                ("source monotonic timestamp", source_time),
                ("receive monotonic timestamp", received),
                ("ego_view timestamp", image_time),
            ):
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"missing or invalid {name}")

            if last_sequence is not None:
                if sequence <= last_sequence or source_time <= last_source_time:
                    raise ValueError("camera sequence/source timestamp did not advance")
                if received <= last_received:
                    raise ValueError("local receive timestamp did not advance")
                max_gap = max(max_gap, received - last_received)
            else:
                first_received = received
            count += 1
            last_sequence, last_source_time = sequence, source_time
            last_received, shape = received, frame.shape
    finally:
        client.close()

    elapsed = now - first_received
    fps = (count - 1) / elapsed
    print(
        f"frames={count} unique_fps={fps:.2f} max_gap_ms={max_gap * 1000:.1f} "
        f"shape={shape} elapsed_s={elapsed:.2f}"
    )
    if count < 2 or not args.expected_fps * 0.9 <= fps <= args.expected_fps * 1.1:
        raise ValueError(f"camera rate must be within 10% of {args.expected_fps:g} Hz")
    print("PASS camera stream")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-host", "--host", dest="host", required=True)
    parser.add_argument("--camera-port", "--port", dest="port", type=int, default=5555)
    parser.add_argument("--duration", type=positive_float, default=10.0)
    parser.add_argument("--startup-timeout", type=positive_float, default=15.0)
    parser.add_argument("--expected-fps", type=positive_float, default=30.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        check_stream(args)
    except KeyboardInterrupt:
        print("FAIL camera check interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAIL camera stream: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
