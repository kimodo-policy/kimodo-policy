"""Replay logged observations through bridge dry-run; never publish to a robot.

Uses sampled feedback and diagnostic JPEGs, not a closed-loop simulator.
Resets the model server session; do not run beside a live deployment.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import run_kimodo_sonic as bridge
from kimodo_sonic_bridge import MUJOCO_TO_ISAACLAB, REALWORLD_HAND_CLOSE_POSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--legacy-lead", action="store_true",
                        help="Use old scheduling settings for offline comparisons")
    parser.add_argument("--sonic-lookahead-frames", type=int, choices=(0,46), default=0)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    def seeded_reset(client, session_id=None):
        result=client._post("/reset", {"session_id":session_id,"seed":args.seed})
        if result.get("status") != "ok": raise RuntimeError(result)
    bridge.KimodoHttpClient.reset = seeded_reset
    if (args.output / "events.jsonl").exists():
        parser.error("output already contains events; choose a new directory")
    rows = [json.loads(line) for line in (args.run / "events.jsonl").read_text().splitlines()]
    start = rows[0]["monotonic"]
    controls = [row for row in rows if row["event"] == "control" and row.get("state_valid")]
    times = [row["monotonic"] - start for row in controls]
    frames = []
    for row in rows:
        if row["event"] == "submit":
            path = args.run / "images" / f"request-{row['deployment']['request_id']:06d}.jpg"
            if path.exists():
                with Image.open(path) as image:
                    frames.append((row["monotonic"] - start, np.asarray(image.convert("RGB"))))
    if not controls or not frames:
        parser.error("recording needs measured controls and request images")
    if args.steps <= 0 or args.steps / 50 > times[-1]:
        parser.error(f"steps must be positive and no greater than {int(times[-1] * 50)}")
    frame_times = [frame[0] for frame in frames]
    inverse_order = np.argsort(MUJOCO_TO_ISAACLAB)

    def recorded_state(step):
        timestamp = step / 50
        control = controls[max(0, bisect.bisect_right(times, timestamp) - 1)]
        image = frames[max(0, bisect.bisect_right(frame_times, timestamp) - 1)][1]
        closure = control.get("measured_hand")
        if closure is None:
            raise ValueError("recording is missing measured hand closure")
        # Reconstruct the bridge's two closure scalars, not fourteen original angles.
        hands = np.asarray(closure)[:, None] * REALWORLD_HAND_CLOSE_POSES
        state = dict(body_q=np.asarray(control["measured_q"])[inverse_order],
                     body_dq=np.asarray(control["measured_dq"])[inverse_order],
                     base_quat=np.asarray(control["measured_root_quat"]),
                     left_hand_q=hands[0], right_hand_q=hands[1], index=step)
        return state, image

    bridge._synthetic_state = recorded_state
    cap, initial, margin = (30, .25, .04) if args.legacy_lead else (32, .50, .08)
    sys.argv = ["replay_session", "--task", "teleop-banana", "--dry-run",
                "--sonic-lookahead-frames", str(args.sonic_lookahead_frames), "--max-reference-xy-radius", "0", "--max-steps", str(args.steps), "--diagnostics-dir", str(args.output),
                "--request-when-queue-below", str(cap),
                "--initial-inference-latency-s", str(initial),
                "--inference-lead-margin-s", str(margin), "--verbose"]
    bridge.main()


if __name__ == "__main__":
    main()
