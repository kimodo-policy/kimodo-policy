#!/usr/bin/env python3
"""Run Kimodo as the high-level policy and SONIC as the G1 low-level tracker.

Continuous-hand checkpoints receive measured closure history alongside the
64-D body state history so their hand delta predictions use the robot's actual
current hand state.
"""

from __future__ import annotations

import argparse
import os
import base64
import copy
import gzip
import http.client
import socket
from urllib.parse import urlsplit
import itertools
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("KIMODO_PROJECT_ROOT", THIS_DIR.parents[1])).expanduser().resolve()
SONIC_ROOT = Path(os.environ.get("WBC_ROOT", THIS_DIR.parent / "external/GR00T-WholeBodyControl")).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SONIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from kimodo_sonic_bridge import (  # noqa: E402
    BridgeSafetyError,
    KimodoActionRuntime,
    SafetyLimits,
    MUJOCO_TO_ISAACLAB,
    measured_hand_closure_from_state,
    pack_pose_v1,
    SonicPoseWindow,
    state_msg_to_kimodo_state,
    reference_to_boundary_action,
    rot6d_row_to_quat_wxyz,
)
from kimodo_rtc import (  # noqa: E402
    PROTOCOL_VERSION,
    BridgePlanTracker,
    deployment_identity,
    validate_observation_window,
)


DEFAULT_STANDING_JOINT_POS_MUJOCO = np.asarray(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
DEFAULT_STANDING_JOINT_POS_ISAACLAB = DEFAULT_STANDING_JOINT_POS_MUJOCO[MUJOCO_TO_ISAACLAB]
ELBOWS_90_STANDING_JOINT_POS_MUJOCO = DEFAULT_STANDING_JOINT_POS_MUJOCO.copy()
# The elbow pitch joints are indices 18 (left) and 25 (right) in SONIC's
# MuJoCo body_q order.  In the G1 URDF, 0 rad is the physical right-angle
# calibration pose: upper arms down and forearms pointing forward.  Using
# +pi/2 here rotates the forearms down and behind the torso.
ELBOWS_90_STANDING_JOINT_POS_MUJOCO[[18, 25]] = 0.0
ELBOWS_90_STANDING_JOINT_POS_ISAACLAB = (
    ELBOWS_90_STANDING_JOINT_POS_MUJOCO[MUJOCO_TO_ISAACLAB]
)
# G1's elbow encoder is zeroed at the calibration pose.  A negative 20-degree
# encoder offset raises the forearms from that pose, corresponding to the
# requested approximately 110-degree elbow configuration.
ELBOWS_110_STANDING_JOINT_POS_MUJOCO = DEFAULT_STANDING_JOINT_POS_MUJOCO.copy()
ELBOWS_110_STANDING_JOINT_POS_MUJOCO[[18, 25]] = np.deg2rad(-20.0)
ELBOWS_110_STANDING_JOINT_POS_ISAACLAB = (
    ELBOWS_110_STANDING_JOINT_POS_MUJOCO[MUJOCO_TO_ISAACLAB]
)
DEFAULT_STANDING_ROOT_POSITION = np.asarray([0.0, 0.0, 0.793], dtype=np.float32)


def _build_standing_reference(
    root_quat_wxyz: np.ndarray,
    frame_index: int,
    joint_pos_isaaclab: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    # SONIC applies its initial-heading alignment to the first streamed
    # reference quaternion.  Keep the bootstrap frame in that reference
    # frame (identity), rather than sending the measured world quaternion.
    # Sending the measured quaternion here and a Kimodo relative quaternion
    # after start mixes two frames and can create a 180-degree yaw jump.
    if joint_pos_isaaclab is None:
        joint_pos_isaaclab = DEFAULT_STANDING_JOINT_POS_ISAACLAB
    return {
        "joint_pos": np.asarray(joint_pos_isaaclab, dtype=np.float32).copy(),
        "joint_vel": np.zeros(29, dtype=np.float32),
        "body_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        # Match SONIC's measured pelvis height used by its normal INIT pose.
        # A zero streamed root position is interpreted as an actual target and
        # would pull the pelvis to the floor during a hold/fallback.
        "root_position": DEFAULT_STANDING_ROOT_POSITION.copy(),
        "hand_closure": np.zeros(2, dtype=np.float32),
        "hand_binary": np.zeros(2, dtype=np.float32),
        "frame_index": int(frame_index),
    }


from diagnostics import Recorder
from deployment_lifecycle import InitialPoseController, hold_last_reference


from image_wire import encode_image_bytes


class KimodoHttpClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.last_diagnostics = {}
        self._connections = threading.local()

    def _post(self, path: str, payload: dict) -> dict:
        encode_started = time.monotonic()
        body = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), compresslevel=1)
        encode_s = time.monotonic()-encode_started
        endpoint = urlsplit(self.base_url)
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
            connection = cls(endpoint.hostname, endpoint.port, timeout=self.timeout_s)
            self._connections.connection = connection
        try:
            if connection.sock is None:
                connection.connect()
                connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.request("POST", endpoint.path.rstrip('/') + '/' + path.lstrip('/'), body=body,
                               headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 400:
                raise RuntimeError(f"Kimodo HTTP {response.status}: {raw.decode('utf-8', errors='replace')}")
            result = json.loads(raw)
            self.last_diagnostics = dict(result.get("diagnostics", {}), client_encode_s=encode_s, request_bytes=len(body))
            return result
        except Exception:
            connection.close()
            self._connections.connection = None
            # Never retry a possibly committed inference automatically.
            raise

    def health(self) -> dict:
        request = urllib.request.Request(f"{self.base_url}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kimodo health HTTP {exc.code}: {detail}") from exc
        if result.get("status") != "ok":
            raise RuntimeError(f"Kimodo health check failed: {result}")
        return result

    def reset(self, session_id: str | None = None) -> None:
        result = self._post("/reset", {} if session_id is None else {"session_id": session_id})
        if result.get("status") != "ok":
            raise RuntimeError(f"Kimodo reset failed: {result}")

    def infer_chunk(
        self,
        image: np.ndarray,
        state: np.ndarray,
        state_history: np.ndarray,
        task: str,
        hand_closure_history: np.ndarray | None = None,
        *,
        deployment: dict | None = None,
        state_sequences: list[int] | None = None,
        state_timestamps: list[float] | None = None,
    ) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected uint8 HWC RGB image, got {image.shape} {image.dtype}")
        state = np.asarray(state, dtype=np.float32).reshape(64)
        history = np.asarray(state_history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 64 or history.shape[0] == 0:
            raise ValueError(f"expected state history [T,64], got {history.shape}")
        hand_history = None
        if hand_closure_history is not None:
            hand_history = np.asarray(hand_closure_history, dtype=np.float32)
            if hand_history.ndim == 1:
                hand_history = hand_history.reshape(1, -1)
            if (
                hand_history.ndim != 2
                or hand_history.shape[1] != 2
                or hand_history.shape[0] != history.shape[0]
            ):
                raise ValueError(
                    "expected hand closure history [T,2] matching state history, "
                    f"got {hand_history.shape} for state history {history.shape}"
                )
            if not np.isfinite(hand_history).all() or (
                (hand_history < -1e-6) | (hand_history > 1.0 + 1e-6)
            ).any():
                raise ValueError("hand closure history must be finite and within [0,1]")
        payload = {
            "task": str(task),
            "observation": {
                "images": {
                    "front": {
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "compression": "zlib",
                        "data_b64": encode_image_bytes(image),
                    }
                },
                "state": state.tolist(),
                "state_history": history.tolist(),
            },
        }
        if hand_history is not None:
            payload["observation"]["hand_closure_history"] = np.clip(
                hand_history, 0.0, 1.0
            ).tolist()
        if deployment is not None:
            payload["deployment"] = deployment
            payload["observation"]["state_sequences"] = state_sequences
            payload["observation"]["state_timestamps"] = state_timestamps
            validate_observation_window(payload["observation"], len(history))
        response = self._post("/infer", payload)
        if deployment is not None:
            if deployment_identity(response.get("deployment")) != deployment_identity(deployment):
                raise RuntimeError("Kimodo response deployment identity does not match request")
        action = response.get("action_chunk")
        if action is None:
            raise RuntimeError(f"Kimodo response has no action_chunk: {response.keys()}")
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] != 40 or action.shape[0] == 0:
            raise RuntimeError(f"Kimodo returned invalid action chunk shape {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError("Kimodo returned NaN/Inf action")
        return action


class AsyncKimodoWorker:
    """One outstanding inference request, keeping the 50 Hz loop non-blocking."""

    def __init__(self, client: KimodoHttpClient):
        self.client = client
        self.requests: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.busy = threading.Event()
        self._submit_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="kimodo-inference", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            inference_start = time.monotonic()
            try:
                infer_args = (
                    request["image"],
                    request["state"],
                    request["history"],
                    request["task"],
                )
                extra = {}
                if request.get("deployment") is not None:
                    extra = {key: request[key] for key in (
                        "deployment", "state_sequences", "state_timestamps",
                    )}
                if request.get("hand_closure_history") is not None:
                    extra["hand_closure_history"] = request["hand_closure_history"]
                action = self.client.infer_chunk(*infer_args, **extra)
                result = {
                    "ok": True,
                    "action": action,
                    "end_seq": request["end_seq"],
                    "start_seq": request["start_seq"],
                    "end_state_ts": request.get("end_state_ts"),
                    "episode_id": request.get("episode_id", 0),
                    "submitted_monotonic": request.get("submitted_monotonic"),
                }
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": exc,
                    "end_seq": request["end_seq"],
                    "start_seq": request["start_seq"],
                    "end_state_ts": request.get("end_state_ts"),
                    "episode_id": request.get("episode_id", 0),
                    "submitted_monotonic": request.get("submitted_monotonic"),
                }
            result["timing"] = dict(getattr(self.client, "last_diagnostics", {}))
            result["inference_elapsed_s"] = time.monotonic() - inference_start
            result["deployment"] = request.get("deployment")
            result["queue_at_submission"] = request.get("queue_at_submission", 0)
            try:
                self.results.put_nowait(result)
            except queue.Full:
                try:
                    self.results.get_nowait()
                except queue.Empty:
                    pass
                self.results.put_nowait(result)
            self.busy.clear()

    def submit(self, request: dict) -> bool:
        with self._submit_lock:
            if self.busy.is_set():
                return False
            self.busy.set()
            try:
                self.requests.put_nowait(request)
                return True
            except queue.Full:
                self.busy.clear()
                return False

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)


def _load_sonic_runtime():
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    return ComposedCameraClientSensor, ZMQStateSubscriber, build_command_message


def _normalize_hand_control_mode(value: str, *, allow_auto: bool = False) -> str:
    mode = str(value).strip().lower()
    allowed = {"binary", "continuous"}
    if allow_auto:
        allowed.add("auto")
    if mode not in allowed:
        choices = "'auto', 'binary', or 'continuous'" if allow_auto else "'binary' or 'continuous'"
        raise ValueError(f"hand_control_mode must be {choices}, got {mode!r}")
    return mode


def _resolve_bridge_hand_control_mode(requested_mode: str, health: dict) -> str:
    requested_mode = _normalize_hand_control_mode(requested_mode, allow_auto=True)
    if requested_mode != "auto":
        return requested_mode
    server_mode = health.get("hand_control_mode")
    if server_mode is None:
        return "binary"
    return _normalize_hand_control_mode(server_mode)


def _resolve_model_hand_observation_mode(health: dict) -> str:
    mode = health.get("hand_observation_mode")
    if mode is None:
        mode = health.get("hand_control_mode", "binary")
    mode = str(mode).strip().lower()
    if mode not in {"binary", "continuous"}:
        raise ValueError(
            "Kimodo /health hand_observation_mode must be 'binary' or "
            f"'continuous', got {mode!r}"
        )
    return mode


def _hand_close_poses() -> tuple[np.ndarray, np.ndarray]:
    try:
        from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver

        left_close = G1GripperInverseKinematicsSolver(side="left")._get_middle_close_q_desired().astype(np.float32)
        right_close = G1GripperInverseKinematicsSolver(side="right")._get_middle_close_q_desired().astype(np.float32)
    except Exception as exc:
        raise RuntimeError("SONIC hand solver is unavailable; refusing to silently drop hand action fields") from exc
    return left_close, right_close


def _hand_targets(
    hand_action: np.ndarray,
    hand_control_mode: str = "binary",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert Kimodo hand closure/binary commands to SONIC's 7D hand fields."""
    mode = _normalize_hand_control_mode(hand_control_mode)
    hand = np.asarray(hand_action, dtype=np.float32).reshape(-1)
    if hand.shape != (2,) or not np.isfinite(hand).all():
        raise ValueError(f"expected finite 2D hand action, got shape {hand.shape}")
    left_close, right_close = _hand_close_poses()
    if mode == "continuous":
        closure = np.clip(hand, 0.0, 1.0).astype(np.float32)
        return closure[0] * left_close, closure[1] * right_close
    left = left_close if float(hand[0]) >= 0.5 else np.zeros(7, dtype=np.float32)
    right = right_close if float(hand[1]) >= 0.5 else np.zeros(7, dtype=np.float32)
    return left, right


def _read_measured_hand_state(
    state_msg: dict,
) -> tuple[np.ndarray | None, BridgeSafetyError | None]:
    try:
        return measured_hand_closure_from_state(state_msg), None
    except BridgeSafetyError as exc:
        return None, exc


def _scalar(value, default=None) -> float | int | None:
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        return default
    try:
        result = float(arr[0])
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _integer(value, default: int | None = None) -> int | None:
    result = _scalar(value, default)
    if result is None:
        return default
    return int(result)


def _camera_metadata(image_msg: dict, receive_now: float) -> dict:
    return {
        "sequence": _integer(image_msg.get("sequence"), None),
        "source_ts": _scalar(
            image_msg.get("server_monotonic_timestamp"),
            _scalar(image_msg.get("receive_monotonic_timestamp"), receive_now),
        ),
        "receive_ts": _scalar(image_msg.get("receive_monotonic_timestamp"), receive_now),
        "is_new": bool(image_msg.get("is_new", True)),
    }


def _state_metadata(state_msg: dict, receive_now: float, fallback_seq: int) -> dict:
    source_seq = _integer(state_msg.get("index"), None)
    has_source_seq = source_seq is not None
    if source_seq is None:
        source_seq = fallback_seq
    source_ts = _scalar(state_msg.get("state_monotonic_timestamp"), None)
    has_source_ts = source_ts is not None and source_ts > 0.0
    if source_ts is None or source_ts <= 0.0:
        source_ts = receive_now
    return {
        "sequence": int(source_seq),
        "source_ts": float(source_ts),
        "receive_ts": receive_now,
        "has_source_seq": has_source_seq,
        "has_source_ts": has_source_ts,
    }


def _camera_age_and_sensor_skew(image_meta: dict, state_meta: dict, now: float) -> tuple[float, float]:
    camera_age = max(0.0, float(now) - float(image_meta["receive_ts"]))
    sensor_skew = abs(float(state_meta["receive_ts"]) - float(image_meta["receive_ts"]))
    return camera_age, sensor_skew


def _inference_is_stale(
    has_queued_actions: bool,
    last_inference_completed: float | None,
    now: float,
    max_age_s: float,
) -> bool:
    return (
        not has_queued_actions
        and last_inference_completed is not None
        and float(now) - float(last_inference_completed) > float(max_age_s)
    )


def _preview_actions(runtime, queued, candidate, measured_joints, root_quat):
    """Validate a whole candidate and recover its actual limited endpoint."""
    snapshot = runtime.snapshot()
    endpoint = None
    boundary_step = 0.0
    try:
        runtime.prime_joints(measured_joints)
        for index, action in enumerate(itertools.chain(queued, candidate)):
            previous = runtime.previous_joint_q.copy()
            reference = runtime.decode(action, current_root_quat_wxyz=root_quat)
            if index == len(queued):
                boundary_step = float(np.max(np.abs(reference["joint_pos"] - previous)))
            endpoint = reference_to_boundary_action(reference)
    finally:
        runtime.restore(snapshot)
    return endpoint, boundary_step


def _preview_reference_queue(runtime, actions, measured_joints, root_quat):
    snapshot = runtime.snapshot()
    try:
        runtime.prime_joints(measured_joints)
        return deque(runtime.decode(action, current_root_quat_wxyz=root_quat) for action in actions)
    finally:
        runtime.restore(snapshot)


def _admit_result(result, tracker, runtime, action_queue, *, now, latest_state_ts,
                  measured_joints, root_quat, max_state_lag_s, max_result_age_s,
                  max_start_age_s):
    """Admission is atomic: failed/stale candidates never partly enter the FIFO."""
    details = {"reason": "superseded plan"}
    if not tracker.matches(result.get("deployment")):
        return False, details
    age = now - float(result["submitted_monotonic"])
    tracker.observe_latency(age)
    state_lag = max(0.0, latest_state_ts - float(result["end_state_ts"]))
    first_age = age + len(action_queue) / tracker.control_fps
    details.update(age_s=age, state_lag_s=state_lag,
                   first_execution_age_s=first_age, queued_frames=len(action_queue),
                   consumed_frames=max(0, result.get("queue_at_submission", len(action_queue))
                                       - len(action_queue)))
    reason = None
    if not result["ok"]:
        reason = f"inference failed: {result['error']}"
    elif (not np.isfinite([age, state_lag, first_age]).all() or age < 0
          or state_lag > max_state_lag_s or age > max_result_age_s
          or first_age > max_start_age_s):
        reason = "stale Kimodo result"
    else:
        actions = np.asarray(result["action"], dtype=np.float32)
        if actions.shape != (tracker.chunk_frames, 40):
            reason = f"unexpected action chunk shape {actions.shape}"
        else:
            try:
                _, delta = _preview_actions(
                    runtime, action_queue, actions, measured_joints, root_quat,
                )
            except (BridgeSafetyError, ValueError) as exc:
                reason = f"invalid Kimodo action: {exc}"
            else:
                tracker.accept(result["deployment"])
                action_queue.extend(actions)
                details.update(reason="accepted", boundary_joint_step_rad=delta)
                return True, details
    tracker.invalidate()
    details["reason"] = reason
    return False, details


def _streamed_motion_should_start(
    *,
    state_valid: bool,
    standby: bool,
    auto_start: bool,
    policy_action_seen: bool,
) -> bool:
    """Start SONIC only after a valid state and, for auto-start, a policy action."""
    return bool(
        state_valid
        and (standby or (auto_start and policy_action_seen))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--task", required=True)
    parser.add_argument("--server-timeout", type=float, default=5.0)
    parser.add_argument("--sonic-lookahead-frames", type=int, choices=(0,46), default=0,
                        help="46 for installed g1 SONIC future window; 0 for legacy transport")
    parser.add_argument("--control-rate", type=float, default=50.0)
    parser.add_argument("--state-kp-host", "--state-zmq-host", dest="state_kp_host", default="127.0.0.1")
    parser.add_argument("--state-kp-port", "--state-zmq-port", dest="state_kp_port", type=int, default=5557)
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--action-kp-host", "--action-zmq-host", dest="action_kp_host", default="127.0.0.1")
    parser.add_argument("--action-kp-port", "--action-zmq-port", dest="action_kp_port", type=int, default=5556)
    parser.add_argument("--history-frames", type=int, default=0,
                        help="measured history window; 0 uses the server's bounded model context")
    parser.add_argument(
        "--request-when-queue-below",
        type=int,
        default=30,
        help="maximum request lead in control frames; also capped below one returned chunk",
    )
    parser.add_argument("--adaptive-request-lead", action=argparse.BooleanOptionalAction,
                        default=True, help="estimate request lead from recent inference latency")
    parser.add_argument("--initial-inference-latency-s", type=float, default=0.25)
    parser.add_argument("--inference-lead-margin-s", type=float, default=0.04)
    parser.add_argument("--max-root-delta-deg", type=float, default=26.0)
    parser.add_argument("--max-reference-xy-radius", type=float, default=10.0,
                        help="optional integrated reference radius in meters; 0 disables, not a measured workspace")
    parser.add_argument("--max-joint-step", type=float, default=0.75)
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument("--max-camera-stale-s", type=float, default=0.25)
    parser.add_argument("--max-state-gap-s", type=float, default=0.08)
    parser.add_argument(
        "--max-result-state-lag-s",
        type=float,
        default=1.00,
        help="discard inference results older than this state lag (default: 1.00s)",
    )
    parser.add_argument("--max-inference-age-s", type=float, default=1.00)
    parser.add_argument("--max-action-start-age-s", type=float, default=1.00,
                        help="maximum request age including remaining action queue time")
    parser.add_argument(
        "--hand-control-mode",
        choices=("auto", "binary", "continuous"),
        default="auto",
        help="hand action interpretation; auto follows Kimodo /health (default: auto)",
    )
    parser.add_argument(
        "--auto-stop-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send SONIC stop on controlled shutdown (default: true)",
    )
    standby_group = parser.add_mutually_exclusive_group()
    standby_group.add_argument(
        "--standby",
        action="store_true",
        help="start SONIC and hold the default standing pose before VLA closed-loop control",
    )
    standby_group.add_argument(
        "--standby-elbows-90",
        action="store_true",
        help="start SONIC in the G1 physical 90-degree elbow calibration pose",
    )
    standby_group.add_argument(
        "--standby-elbows-110",
        action="store_true",
        help="start SONIC with the forearms raised to approximately 110 degrees",
    )
    parser.add_argument("--initial-pose", default=None, help="Data-sourced pose JSON; prepare and wait for measured convergence before inference")
    parser.add_argument("--hold-on-interrupt", action=argparse.BooleanOptionalAction, default=True, help="Ctrl+C pauses task and continuously holds the last published reference")
    parser.add_argument("--auto-start", action="store_true", help="send command planner=false,start=true")
    parser.add_argument("--no-publish", action="store_true", help="run sensors/inference but do not publish to SONIC")
    parser.add_argument("--dry-run", action="store_true", help="use synthetic sensors; requires a reachable Kimodo server")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl-C")
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _synthetic_state(step: int) -> tuple[dict, np.ndarray]:
    return {
        "body_q": np.zeros(29, dtype=np.float32),
        "body_dq": np.zeros(29, dtype=np.float32),
        "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "left_hand_q": np.zeros(7, dtype=np.float32),
        "right_hand_q": np.zeros(7, dtype=np.float32),
        "index": step,
    }, np.full((96, 128, 3), 127, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    standby_mode = args.standby or args.standby_elbows_90 or args.standby_elbows_110
    standby_mode_name = (
        "--standby-elbows-110"
        if args.standby_elbows_110
        else "--standby-elbows-90"
        if args.standby_elbows_90
        else "--standby"
    )
    if standby_mode and args.auto_start:
        raise ValueError("a standby mode and --auto-start are mutually exclusive")
    if not np.isfinite(args.control_rate) or args.control_rate <= 0:
        raise ValueError("--control-rate must be finite and positive")
    if any(not np.isfinite(value) or value <= 0 for value in (
        args.max_result_state_lag_s, args.max_inference_age_s, args.max_action_start_age_s,
    )):
        raise ValueError("result and action age limits must be finite and positive")
    if args.history_frames < 0:
        raise ValueError("--history-frames must be non-negative")
    if args.initial_pose and standby_mode:
        raise ValueError("--initial-pose cannot be combined with a standby mode")
    if args.initial_pose and not (args.auto_start or args.dry_run or args.no_publish):
        raise ValueError("Initial pose preparation requires --auto-start for real motion")
    startup = InitialPoseController(args.initial_pose, task=args.task) if args.initial_pose else None
    startup_last_phase = None
    startup_last_report = 0.0
    period = 1.0 / float(args.control_rate)
    client = KimodoHttpClient(args.server_url, args.server_timeout)
    health = client.health()
    served_task = health.get('served_task')
    if served_task is not None and served_task != args.task:
        raise RuntimeError(
            f"Server serves {served_task!r}, but bridge requested {args.task!r}; restart the task-specific server"
        )
    if health.get("deployment_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Kimodo server does not support synchronized plans; restart the updated server first")
    server_fps = _scalar(health.get("control_fps"), None)
    if server_fps is None:
        raise RuntimeError("Kimodo /health did not report control_fps; refusing to run with an unverified rate")
    if abs(float(server_fps) - float(args.control_rate)) > 1e-3:
        raise RuntimeError(
            f"control-rate mismatch: bridge={args.control_rate:g}Hz, Kimodo server={float(server_fps):g}Hz"
        )
    hand_control_mode = _resolve_bridge_hand_control_mode(
        args.hand_control_mode, health
    )
    model_hand_control_mode = _normalize_hand_control_mode(
        health.get("hand_control_mode", "binary")
    )
    model_hand_observation_mode = _resolve_model_hand_observation_mode(health)
    measured_hand_required = model_hand_observation_mode == "continuous"
    chunk_frames = int(health["execution_control_frames"])
    tracker = BridgePlanTracker(
        control_fps=float(args.control_rate), chunk_frames=chunk_frames,
        queue_cap=int(args.request_when_queue_below),
        initial_latency_s=args.initial_inference_latency_s,
        margin_s=args.inference_lead_margin_s, adaptive=args.adaptive_request_lead,
    )
    if not np.isfinite(args.max_reference_xy_radius) or args.max_reference_xy_radius < 0:
        raise ValueError("max-reference-xy-radius must be finite and non-negative")
    client.reset(session_id=tracker.session_id)
    worker = AsyncKimodoWorker(client)
    runtime = KimodoActionRuntime(
        limits=SafetyLimits(
            max_joint_step=float(args.max_joint_step),
            max_root_xy_abs=float(args.max_reference_xy_radius),
            max_reference_age_s=min(0.25, float(args.max_inference_age_s)),
        ),
        max_root_delta_deg=float(args.max_root_delta_deg),
        reference_rate_hz=float(args.control_rate),
    )
    state_history: deque[tuple[int, np.ndarray, np.ndarray | None, float]] = deque(
        maxlen=min(int(args.history_frames) or int(health["history_control_frames"]),
                   int(health["history_control_frames"]))
    )
    fallback_state_seq = 0
    last_submitted_seq = -1
    last_reference: dict | None = None
    last_published_reference = None
    last_published_hands = (None, None)
    last_inference_completed = None
    last_state_meta: dict | None = None
    last_camera_meta: dict | None = None
    initial_root_quat = None
    previous_raw_state = None
    action_queue: deque[np.ndarray] = deque()
    publish_frame = 0
    sonic_pose_window = SonicPoseWindow(size=5, lookahead_frames=args.sonic_lookahead_frames, rate=args.control_rate)
    previous_sonic_heading = None
    episode_id = 0
    hold_announced = False
    streamed_motion_mode_selected = False
    streamed_motion_command_attempts = 0
    streamed_motion_started = False
    policy_action_seen = False
    stop_sent = False
    last_standing_reference: dict | None = None
    state_stream_reported = False
    hand_stream_reported = False
    last_gate_diagnostic_time = 0.0
    last_gate_diagnostic = None
    last_latency_warning = 0.0
    kp_socket = None
    build_command_message = None
    camera = state_subscriber = None

    def send_stop(reason: str) -> None:
        nonlocal stop_sent
        if (
            stop_sent
            or not args.auto_stop_on_timeout
            or kp_socket is None
            or args.no_publish
            or not streamed_motion_started
            or build_command_message is None
        ):
            return
        try:
            if last_standing_reference is not None:
                for _ in range(3):
                    kp_socket.send(build_command_message(start=True, stop=False, planner=False))
                    kp_socket.send(
                        pack_pose_v1(
                            joint_pos_isaaclab=last_standing_reference["joint_pos"],
                            joint_vel_isaaclab=np.zeros((29,), dtype=np.float32),
                            body_quat_wxyz=last_standing_reference["body_quat"],
                            root_position=last_standing_reference["root_position"],
                            frame_index=last_standing_reference["frame_index"],
                            timestamp_monotonic=time.monotonic(),
                            left_hand_joints=np.zeros(7, dtype=np.float32),
                            right_hand_joints=np.zeros(7, dtype=np.float32),
                        )
                    )
                stop_sent = True
                print(f"[deploy] sent SONIC standing hold: {reason}", flush=True)
            else:
                kp_socket.send(build_command_message(start=False, stop=True, planner=False))
                stop_sent = True
                print(f"[deploy] sent SONIC stop: {reason}", flush=True)
        except Exception as exc:
            print(f"[deploy] failed to send SONIC stop ({reason}): {exc}", flush=True)

    def note_hold(reason: str) -> None:
        nonlocal hold_announced
        if hold_announced:
            return
        hold_announced = True
        recorder.emit("hold", reason=reason, queue=len(action_queue))
        print(f"[deploy] holding last safe pose: {reason}", flush=True)

    def clear_hold() -> None:
        nonlocal hold_announced
        hold_announced = False

    reference_preview_queue = deque()

    def invalidate_plan(reason: str, *, force: bool = False) -> None:
        nonlocal last_submitted_seq
        recorder.emit("invalidate", reason=reason, queue=len(action_queue), epoch=tracker.epoch)
        action_queue.clear()
        reference_preview_queue.clear()
        if tracker.invalidate(force=force):
            print(f"[deploy] rebuilding plan: {reason}; epoch={tracker.epoch}", flush=True)
        last_submitted_seq = -1

    def report_inference_gate(reasons: list[str]) -> None:
        nonlocal last_gate_diagnostic_time, last_gate_diagnostic
        if not args.verbose or not reasons:
            return
        now = time.monotonic()
        diagnostic = "; ".join(reasons)
        if (
            now - last_gate_diagnostic_time >= 1.0
        ):
            print(f"[deploy] inference waiting: {diagnostic}", flush=True)
            last_gate_diagnostic = diagnostic
            last_gate_diagnostic_time = now

    def reset_episode(reason: str) -> None:
        nonlocal episode_id, last_submitted_seq, last_inference_completed
        nonlocal last_state_meta, last_camera_meta
        nonlocal previous_raw_state
        episode_id += 1
        # SONIC has not restarted: preserve its heading and integrated command
        # frame. Recovery travels with the next asynchronous infer request.
        invalidate_plan(reason, force=True)
        state_history.clear()
        last_submitted_seq = -1
        last_inference_completed = None
        last_state_meta = None
        last_camera_meta = None
        previous_raw_state = None
        print(f"[deploy] local episode reset ({reason}), episode_id={episode_id}", flush=True)

    if not args.dry_run:
        import zmq

        CameraClient, StateSubscriber, build_command_message = _load_sonic_runtime()
        camera = CameraClient(server_ip=args.camera_host, port=args.camera_port)
        state_subscriber = StateSubscriber(host=args.state_kp_host, port=args.state_kp_port)
        kp_socket = zmq.Context.instance().socket(zmq.PUB)
        kp_socket.bind(f"tcp://{args.action_kp_host}:{args.action_kp_port}")
        time.sleep(0.2)
        if args.no_publish:
            print(
                "[deploy] connected to SONIC action endpoint "
                f"tcp://{args.action_kp_host}:{args.action_kp_port}; "
                "action publishing disabled (--no-publish)",
                flush=True,
            )
        else:
            print(f"[deploy] publishing SONIC pose on tcp://{args.action_kp_host}:{args.action_kp_port}", flush=True)
    else:
        print("[deploy] dry-run enabled; no SONIC sensor or publisher is opened", flush=True)

    print(
        "[deploy] Kimodo bridge ready; verified control rate "
        f"{float(server_fps):g}Hz "
        f"model_hand_control_mode={model_hand_control_mode} "
        f"model_hand_observation_mode={model_hand_observation_mode} "
        f"output_hand_control_mode={hand_control_mode} "
        f"chunk_frames={chunk_frames} request_lead={tracker.threshold}/{tracker.queue_cap} "
        f"history_frames={state_history.maxlen}",
        flush=True,
    )
    recorder = Recorder(args.diagnostics_dir)
    recorder.emit("session", args=vars(args), health=health, session_id=tracker.session_id, sonic_pose_window_frames=5)
    steps = 0
    try:
        while not args.max_steps or steps < args.max_steps:
            tick_start = time.monotonic()
            receive_now = time.monotonic()
            try:
                sensor_missing = False
                if args.dry_run:
                    state_msg, image = _synthetic_state(steps)
                    image_meta = {
                        "sequence": steps,
                        "source_ts": receive_now,
                        "receive_ts": receive_now,
                        "is_new": True,
                    }
                else:
                    try:
                        image_msg = camera.read()
                        state_msg = state_subscriber.get_msg(clear=False)
                    except Exception as exc:
                        sensor_missing = True
                        if last_reference is not None:
                            note_hold(f"sensor read failure: {exc}")
                        image_msg = {
                            "sequence": None,
                            "server_monotonic_timestamp": receive_now,
                            "receive_monotonic_timestamp": receive_now,
                            "is_new": False,
                        }
                        state_msg = {
                            "body_q": (
                                previous_raw_state[0].copy()
                                if previous_raw_state is not None
                                else np.zeros(29, dtype=np.float32)
                            ),
                            "body_dq": np.zeros(29, dtype=np.float32),
                            "base_quat": (
                                initial_root_quat.copy()
                                if initial_root_quat is not None
                                else np.array([1, 0, 0, 0], dtype=np.float32)
                            ),
                            "state_monotonic_timestamp": receive_now,
                            "index": fallback_state_seq,
                        }
                    image = None if image_msg is None else image_msg.get("images", {}).get("ego_view")
                    sensor_missing = sensor_missing or state_msg is None or image is None
                    if sensor_missing:
                        if last_reference is not None:
                            note_hold("camera/state stream unavailable")
                        image_msg = {
                            "sequence": None,
                            "server_monotonic_timestamp": receive_now,
                            "receive_monotonic_timestamp": receive_now,
                            "is_new": False,
                        }
                        state_msg = {
                            "body_q": (
                                previous_raw_state[0].copy()
                                if previous_raw_state is not None
                                else np.zeros(29, dtype=np.float32)
                            ),
                            "body_dq": np.zeros(29, dtype=np.float32),
                            "base_quat": (
                                initial_root_quat.copy()
                                if initial_root_quat is not None
                                else np.array([1, 0, 0, 0], dtype=np.float32)
                            ),
                            "state_monotonic_timestamp": receive_now,
                            "index": fallback_state_seq,
                        }
                    image_meta = _camera_metadata(image_msg, receive_now)
                    if not sensor_missing:
                        image = np.asarray(image)
                        if image.ndim == 4:
                            image = image[-1]
                        if image.ndim != 3 or image.shape[-1] not in (3, 4):
                            print(f"[deploy] rejecting camera frame with shape {image.shape}", flush=True)
                            time.sleep(period)
                            continue
                        if image.shape[-1] == 4:
                            image = image[..., :3]
                        if image.dtype != np.uint8:
                            image = np.clip(image, 0, 255).astype(np.uint8)

                if not sensor_missing and last_camera_meta is not None:
                    camera_rollback = (
                        image_meta["source_ts"] < last_camera_meta["source_ts"]
                        or (
                            image_meta["sequence"] is not None
                            and last_camera_meta["sequence"] is not None
                            and image_meta["sequence"] < last_camera_meta["sequence"]
                        )
                    )
                    if camera_rollback:
                        reset_episode("camera sequence/timestamp rollback")
                if not sensor_missing and (
                    last_camera_meta is None
                    or image_meta["source_ts"] > last_camera_meta["source_ts"]
                    or image_meta["sequence"] != last_camera_meta["sequence"]
                ):
                    last_camera_meta = image_meta

                state_meta = _state_metadata(state_msg, receive_now, fallback_state_seq)
                fallback_state_seq = state_meta["sequence"] + 1
                if not sensor_missing and last_state_meta is not None:
                    if (
                        state_meta["sequence"] < last_state_meta["sequence"]
                        or state_meta["source_ts"] < last_state_meta["source_ts"]
                    ):
                        reset_episode("state sequence/timestamp rollback")
                    elif state_meta["source_ts"] - last_state_meta["source_ts"] > args.max_state_gap_s:
                        reset_episode("state source gap")

                measured_hand_closure = None
                hand_state_valid = not measured_hand_required
                if not sensor_missing and measured_hand_required:
                    measured_hand_closure, hand_error = _read_measured_hand_state(state_msg)
                    if hand_error is None:
                        hand_state_valid = True
                        if args.verbose and not hand_stream_reported:
                            print(
                                "[deploy] measured hand state received: "
                                f"closure={measured_hand_closure.tolist()}",
                                flush=True,
                            )
                            hand_stream_reported = True
                    else:
                        hand_state_valid = False
                        invalidate_plan("invalid measured hand state")
                        note_hold(f"invalid measured hand state: {hand_error}")

                if sensor_missing:
                    invalidate_plan("camera/state stream unavailable")
                    state = np.zeros(64, dtype=np.float32)
                    q = np.zeros(29, dtype=np.float32)
                    dq = np.zeros(29, dtype=np.float32)
                    root_quat = (
                        initial_root_quat.copy()
                        if initial_root_quat is not None
                        else np.array([1, 0, 0, 0], dtype=np.float32)
                    )
                    state_ts = receive_now
                    state_valid = False
                else:
                    try:
                        state, q, dq, root_quat, state_ts, initial_root_quat = state_msg_to_kimodo_state(
                            state_msg,
                            initial_root_quat_wxyz=initial_root_quat,
                            previous_state=previous_raw_state,
                        )
                    except BridgeSafetyError as exc:
                        note_hold(f"invalid robot state: {exc}")
                        invalidate_plan("invalid robot state")
                        state = root_quat = None
                        state_ts = None
                        state_valid = False
                    else:
                        state_valid = True
                        if args.verbose and not state_stream_reported:
                            print(
                                "[deploy] g1_debug state received: "
                                f"seq={state_meta['sequence']} "
                                f"keys={','.join(sorted(state_msg))}",
                                flush=True,
                            )
                            state_stream_reported = True
                        previous_raw_state = (
                            np.asarray(state_msg["body_q"], dtype=np.float32).reshape(29),
                            state_ts,
                        )
                        is_new_state = (
                            last_state_meta is None
                            or state_meta["sequence"] > last_state_meta["sequence"]
                            or (
                                not state_meta["has_source_seq"]
                                and state_meta["source_ts"] > last_state_meta["source_ts"]
                            )
                        )
                        if is_new_state and (
                            hand_state_valid or not measured_hand_required
                        ):
                            state_history.append(
                                (
                                    state_meta["sequence"],
                                    state.copy(),
                                    None
                                    if measured_hand_closure is None
                                    else measured_hand_closure.copy(),
                                    state_ts,
                                )
                            )
                            last_state_meta = state_meta

                standing_root_quat = (
                    root_quat
                    if state_valid
                    else (
                        initial_root_quat
                        if initial_root_quat is not None
                        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    )
                )
                standing_joint_pos = (
                    ELBOWS_110_STANDING_JOINT_POS_ISAACLAB
                    if args.standby_elbows_110
                    else ELBOWS_90_STANDING_JOINT_POS_ISAACLAB
                    if args.standby_elbows_90
                    else DEFAULT_STANDING_JOINT_POS_ISAACLAB
                )
                standing_reference = _build_standing_reference(
                    standing_root_quat,
                    publish_frame,
                    standing_joint_pos,
                )
                last_standing_reference = standing_reference

                preparing = startup is not None and not startup.ready
                if preparing:
                    relative_quat = rot6d_row_to_quat_wxyz(state[:6]) if state_valid else np.array([1,0,0,0])
                    startup.step(time.monotonic(), q, dq, relative_quat,
                                 valid=state_valid and hand_state_valid)
                    if startup.reset_history:
                        state_history.clear()
                        recorder.emit("initial_pose_history_reset")
                    if startup.phase != startup_last_phase:
                        blocked = {key: {"value": startup.errors[key], "limit": startup.config[setting]}
                                   for key, setting in [("arm_rms", "arm_rms_tolerance_rad"),
                                                        ("arm_max", "arm_max_tolerance_rad"),
                                                        ("body_rms", "body_rms_tolerance_rad"),
                                                        ("velocity_rms", "velocity_rms_tolerance_rad_s"),
                                                        ("orientation", "orientation_tolerance_rad")]
                                   if key in startup.errors and startup.errors[key] > startup.config[setting]}
                        recorder.emit("initial_pose_phase", phase=startup.phase, errors=startup.errors, blocked=blocked)
                        if blocked:
                            print(f"[PREPARE] blocked: {blocked}", flush=True)
                        print(f"[PREPARE] {startup.phase}; source={startup.config['source_episode']} frame=0", flush=True)
                        startup_last_phase = startup.phase
                    if startup.phase in ("settling", "history") and time.monotonic()-startup_last_report > 2:
                        print(f"[PREPARE] {startup.phase}: measured errors {startup.errors}; model inference remains gated", flush=True)
                        startup_last_report = time.monotonic()
                    if startup.ready:
                        runtime.previous_joint_q = startup.reference["joint_pos"].copy()
                        runtime.previous_reference_quat = startup.reference["body_quat"].copy()
                        runtime.previous_reference_z = float(startup.reference["root_position"][2])
                        runtime.reference_root_position = startup.reference["root_position"].copy()
                        print("[PREPARE] Stable measured initial pose and fresh history ready; enabling model inference", flush=True)
                    preparing = not startup.ready

                if not standby_mode and not preparing:
                    while True:
                        try:
                            result = worker.results.get_nowait()
                        except queue.Empty:
                            break
                        if result.get("episode_id", episode_id) != episode_id:
                            print("[deploy] discarded result from an old episode", flush=True)
                            continue
                        if not state_valid or not hand_state_valid:
                            invalidate_plan("result arrived without valid measured state")
                            continue
                        accepted, details = _admit_result(
                            result, tracker, runtime, action_queue,
                            now=time.monotonic(), latest_state_ts=state_ts,
                            measured_joints=q, root_quat=root_quat,
                            max_state_lag_s=args.max_result_state_lag_s,
                            max_result_age_s=args.max_inference_age_s,
                            max_start_age_s=args.max_action_start_age_s,
                        )
                        recorder.emit("result", timing=result.get("timing"), accepted=accepted, details=details, elapsed_s=result.get("inference_elapsed_s"), deployment=result.get("deployment"), action=result.get("action"), error=result.get("error"))
                        if accepted:
                            if args.sonic_lookahead_frames:
                                reference_preview_queue = _preview_reference_queue(runtime, action_queue, q, root_quat)
                            last_inference_completed = time.monotonic()
                            if args.verbose:
                                print(
                                    "[deploy] Kimodo chunk accepted: "
                                    f"plan={result['deployment']['request_id']} "
                                    f"queue={details['queued_frames']}+{chunk_frames} "
                                    f"consumed={details['consumed_frames']} "
                                    f"age={details['age_s']:.3f}s "
                                    f"start_age={details['first_execution_age_s']:.3f}s "
                                    f"boundary_step={details['boundary_joint_step_rad']:.4f}rad "
                                    f"lead={tracker.threshold}",
                                    flush=True,
                                )
                        else:
                            print(f"[deploy] discarded Kimodo result: {details}", flush=True)
                            last_submitted_seq = -1
                        if tracker.lead_saturated and time.monotonic() - last_latency_warning > 2.0:
                            last_latency_warning = time.monotonic()
                            print(
                                "[deploy] inference lead saturated: "
                                f"recent_latency={max(tracker.latencies):.3f}s "
                                f"lead_budget={tracker.queue_cap / args.control_rate:.3f}s "
                                f"chunk_duration={chunk_frames / args.control_rate:.3f}s",
                                flush=True,
                            )

                if startup is not None and (preparing or not policy_action_seen):
                    if startup.reference is not None:
                        last_reference = startup.reference
                elif (
                    state_valid
                    and standing_reference is not None
                    and (standby_mode or (args.auto_start and not policy_action_seen))
                ):
                    last_reference = standing_reference
                    if standby_mode:
                        clear_hold()

                reference_updated = False
                if (
                    not standby_mode
                    and not preparing
                    and state_valid
                    and hand_state_valid
                    and action_queue
                ):
                    try:
                        # The first policy target must be continuous with the
                        # measured canonical joint state.  Otherwise a model
                        # action can be accepted before the bridge has any
                        # physical reference for its per-joint jump check.
                        runtime_snapshot = runtime.snapshot()
                        runtime.prime_joints(q)
                        last_reference = runtime.decode(action_queue.popleft(), current_root_quat_wxyz=root_quat)
                        if reference_preview_queue:
                            reference_preview_queue.popleft()
                        reference_updated = True
                        policy_action_seen = True
                    except BridgeSafetyError as exc:
                        runtime.restore(runtime_snapshot)
                        print(f"[deploy] rejecting Kimodo action and holding last safe reference: {exc}", flush=True)
                        invalidate_plan("action safety rejection")
                        note_hold(f"invalid Kimodo action: {exc}")
                if (not standby_mode and not preparing and not reference_updated and not action_queue
                        and tracker.parent_request_id is not None):
                    # The in-flight child starts at the exhausted queue endpoint.
                    # Keep its identity while holding that endpoint. Admission still
                    # enforces age/state-lag/joint checks; do not discard a valid
                    # late child solely because its parent finished executing.
                    if tracker.exhausted_queue_needs_replan():
                        invalidate_plan("action queue ran out without pending result")
                    note_hold("Kimodo action queue ran out; waiting for pending result")
                if last_reference is not None:
                    try:
                        hand_action = last_reference.get("hand_closure")
                        if hand_action is None:
                            hand_action = last_reference["hand_binary"]
                        left_hand, right_hand = _hand_targets(
                            hand_action,
                            hand_control_mode=hand_control_mode,
                        )
                    except Exception as exc:
                        note_hold(f"hand target failure: {exc}")
                        left_hand = right_hand = None
                    inference_stale = False if preparing else _inference_is_stale(
                        bool(action_queue) or reference_updated,
                        last_inference_completed,
                        time.monotonic(),
                        runtime.limits.max_reference_age_s,
                    )
                    if inference_stale:
                        note_hold("Kimodo inference timeout")
                    if standing_reference is not None and standby_mode:
                        last_reference = standing_reference
                        left_hand = right_hand = np.zeros(7, dtype=np.float32)
                    publish_reference = dict(last_reference)
                    if (not reference_updated or inference_stale) and not preparing:
                        publish_reference["joint_vel"] = np.zeros((29,), dtype=np.float32)
                    packet = sonic_pose_window.append(
                        future_references=(reference_preview_queue if state_valid and hand_state_valid and not preparing and not standby_mode else ()),
                        joint_pos_isaaclab=publish_reference["joint_pos"],
                        joint_vel_isaaclab=publish_reference["joint_vel"],
                        body_quat_wxyz=publish_reference["body_quat"],
                        root_position=publish_reference["root_position"],
                        frame_index=publish_frame,
                        timestamp_monotonic=time.monotonic(),
                        left_hand_joints=left_hand,
                        right_hand_joints=right_hand,
                    )
                    publish_frame += 1
                else:
                    packet = None
                if (
                    kp_socket is not None
                    and not args.no_publish
                    and packet is not None
                ):
                    should_start = _streamed_motion_should_start(
                        state_valid=state_valid,
                        standby=standby_mode or preparing,
                        auto_start=args.auto_start,
                        policy_action_seen=policy_action_seen,
                    )
                    send_command = streamed_motion_command_attempts < 20
                    if args.auto_start and policy_action_seen and not streamed_motion_started:
                        send_command = True
                    if send_command:
                        kp_socket.send(
                            build_command_message(start=should_start, stop=False, planner=False)
                        )
                        streamed_motion_command_attempts += 1
                    if not streamed_motion_mode_selected:
                        streamed_motion_mode_selected = True
                        if should_start:
                            streamed_motion_started = True
                            if standby_mode:
                                print("[deploy] sent standing-hold pose after first valid pose", flush=True)
                            elif preparing:
                                print("[PREPARE] started SONIC for data-sourced initial-pose transition", flush=True)
                            elif args.auto_start:
                                print("[deploy] sent streamed-motion start after first valid Kimodo action", flush=True)
                        elif args.verbose:
                            print(
                                "[deploy] selected streamed-motion mode; control loop remains stopped",
                                flush=True,
                            )
                    elif should_start and not streamed_motion_started:
                        streamed_motion_started = True
                        print("[deploy] sent streamed-motion start after first valid Kimodo action", flush=True)
                    kp_socket.send(packet)
                    last_published_reference = copy.deepcopy(publish_reference)
                    last_published_hands = (None if left_hand is None else left_hand.copy(), None if right_hand is None else right_hand.copy())
                    if (
                        reference_updated
                        and not inference_stale
                        and left_hand is not None
                        and right_hand is not None
                    ):
                        clear_hold()

                newest_seq = state_history[-1][0] if state_history else -1
                newest_state_ts = state_history[-1][3] if state_history else state_ts
                pending = list(state_history)
                has_new_state = newest_seq > last_submitted_seq
                camera_age, sensor_skew = _camera_age_and_sensor_skew(image_meta, state_meta, time.monotonic())
                camera_usable = bool(image_meta["is_new"]) and camera_age <= args.max_camera_stale_s
                if sensor_skew * 1000.0 > args.max_sensor_skew_ms:
                    camera_usable = False
                    if args.verbose:
                        print(f"[deploy] withholding inference for sensor skew {sensor_skew*1000:.1f}ms", flush=True)
                inference_submitted = False
                if (
                    state_valid
                    and hand_state_valid
                    and not standby_mode
                    and not preparing
                    and not worker.busy.is_set()
                    and tracker.pending is None
                    and len(action_queue) <= tracker.threshold
                    and pending
                    and has_new_state
                    and camera_usable
                ):
                    segment = np.stack([item[1] for item in pending], axis=0).astype(np.float32)
                    hand_segment = None
                    if measured_hand_required:
                        if any(item[2] is None for item in pending):
                            note_hold("measured hand history is incomplete")
                            hand_state_valid = False
                        else:
                            hand_segment = np.stack(
                                [item[2] for item in pending], axis=0
                            ).astype(np.float32)
                    boundary, _ = _preview_actions(runtime, action_queue, [], q, root_quat)
                    if boundary is None:
                        if last_reference is not None:
                            boundary = reference_to_boundary_action(last_reference)
                        else:
                            boundary = np.zeros(40, dtype=np.float32)
                            boundary[2] = DEFAULT_STANDING_ROOT_POSITION[2]
                            boundary[3:9] = state[:6]
                            boundary[9:38] = q
                            if measured_hand_closure is not None:
                                boundary[38:40] = measured_hand_closure
                    if hand_control_mode == "binary":
                        # The output override thresholds closure before publishing.
                        # Report that actual command, while observations stay measured.
                        boundary[38:40] = (boundary[38:40] >= 0.5).astype(np.float32)
                    deployment = tracker.begin(boundary)
                    request = {
                        "image": image.copy(),
                        "state": segment[-1].copy(),
                        "history": segment,
                        "hand_closure_history": hand_segment,
                        "task": args.task,
                        "start_seq": pending[0][0],
                        "end_seq": newest_seq,
                        "end_state_ts": newest_state_ts,
                        "episode_id": episode_id,
                        "submitted_monotonic": time.monotonic(),
                        "deployment": deployment,
                        "state_sequences": [int(item[0]) for item in pending],
                        "state_timestamps": [float(item[3]) for item in pending],
                        "queue_at_submission": len(action_queue),
                    }
                    if worker.submit(request):
                        recorder.image(f"request-{deployment['request_id']:06d}.jpg", image)
                        recorder.emit("submit", deployment=deployment, state_seq=newest_seq, state_timestamp=newest_state_ts, queue=len(action_queue), history_frames=len(segment), camera_age_s=camera_age, sensor_skew_s=sensor_skew)
                        last_submitted_seq = newest_seq
                        inference_submitted = True
                        if args.verbose:
                            print(
                                "[deploy] Kimodo inference submitted: "
                                f"plan={deployment['request_id']} epoch={deployment['epoch']} "
                                f"parent={deployment['parent_request_id']} "
                                f"state_seq={newest_seq} history={segment.shape[0]} "
                                f"queue={len(action_queue)} lead={tracker.threshold}",
                                flush=True,
                            )
                    else:
                        tracker.invalidate()

                if not inference_submitted:
                    gate_reasons = []
                    if not state_valid:
                        gate_reasons.append("no valid g1_debug state")
                    if measured_hand_required and not hand_state_valid:
                        gate_reasons.append("measured hand state invalid/missing")
                    if preparing:
                        gate_reasons.append(f"initial pose {startup.phase}")
                    if standby_mode:
                        gate_reasons.append(f"{standby_mode_name} active")
                    if worker.busy.is_set():
                        gate_reasons.append("Kimodo inference busy")
                    if len(action_queue) > tracker.threshold:
                        gate_reasons.append(
                            f"action queue={len(action_queue)}>{tracker.threshold} (adaptive lead)"
                        )
                    if not pending or not has_new_state:
                        gate_reasons.append("no new state frames")
                    if not camera_usable:
                        if not image_meta["is_new"]:
                            gate_reasons.append("waiting for a new camera frame")
                        elif camera_age > args.max_camera_stale_s:
                            gate_reasons.append(f"camera stale={camera_age:.3f}s")
                        if sensor_skew * 1000.0 > args.max_sensor_skew_ms:
                            gate_reasons.append(
                                f"sensor skew={sensor_skew * 1000.0:.1f}ms"
                            )
                    report_inference_gate(gate_reasons or ["inference submit race/full"])

                sonic_heading = state_msg.get("init_base_quat") if state_valid else None
                if sonic_heading is not None:
                    sonic_heading = np.asarray(sonic_heading, dtype=np.float64)
                    if previous_sonic_heading is not None and not np.allclose(sonic_heading, previous_sonic_heading, atol=1e-7, rtol=0):
                        recorder.emit("sonic_heading_reset", step=steps, previous=previous_sonic_heading, current=sonic_heading, policy_action_seen=policy_action_seen, streaming_started=streamed_motion_started)
                        if streamed_motion_started:
                            print("[deploy] SONIC heading baseline changed; inspect sonic_heading_reset in diagnostics", flush=True)
                    previous_sonic_heading = sonic_heading.copy()
                if steps % 5 == 0:
                    recorder.emit("control", sonic_lookahead_frames=args.sonic_lookahead_frames, committed_preview_frames=min(len(reference_preview_queue),args.sonic_lookahead_frames), step=steps, queue=len(action_queue), state_valid=state_valid, hand_valid=hand_state_valid, state_timestamp=state_ts, measured_q=q, measured_dq=dq, measured_root_quat=root_quat, initial_root_quat=initial_root_quat, measured_hand=measured_hand_closure, reference=last_reference, reference_updated=reference_updated, publish_enabled=not args.no_publish and not args.dry_run, loop_elapsed_s=time.monotonic()-tick_start, sonic_initial_root_quat=state_msg.get("init_base_quat"), sonic_root_target=state_msg.get("base_quat_target"), sonic_delta_heading=state_msg.get("delta_heading"), sonic_body_q_target=state_msg.get("body_q_target"), startup_phase=None if startup is None else startup.phase)
                steps += 1
                elapsed = time.monotonic() - tick_start
                if elapsed < period:
                    time.sleep(period - elapsed)
            except Exception as exc:
                invalidate_plan("unexpected bridge error")
                note_hold(f"unexpected bridge error: {exc}")
                time.sleep(period)
                continue
    except KeyboardInterrupt:
        if (args.hold_on_interrupt and kp_socket is not None and not args.no_publish
                and last_published_reference is not None and streamed_motion_started):
            # Latch out all inference. Do not join a worker before starting hold.
            worker.stop_event.set()
            stop_sent = True  # finally must never send a default pose/stop after hold.
            publish_frame = max(publish_frame, int(sonic_pose_window.frames[-1]["frame_index"][0])+1)
            publish_frame = hold_last_reference(last_published_reference, last_published_hands,
                                               window=sonic_pose_window, next_frame=publish_frame,
                                               send=kp_socket.send, recorder=recorder, rate=args.control_rate)
        else:
            send_stop("Ctrl-C")
        print("[deploy] task ended", flush=True)
    finally:
        send_stop("bridge shutdown")
        worker.close()
        recorder.emit("shutdown", steps=steps)
        recorder.close()
        if kp_socket is not None:
            kp_socket.close(0)
        if state_subscriber is not None:
            state_subscriber.close()


if __name__ == "__main__":
    main()
