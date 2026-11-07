"""Wire validation and client-side scheduling for sequential Kimodo plans."""

from __future__ import annotations

import math
import uuid
from collections import deque

import numpy as np


PROTOCOL_VERSION = 1
_IDENTITY_FIELDS = (
    "version", "session_id", "epoch", "request_id", "parent_request_id",
)


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_deployment(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("deployment must be an object")
    if type(value.get("version")) is not int or value["version"] != PROTOCOL_VERSION:
        raise ValueError(f"deployment.version must be {PROTOCOL_VERSION}")
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
        raise ValueError("deployment.session_id must be a non-empty string of at most 128 characters")
    epoch = _nonnegative_int(value.get("epoch"), "deployment.epoch")
    request_id = _nonnegative_int(value.get("request_id"), "deployment.request_id")
    parent = value.get("parent_request_id")
    if parent is not None:
        parent = _nonnegative_int(parent, "deployment.parent_request_id")
        if parent >= request_id:
            raise ValueError("deployment.parent_request_id must precede request_id")
    result = dict(version=PROTOCOL_VERSION, session_id=session_id, epoch=epoch,
                  request_id=request_id, parent_request_id=parent)
    boundary = value.get("boundary_action")
    if boundary is not None:
        boundary = np.asarray(boundary, dtype=np.float32)
        if boundary.shape != (40,) or not np.isfinite(boundary).all():
            raise ValueError("deployment.boundary_action must be finite shape [40]")
        if np.any(boundary[:2] != 0):
            raise ValueError("boundary_action XY deltas must be zero")
        columns = boundary[3:9].reshape(3, 2)
        if np.linalg.norm(np.cross(columns[:, 0], columns[:, 1])) < 1e-6:
            raise ValueError("boundary_action root rotation is degenerate")
        if np.any((boundary[38:40] < 0) | (boundary[38:40] > 1)):
            raise ValueError("boundary_action hands must be within [0,1]")
        result["boundary_action"] = boundary.tolist()
    return result


def deployment_identity(value: dict) -> dict:
    validated = validate_deployment(value)
    return {key: validated[key] for key in _IDENTITY_FIELDS}


def validate_observation_window(observation: dict, max_frames: int):
    states = np.asarray(observation.get("state_history"), dtype=np.float32)
    if (states.ndim != 2 or states.shape[1:] != (64,)
            or not 1 <= len(states) <= max_frames or not np.isfinite(states).all()):
        raise ValueError(f"state_history must be finite [T,64], 1 <= T <= {max_frames}")
    sequence_values = observation.get("state_sequences")
    if not isinstance(sequence_values, list) or len(sequence_values) != len(states):
        raise ValueError("state_sequences must match state_history length")
    for value in sequence_values:
        _nonnegative_int(value, "state_sequences entry")
        if value > np.iinfo(np.int64).max:
            raise ValueError("state sequence exceeds int64")
    sequences = np.asarray(sequence_values, dtype=np.int64)
    timestamps = np.asarray(observation.get("state_timestamps"), dtype=np.float64)
    if (timestamps.shape != (len(states),) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or np.any(sequences[1:] <= sequences[:-1])):
        raise ValueError("state sequences and timestamps must be finite and strictly increasing")
    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (64,) or not np.array_equal(state, states[-1]):
        raise ValueError("observation.state must equal the newest state_history row")
    hand = observation.get("hand_closure_history")
    if hand is not None:
        hand = np.asarray(hand, dtype=np.float32)
        if (hand.shape != (len(states), 2) or not np.isfinite(hand).all()
                or np.any((hand < 0) | (hand > 1))):
            raise ValueError("hand_closure_history must be aligned finite [T,2] within [0,1]")
    return states, sequences, timestamps, hand


class BridgePlanTracker:
    """A single speculative request and the last plan admitted to the FIFO."""

    def __init__(self, *, control_fps: float, chunk_frames: int, queue_cap: int,
                 initial_latency_s: float = 0.25, margin_s: float = 0.04,
                 adaptive: bool = True, session_id: str | None = None):
        if not math.isfinite(control_fps) or control_fps <= 0:
            raise ValueError("control_fps must be finite and positive")
        if chunk_frames < 1 or queue_cap < 0:
            raise ValueError("chunk_frames must be positive and queue_cap non-negative")
        if any(not math.isfinite(v) or v < 0 for v in (initial_latency_s, margin_s)):
            raise ValueError("inference latency estimate and margin must be finite and non-negative")
        self.control_fps = control_fps
        self.chunk_frames = chunk_frames
        self.queue_cap = min(queue_cap, chunk_frames - 1)
        self.margin_s = margin_s
        self.adaptive = adaptive
        self.latencies = deque([initial_latency_s], maxlen=6)
        self.session_id = session_id or uuid.uuid4().hex
        self.epoch = 0
        self.next_request_id = 0
        self.parent_request_id = None
        self.pending = None

    @property
    def threshold(self) -> int:
        if not self.adaptive:
            return self.queue_cap
        required = math.ceil((max(self.latencies) + self.margin_s) * self.control_fps - 1e-9)
        return min(self.queue_cap, required)

    @property
    def lead_saturated(self) -> bool:
        return (max(self.latencies) + self.margin_s) * self.control_fps > self.queue_cap

    def observe_latency(self, elapsed_s: float) -> None:
        if math.isfinite(elapsed_s) and elapsed_s >= 0:
            self.latencies.append(elapsed_s)

    def begin(self, boundary_action=None) -> dict:
        if self.pending is not None:
            raise RuntimeError("a Kimodo request is already pending")
        value = dict(version=PROTOCOL_VERSION, session_id=self.session_id,
                     epoch=self.epoch, request_id=self.next_request_id,
                     parent_request_id=self.parent_request_id)
        if boundary_action is not None:
            value["boundary_action"] = np.asarray(boundary_action).tolist()
        value = validate_deployment(value)
        self.next_request_id += 1
        self.pending = deployment_identity(value)
        return value

    def matches(self, value) -> bool:
        try:
            return self.pending is not None and deployment_identity(value) == self.pending
        except (ValueError, TypeError, OverflowError):
            return False

    def accept(self, value) -> None:
        if not self.matches(value):
            raise ValueError("result does not belong to the current pending plan")
        self.parent_request_id = self.pending["request_id"]
        self.pending = None

    def exhausted_queue_needs_replan(self) -> bool:
        """An in-flight child remains eligible at the held parent endpoint."""
        return self.pending is None and self.parent_request_id is not None

    def invalidate(self, *, force: bool = False) -> bool:
        # Empty held loops must not invalidate a newly issued recovery repeatedly.
        if not force and self.pending is None and self.parent_request_id is None:
            return False
        self.epoch += 1
        self.parent_request_id = None
        self.pending = None
        return True
