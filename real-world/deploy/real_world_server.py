#!/usr/bin/env python3
"""Kimodo inference HTTP server for the real-robot SONIC bridge.

This is the deployment entry point.  It owns its own HTTP handler, command
line interface, RealWorld text-cache loading, and checkpoint runtime.  The
model runtime is loaded lazily so the protocol can be tested without a
GPU/checkpoint.

The bridge sends one RGB camera frame and a 64-D state history at every
replanning request.  Continuous-hand checkpoints additionally receive a
same-length ``[T, 2]`` measured hand-closure history.  A successful request
returns a finite ``[N, 40]`` Kimodo action chunk.  The SONIC bridge, rather
than this server, performs the last safety checks and translates that semantic
chunk into Protocol v1.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import math
import os
import sys
import threading
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("KIMODO_PROJECT_ROOT", THIS_DIR.parents[1])).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from image_wire import decode_image_bytes  # noqa: E402
from hand_stability import stabilize_binary_hand  # noqa: E402

from kimodo_rtc import (  # noqa: E402 - direct-script and importlib entry points
    PROTOCOL_VERSION,
    deployment_identity,
    validate_deployment,
    validate_observation_window,
)


DEFAULT_TEXT_EMBEDDING_CACHE = Path(os.environ.get("KIMODO_TEXT_CACHE", PROJECT_ROOT / "data/cache/RealWorld"))
G1_XML_PATH = PROJECT_ROOT / "skeleton/assets/g1skel34/xml/g1.xml"
MAX_REQUEST_BYTES = 128 * 1024 * 1024
REAL_WORLD_TEXT_CACHE_SOURCE = "RealWorld"
_SEED_MASK = (1 << 63) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


def _load_torch():
    if os.environ.get("KIMODO_DETERMINISTIC_EVAL", "0").strip() == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    return torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _derive_inference_seed(episode_seed: int, inference_index: int) -> int:
    value = (
        (int(episode_seed) & 0xFFFFFFFFFFFFFFFF)
        + _SPLITMIX_INCREMENT * (int(inference_index) + 1)
    ) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & _SEED_MASK


def _configure_deterministic_torch() -> None:
    torch = _load_torch()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _dtype_from_name(name: str):
    torch = _load_torch()
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _validate_torch_device(torch, device_name: str):
    device = torch.device(device_name)
    if device.type == "cuda":
        if getattr(torch.version, "cuda", None) is None:
            raise RuntimeError(
                f"--device {device_name} requires CUDA-enabled PyTorch, but "
                f"the active torch build is {torch.__version__} with CUDA=None. "
                "Install a CUDA PyTorch build in the kimodo conda environment, "
                "or use --device cpu only for a non-realtime smoke test."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_name} was requested, but torch.cuda.is_available() "
                "is false. Check that the NVIDIA driver is running and /dev/nvidia* "
                "devices are visible in this shell."
            )
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {device_name} requests CUDA device {device_index}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device


def _resolve_execution_frames(requested_frames: int, prediction_frames: int) -> int:
    requested_frames = int(requested_frames)
    prediction_frames = int(prediction_frames)
    if prediction_frames <= 0:
        raise ValueError(f"prediction_frames must be positive, got {prediction_frames}")
    if requested_frames < 0:
        raise ValueError(f"execution_frames must be non-negative, got {requested_frames}")
    if requested_frames == 0:
        return prediction_frames
    if requested_frames > prediction_frames:
        raise ValueError(
            f"execution_frames ({requested_frames}) cannot exceed the checkpoint "
            f"action_chunk ({prediction_frames})"
        )
    return requested_frames


def _resolve_rtc_parameters(
    enabled: bool,
    *,
    prediction_frames: int,
    execution_frames: int,
    overlap_frames: int,
    frozen_frames: int,
    ramp_power: float,
) -> tuple[bool, int, int, float]:
    prediction_frames = int(prediction_frames)
    execution_frames = int(execution_frames)
    overlap_frames = int(overlap_frames)
    frozen_frames = int(frozen_frames)
    ramp_power = float(ramp_power)
    if overlap_frames < 0:
        raise ValueError(f"rtc_overlap_frames must be non-negative, got {overlap_frames}")
    if frozen_frames < 0:
        raise ValueError(f"rtc_frozen_frames must be non-negative, got {frozen_frames}")
    if frozen_frames > overlap_frames:
        raise ValueError(
            "rtc_frozen_frames cannot exceed rtc_overlap_frames, got "
            f"{frozen_frames} and {overlap_frames}"
        )
    if not np.isfinite(ramp_power) or ramp_power <= 0:
        raise ValueError(f"rtc_ramp_power must be finite and positive, got {ramp_power}")
    available_tail = max(0, prediction_frames - execution_frames)
    effective_overlap = min(overlap_frames, available_tail)
    effective_frozen = min(frozen_frames, effective_overlap)
    active = bool(enabled) and effective_overlap > 0
    return active, effective_overlap, effective_frozen, ramp_power


def _select_execution_prefix(
    future_features,
    local_rot_mats,
    root_positions,
    execution_frames: int,
):
    lengths = {
        int(future_features.shape[0]),
        int(local_rot_mats.shape[0]),
        int(root_positions.shape[0]),
    }
    if len(lengths) != 1:
        raise ValueError(f"Prediction outputs have inconsistent frame counts: {sorted(lengths)}")
    available_frames = lengths.pop()
    if execution_frames > available_frames:
        raise ValueError(
            f"execution_frames ({execution_frames}) exceeds prediction output ({available_frames})"
        )
    return (
        future_features[:execution_frames],
        local_rot_mats[:execution_frames],
        root_positions[:execution_frames],
    )


def _select_hand_execution_prefix(hand_binary, execution_frames: int):
    if hand_binary.ndim != 2 or hand_binary.shape[-1] != 2:
        raise ValueError(
            f"Expected predicted hand shape [T, 2], got {tuple(hand_binary.shape)}"
        )
    if execution_frames > hand_binary.shape[0]:
        raise ValueError(
            f"execution_frames ({execution_frames}) exceeds hand prediction "
            f"({hand_binary.shape[0]})"
        )
    return hand_binary[:execution_frames]


def _resolve_hand_control_mode(value: Any) -> str:
    mode = str(value if value is not None else "binary").strip().lower()
    if mode not in {"binary", "continuous"}:
        raise ValueError(
            "Checkpoint model.hand_control_mode must be 'binary' or 'continuous', "
            f"got {mode!r}"
        )
    return mode


def _resolve_hand_observation_mode(
    value: Any,
    hand_control_mode: str,
) -> str:
    mode = hand_control_mode if value is None else str(value).strip().lower()
    if mode not in {"binary", "continuous"}:
        raise ValueError(
            "Checkpoint model.hand_observation_mode must be 'binary' or "
            f"'continuous', got {mode!r}"
        )
    return mode


def _prediction_hand_tensor(value: Any, name: str):
    torch = _load_torch()
    hand = torch.as_tensor(value).detach().float().cpu()
    if hand.ndim == 3 and hand.shape[0] == 1:
        hand = hand[0]
    if hand.ndim != 2 or hand.shape[-1] != 2:
        raise ValueError(f"Expected {name} shape [T, 2], got {tuple(hand.shape)}")
    if not torch.isfinite(hand).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return hand


def _validated_hand_closure_history(value: Any, *, name: str = "hand_closure_history"):
    hand = np.asarray(value, dtype=np.float32)
    if hand.ndim == 1:
        hand = hand.reshape(1, -1)
    if hand.ndim != 2 or hand.shape[1:] != (2,) or hand.shape[0] == 0:
        raise ValueError(f"observation.{name} must have shape [T, 2]")
    if not np.isfinite(hand).all():
        raise ValueError(f"observation.{name} contains NaN or Inf")
    if ((hand < -1e-6) | (hand > 1.0 + 1e-6)).any():
        raise ValueError(f"observation.{name} must be within [0, 1]")
    return np.clip(hand, 0.0, 1.0).astype(np.float32, copy=False)


def _align_hand_history(hand_history, target_length: int):
    torch = _load_torch()
    target_length = int(target_length)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if hand_history.ndim != 3 or hand_history.shape[0] != 1 or hand_history.shape[2] != 2:
        raise ValueError(
            f"Expected hand_history shape [1, T, 2], got {tuple(hand_history.shape)}"
        )
    if hand_history.shape[1] >= target_length:
        return hand_history[:, -target_length:] if target_length else hand_history[:, :0]
    padding = torch.zeros(
        1,
        target_length - hand_history.shape[1],
        2,
        device=hand_history.device,
        dtype=hand_history.dtype,
    )
    return torch.cat((padding, hand_history), dim=1)


def _kimodo_features_from_motion(local_rot_mats, root_positions, representation):
    torch = _load_torch()
    motion_frame_count = local_rot_mats.shape[0]
    representation_local_rot_mats = local_rot_mats
    representation_root_positions = root_positions
    if motion_frame_count < 5:
        padding = 5 - motion_frame_count
        representation_local_rot_mats = torch.cat(
            (
                local_rot_mats,
                local_rot_mats[-1:].expand(padding, -1, -1, -1),
            ),
            dim=0,
        )
        representation_root_positions = torch.cat(
            (
                root_positions,
                root_positions[-1:].expand(padding, -1),
            ),
            dim=0,
        )
    # Measured 64D state has no absolute translation: its root positions are
    # exactly zero. The root smoother's exact solution is zero; avoid running
    # iterative optimization on that constant signal at each request.
    if torch.count_nonzero(representation_root_positions).item() == 0:
        from skeleton.kinematics import fk
        from motion.feature_utils import compute_heading_angle, compute_vel_xyz
        from motion.feet import foot_detect_from_pos_and_vel
        from utils.geometry import matrix_to_cont6d
        local = representation_local_rot_mats.unsqueeze(0)
        root = representation_root_positions.unsqueeze(0)
        rotations, positions, local_positions = fk(local, root, representation.skeleton)
        heading = compute_heading_angle(positions, representation.skeleton)
        lengths = torch.tensor([local.shape[1]], device=local.device)
        velocities = compute_vel_xyz(positions, representation.fps, lengths=lengths)
        contacts = foot_detect_from_pos_and_vel(positions, velocities, representation.skeleton, .15, .10)
        fields = (root, torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1),
                  local_positions, matrix_to_cont6d(rotations), velocities, contacts)
        features = torch.cat([field.flatten(start_dim=2) for field in fields], dim=-1)
        return features.squeeze(0).cpu()[:motion_frame_count]
    return representation(
        representation_local_rot_mats,
        representation_root_positions,
        to_normalize=False,
    ).cpu()[:motion_frame_count]


def _pad_configuration_for_decoder(joint_q, root_positions, root_rotations):
    torch = _load_torch()
    frame_count = joint_q.shape[0]
    if frame_count >= 5:
        return joint_q, root_positions, root_rotations, frame_count
    padding = 5 - frame_count
    return (
        torch.cat((joint_q, joint_q[-1:].expand(padding, -1)), dim=0),
        torch.cat(
            (root_positions, root_positions[-1:].expand(padding, -1)), dim=0
        ),
        torch.cat(
            (
                root_rotations,
                root_rotations[-1:].expand(padding, -1, -1),
            ),
            dim=0,
        ),
        frame_count,
    )


def _timestamp_source_positions(timestamps, target_fps: float, max_frames=None):
    """Sample backwards from the newest measurement, retaining float64 time."""
    times = np.asarray(timestamps, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("source timestamps must be finite and strictly increasing")
    relative = times - times[-1]
    frames = max(1, int(math.floor(-relative[0] * target_fps + 1e-5)) + 1)
    if max_frames is not None:
        frames = min(frames, max_frames)
    targets = -np.arange(frames - 1, -1, -1, dtype=np.float64) / target_fps
    return np.interp(targets, relative, np.arange(len(times), dtype=np.float64))


def _partial_real_world_state_history_to_motion(
    state_history,
    *,
    decoder,
    representation,
    source_fps: float,
    target_fps: float,
    source_timestamps=None,
    max_frames=None,
):
    torch = _load_torch()
    from motion.g1_reference import (
        CANONICAL_G1_JOINT_NAMES_29,
        _interpolate_motion,
        resample_motion,
        rot6d_row_to_matrix,
    )

    states = torch.as_tensor(state_history, dtype=torch.float32)
    if states.ndim != 2 or states.shape[1] != 64 or states.shape[0] == 0:
        raise ValueError(
            f"Expected real-world observation state history shape [T, 64], got {tuple(states.shape)}"
        )
    if not torch.isfinite(states).all():
        raise ValueError("real-world observation state history contains NaN or Inf")

    joint_q, root_positions, root_rotations, source_frame_count = (
        _pad_configuration_for_decoder(
            states[:, 6:35],
            torch.zeros(states.shape[0], 3, dtype=torch.float32),
            rot6d_row_to_matrix(states[:, :6]),
        )
    )
    decoded = decoder.decode_joint_configuration_pose(
        joint_q,
        root_positions,
        root_rotation_matrices=root_rotations,
        joint_names=CANONICAL_G1_JOINT_NAMES_29,
    )
    local_rot_mats = decoded["local_rot_mats"][:source_frame_count]
    root_positions = decoded["root_positions"][:source_frame_count]
    if source_timestamps is None:
        local_rot_mats, root_positions = resample_motion(
            local_rot_mats, root_positions,
            source_fps=source_fps, target_fps=target_fps,
        )
    else:
        if len(source_timestamps) != source_frame_count:
            raise ValueError("source timestamps must match state history length")
        positions = torch.as_tensor(
            _timestamp_source_positions(source_timestamps, target_fps, max_frames),
            dtype=torch.float32,
        )
        local_rot_mats, root_positions = _interpolate_motion(
            local_rot_mats, root_positions, positions,
        )
    motion = _kimodo_features_from_motion(
        local_rot_mats, root_positions, representation
    )
    feature_mask = torch.zeros_like(motion, dtype=torch.bool)
    feature_mask[:, representation.slice_dict["global_root_heading"]] = True
    feature_mask[:, representation.slice_dict["global_rot_data"]] = True
    return motion, feature_mask, local_rot_mats, root_positions


def _validated_text_embedding(value, *, cache_path: Path, task_id: str):
    torch = _load_torch()
    embedding = torch.as_tensor(value).detach().cpu().contiguous()
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(
            f"Expected text embedding [1, D] for task {task_id!r} in {cache_path}, "
            f"got {tuple(embedding.shape)}"
        )
    if embedding.shape[1] <= 0 or not torch.isfinite(embedding).all():
        raise ValueError(
            f"Text embedding for task {task_id!r} in {cache_path} is empty or non-finite"
        )
    return embedding


def _register_task_alias(aliases: dict[str, str], alias: str, task_id: str) -> None:
    alias = str(alias).strip()
    if not alias:
        return
    existing = aliases.get(alias)
    if existing is not None and existing != task_id:
        raise ValueError(
            f"Text cache alias {alias!r} maps to both {existing!r} and {task_id!r}"
        )
    aliases[alias] = task_id


def _load_text_embedding_cache(
    cache_location: str | os.PathLike,
):
    torch = _load_torch()
    cache_path = Path(cache_location).expanduser().resolve()
    if cache_path.is_dir():
        cache_files = sorted(cache_path.glob("*.pt"))
        if not cache_files:
            raise RuntimeError(f"No text embedding .pt files found below {cache_path}")
    elif cache_path.is_file():
        cache_files = [cache_path]
    else:
        raise FileNotFoundError(f"Text embedding cache does not exist: {cache_path}")

    embeddings: dict[str, Any] = {}
    instructions: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for file_path in cache_files:
        payload = torch.load(file_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f"Text embedding cache must contain a dictionary: {file_path}")
        task_id = str(payload.get("task_id", "")).strip()
        instruction = str(payload.get("instruction", "")).strip()
        if not task_id or not instruction or "embedding" not in payload:
            raise ValueError(
                f"Real-world text cache requires task_id, instruction, and embedding: {file_path}"
            )
        source = str(payload.get("source", "")).strip()
        if source and source != REAL_WORLD_TEXT_CACHE_SOURCE:
            raise ValueError(
                f"Expected a {REAL_WORLD_TEXT_CACHE_SOURCE} text cache, got "
                f"source={source!r}: {file_path}"
            )
        embedding = _validated_text_embedding(
            payload["embedding"], cache_path=file_path, task_id=task_id
        )
        if task_id in embeddings:
            if instructions[task_id] != instruction or not torch.equal(
                embeddings[task_id], embedding
            ):
                raise ValueError(
                    f"Conflicting text embedding cache entries for task {task_id!r}"
                )
            continue
        task_name = str(payload.get("task_name", "")).strip()
        embeddings[task_id] = embedding
        instructions[task_id] = instruction
        _register_task_alias(aliases, task_id, task_id)
        if "::" in task_id:
            _register_task_alias(aliases, task_id.split("::", 1)[1], task_id)
        if task_name:
            _register_task_alias(aliases, task_name, task_id)

    if not embeddings:
        raise RuntimeError(f"No text embeddings found in {cache_path}")
    return embeddings, instructions, aliases


def _resolve_cached_task(task: str, aliases: dict[str, str]) -> str:
    task = str(task).strip()
    if task in aliases:
        return aliases[task]
    prefixed = f"{REAL_WORLD_TEXT_CACHE_SOURCE}::{task}"
    if prefixed in aliases:
        return aliases[prefixed]
    available = ", ".join(sorted(aliases))
    raise KeyError(f"No cached embedding for task {task!r}; available aliases: {available}")


def _dtype_from_wire(value: Any) -> np.dtype:
    try:
        dtype = np.dtype(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid image dtype: {value!r}") from exc
    if dtype.kind not in {"u", "i", "f"} or dtype.itemsize > 4:
        raise ValueError(f"unsupported image dtype: {dtype}")
    return dtype


def _validate_image_payload(image: Any) -> None:
    if not isinstance(image, dict):
        raise ValueError("observation.images.front must be an object")
    shape = image.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("image shape must be [height, width, channels]")
    try:
        shape_tuple = tuple(int(v) for v in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("image shape contains a non-integer") from exc
    h, w, channels = shape_tuple
    if h <= 0 or w <= 0 or channels not in (3, 4):
        raise ValueError(f"unsupported image shape: {shape_tuple}")
    # Keep malformed/hostile payloads out of torch before base64 decoding.
    dtype = _dtype_from_wire(image.get("dtype"))
    encoded = image.get("data_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image data_b64 is missing")
    expected = int(np.prod(shape_tuple)) * dtype.itemsize
    decode_image_bytes(image, expected)
    if dtype != np.dtype(np.uint8):
        raise ValueError(
            f"real-world camera input must be uint8 RGB, got wire dtype {dtype}"
        )


def _validate_infer_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("/infer payload must be a JSON object")
    task = str(payload.get("task", "")).strip()
    if not task:
        raise ValueError("/infer requires a non-empty task")
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("/infer requires observation")
    images = observation.get("images")
    if not isinstance(images, dict) or "front" not in images:
        raise ValueError("observation.images.front is required")
    _validate_image_payload(images["front"])

    state = np.asarray(observation.get("state"), dtype=np.float32)
    if state.shape != (64,) or not np.isfinite(state).all():
        raise ValueError("observation.state must be finite shape [64]")
    history = np.asarray(observation.get("state_history"), dtype=np.float32)
    if history.ndim != 2 or history.shape[1:] != (64,) or history.shape[0] == 0:
        raise ValueError("observation.state_history must have shape [T, 64]")
    if not np.isfinite(history).all():
        raise ValueError("observation.state_history contains NaN or Inf")


def _validate_continuous_hand_payload(payload: Any) -> np.ndarray:
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("/infer requires observation")
    hand_history = observation.get("hand_closure_history")
    if hand_history is None:
        raise ValueError(
            "continuous hand observation mode requires "
            "observation.hand_closure_history"
        )
    hand = _validated_hand_closure_history(hand_history)
    state_history = np.asarray(observation.get("state_history"), dtype=np.float32)
    if state_history.ndim == 1:
        state_frames = 1
    elif state_history.ndim == 2 and state_history.shape[1:] == (64,):
        state_frames = state_history.shape[0]
    else:
        raise ValueError("observation.state_history must have shape [T, 64]")
    if hand.shape[0] != state_frames:
        raise ValueError(
            "observation.state_history and observation.hand_closure_history "
            "must contain the same number of frames"
        )
    return hand


class KimodoRealWorldRuntime:
    """Real-world checkpoint runtime used by the SONIC bridge."""

    def __init__(self, args: argparse.Namespace):
        torch = _load_torch()
        from model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
        from motion.g1_reference import HumanoidArenaActionDecoder
        from motion.representation.kimodo_motionrep import KimodoMotionRep
        from skeleton.definitions import G1Skeleton34

        self.deterministic_eval = _env_flag("KIMODO_DETERMINISTIC_EVAL")
        if self.deterministic_eval:
            _configure_deterministic_torch()
        checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
        self.checkpoint_dir = checkpoint_dir
        checkpoint_config = json.loads((checkpoint_dir / "config.json").read_text())
        model_config = checkpoint_config["model"]
        main_config = checkpoint_config["main"]
        loss_config = checkpoint_config["training"]["loss"]
        mse_weights = loss_config.get("mse_weights", {})
        self.hand_control_mode = _resolve_hand_control_mode(
            model_config.get("hand_control_mode", "binary")
        )
        self.hand_observation_mode = _resolve_hand_observation_mode(
            model_config.get("hand_observation_mode"),
            self.hand_control_mode,
        )
        config = KimodoPolicyConfig(
            fps=int(model_config["fps"]),
            motion_mask_mode=str(model_config["motion_mask_mode"]),
            dinov3_model_name=str(model_config["dinov3_model_name"]),
            dinov3_checkpoint=model_config.get("dinov3_checkpoint"),
            action_chunk=int(main_config["action_chunk"]),
            action_history=int(main_config["action_history"]),
            load_text_encoder=False,
            controlnet_num_layers=int(model_config.get("controlnet_num_layers", 8)),
            detach_root_control_for_body=bool(
                model_config.get("detach_root_control_for_body", False)
            ),
            control_fusion_mode=str(
                model_config.get("control_fusion_mode", "both")
            ),
            root_loss_weight=float(
                mse_weights.get("root", loss_config.get("root_weight", 2.0))
            ),
            body_loss_weight=float(
                mse_weights.get("body", loss_config.get("body_weight", 1.0))
            ),
            enable_hand_head=bool(model_config.get("enable_hand_head", True)),
            hand_hidden_dim=int(model_config.get("hand_hidden_dim", 256)),
            hand_num_layers=int(model_config.get("hand_num_layers", 4)),
            hand_num_heads=int(model_config.get("hand_num_heads", 4)),
            hand_ffn_dim=int(model_config.get("hand_ffn_dim", 1024)),
            hand_loss_weight=float(
                mse_weights.get("hand", loss_config.get("hand_weight", 1.0))
            ),
            hand_transition_loss_weight=float(
                mse_weights.get(
                    "hand_transition",
                    loss_config.get("hand_transition_weight", 0.2),
                )
            ),
            hand_init_seed=int(model_config.get("hand_init_seed", 3407)),
            hand_control_mode=self.hand_control_mode,
            hand_observation_mode=self.hand_observation_mode,
        )
        self.device = _validate_torch_device(torch, args.device)
        self.dtype = _dtype_from_name(args.dtype)
        self.hand_rtc_enabled = bool(args.hand_rtc)
        self.hand_stabilization = bool(args.hand_stabilization)
        self.hand_close_threshold = float(args.hand_close_threshold)
        self.hand_open_threshold = float(args.hand_open_threshold)
        self.hand_close_frames = int(args.hand_close_frames)
        self.hand_open_frames = int(args.hand_open_frames)
        stabilize_binary_hand(np.empty((0, 2)), [0, 0],
            close_threshold=self.hand_close_threshold, open_threshold=self.hand_open_threshold,
            close_frames=self.hand_close_frames, open_frames=self.hand_open_frames)
        self.hand_filter_state = None
        self.diffusion_steps = int(args.diffusion_steps)
        self.control_fps = float(args.control_fps)
        if self.control_fps <= 0 or not np.isfinite(self.control_fps):
            raise ValueError("control_fps must be finite and positive")
        self.execution_frames = _resolve_execution_frames(
            args.execution_frames, config.action_chunk
        )
        (
            self.rtc_enabled,
            self.rtc_overlap_frames,
            self.rtc_frozen_frames,
            self.rtc_ramp_power,
        ) = _resolve_rtc_parameters(
            bool(args.rtc),
            prediction_frames=config.action_chunk,
            execution_frames=self.execution_frames,
            overlap_frames=args.rtc_overlap_frames,
            frozen_frames=args.rtc_frozen_frames,
            ramp_power=args.rtc_ramp_power,
        )
        (
            self.text_embeddings,
            self.task_instructions,
            self.task_aliases,
        ) = _load_text_embedding_cache(args.text_embedding_cache)
        self.served_task = args.served_task
        if self.served_task is not None:
            _resolve_cached_task(self.served_task, self.task_aliases)
        self.model = KimodoPolicy(config)
        loaded_step = self.model.load_controlnet_checkpoint(str(checkpoint_dir))
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.action_codec = HumanoidArenaActionDecoder(
            G1Skeleton34(), G1_XML_PATH, fps=self.control_fps
        )
        self.state_motion_representation = KimodoMotionRep(
            G1Skeleton34(), fps=float(self.model.fps), stats_path=None
        )
        self.observation_state_history = np.empty((0, 64), dtype=np.float32)
        self.observation_hand_closure_history = np.empty((0, 2), dtype=np.float32)
        self.observation_state_sequences = np.empty(0, dtype=np.int64)
        self.observation_state_timestamps = np.empty(0, dtype=np.float64)
        self.deployment_session_id = None
        self.last_deployment = None
        self.history_motion = torch.empty(
            1,
            0,
            self.model.representation.motion_rep_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.history_hand = torch.empty(
            1,
            0,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        self.previous_local_rot_mat = None
        self.previous_root_position = None
        self.previous_hand_command = None
        self.rtc_motion_tail = None
        self.rtc_hand_tail = None
        self.rtc_task_id = None
        self.episode_seed: int | None = None
        self.inference_index = 0
        self.lock = threading.Lock()
        print(
            f"Loaded real-world checkpoint step={loaded_step} on {self.device} "
            f"dtype={self.dtype} diffusion_steps={self.diffusion_steps} "
            f"synchronized_hand_diffusion={config.enable_hand_head} "
            f"prediction_frames={config.action_chunk} "
            f"execution_frames={self.execution_frames} "
            f"rtc_enabled={self.rtc_enabled} "
            f"rtc_overlap_frames={self.rtc_overlap_frames} "
            f"rtc_frozen_frames={self.rtc_frozen_frames} "
            f"rtc_ramp_power={self.rtc_ramp_power:g} "
            f"hand_control_mode={self.hand_control_mode} "
            f"hand_observation_mode={self.hand_observation_mode} "
            f"deterministic_eval={self.deterministic_eval} "
            f"text_tasks={sorted(self.text_embeddings)}",
            flush=True,
        )

    @property
    def model_fps(self) -> float:
        return float(self.model.fps)

    @property
    def prediction_frames(self) -> int:
        return int(self.model.config.action_chunk)

    @property
    def execution_control_frames(self) -> int:
        return max(1, int(round(self.execution_frames * self.control_fps / self.model_fps)))

    @property
    def history_control_frames(self) -> int:
        return math.ceil((self.model.config.action_history + 2) * self.control_fps / self.model_fps) + 2

    def reset(self, seed: Any = None, session_id: str | None = None) -> None:
        torch = _load_torch()
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128
        ):
            raise ValueError("reset session_id must be a non-empty string of at most 128 characters")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("reset seed must be an integer") from exc
        with self.lock:
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                self.episode_seed = seed
            self.inference_index = 0
            self.observation_state_history = np.empty((0, 64), dtype=np.float32)
            self.observation_hand_closure_history = np.empty((0, 2), dtype=np.float32)
            self.observation_state_sequences = np.empty(0, dtype=np.int64)
            self.observation_state_timestamps = np.empty(0, dtype=np.float64)
            self.deployment_session_id = session_id
            self.last_deployment = None
            self.history_motion = torch.empty(
                1,
                0,
                self.model.representation.motion_rep_dim,
                device=self.device,
                dtype=torch.float32,
            )
            self.history_hand = torch.empty(
                1,
                0,
                2,
                device=self.device,
                dtype=torch.float32,
            )
            self.previous_local_rot_mat = None
            self.previous_root_position = None
            self.previous_hand_command = None
            self.hand_filter_state = None
            self.rtc_motion_tail = None
            self.rtc_hand_tail = None
            self.rtc_task_id = None

    @staticmethod
    def _decode_rgb(image_payload: dict):
        torch = _load_torch()
        shape = tuple(int(value) for value in image_payload["shape"])
        dtype = np.dtype(image_payload["dtype"])
        raw = decode_image_bytes(image_payload, math.prod(shape) * dtype.itemsize)
        rgb = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA image, got {rgb.shape}")
        rgb = np.array(rgb[..., :3], copy=True, order="C")
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    def infer(self, payload: dict[str, Any]) -> np.ndarray | dict[str, Any]:
        infer_started = time.perf_counter()
        torch = _load_torch()
        from motion.g1_reference import (
            resample_hand_binary_chunk,
            resample_hand_continuous_chunk,
            resample_hand_continuous,
            resample_motion_chunk,
        )

        _validate_infer_payload(payload)
        observation = payload["observation"]
        deployment = (
            validate_deployment(payload["deployment"])
            if "deployment" in payload else None
        )
        next_sequences = np.empty(0, dtype=np.int64)
        next_timestamps = np.empty(0, dtype=np.float64)
        if deployment is not None:
            _, next_sequences, next_timestamps, _ = validate_observation_window(
                observation, self.history_control_frames,
            )
        hand_closure_history_segment = None
        if self.hand_observation_mode == "continuous":
            hand_closure_history_segment = _validate_continuous_hand_payload(payload)
        task = str(payload.get("task", "")).strip()
        task_id = _resolve_cached_task(task, self.task_aliases)
        if self.served_task is not None and task_id != _resolve_cached_task(
            self.served_task, self.task_aliases
        ):
            raise ValueError(f"Server is configured for {self.served_task}, not {task}")
        image = self._decode_rgb(observation["images"]["front"]).to(self.device)
        state_history_segment = observation.get("state_history")
        text_feature = self.text_embeddings[task_id].to(dtype=self.dtype)
        instruction = self.task_instructions[task_id]

        with self.lock, torch.inference_mode():
            session_id = getattr(self, "deployment_session_id", None)
            last_deployment = getattr(self, "last_deployment", None)
            if deployment is None and session_id is not None:
                raise ValueError("synchronized session requires deployment metadata; legacy inference is disabled")
            continuing_parent = deployment is None
            if deployment is not None:
                if session_id is not None and session_id != deployment["session_id"]:
                    raise ValueError("deployment session does not match the bound session")
                if last_deployment is not None:
                    if deployment["epoch"] < last_deployment["epoch"]:
                        raise ValueError("deployment epoch cannot decrease")
                    if deployment["request_id"] <= last_deployment["request_id"]:
                        raise ValueError("deployment request_id must increase")
                    if (
                        deployment["epoch"] == last_deployment["epoch"]
                        and len(self.observation_state_sequences)
                        and (
                            next_sequences[-1] < self.observation_state_sequences[-1]
                            or next_timestamps[-1] < self.observation_state_timestamps[-1]
                        )
                    ):
                        raise ValueError("observation window cannot move backwards within an epoch")
                    continuing_parent = (
                        deployment["session_id"] == last_deployment["session_id"]
                        and deployment["epoch"] == last_deployment["epoch"]
                        and deployment["parent_request_id"] == last_deployment["request_id"]
                        and task_id == last_deployment["task_id"]
                    )
            command_boundary = None
            if deployment is not None and "boundary_action" in deployment:
                command_boundary = self.action_codec.decode_action_pose(
                    np.asarray(deployment["boundary_action"], dtype=np.float32).reshape(1, 40)
                )
            history_motion_snapshot = self.history_motion.clone()
            history_hand_snapshot = self.history_hand.clone()
            observation_state_history_snapshot = np.array(
                self.observation_state_history, dtype=np.float32, copy=True
            )
            observation_hand_closure_history_snapshot = np.array(
                getattr(
                    self,
                    "observation_hand_closure_history",
                    np.empty((0, 2), dtype=np.float32),
                ),
                dtype=np.float32,
                copy=True,
            )
            previous_local_rot_mat_snapshot = (
                None
                if self.previous_local_rot_mat is None
                else self.previous_local_rot_mat.clone()
            )
            previous_root_position_snapshot = (
                None
                if self.previous_root_position is None
                else self.previous_root_position.clone()
            )
            previous_hand_snapshot = getattr(self, "previous_hand_command", None)
            if previous_hand_snapshot is not None:
                previous_hand_snapshot = previous_hand_snapshot.clone()
            rtc_motion_tail_snapshot = (
                None
                if self.rtc_motion_tail is None
                else self.rtc_motion_tail.clone()
            )
            rtc_hand_tail_snapshot = (
                None
                if self.rtc_hand_tail is None
                else self.rtc_hand_tail.clone()
            )
            rtc_task_id_snapshot = self.rtc_task_id
            if not continuing_parent:
                # A generated candidate is speculative until the client names
                # it as its accepted parent. Rebuild after drop/clear/rebase.
                history_hand_snapshot = history_hand_snapshot[:, :0]
                previous_local_rot_mat_snapshot = None
                previous_root_position_snapshot = None
                previous_hand_snapshot = None
                rtc_motion_tail_snapshot = None
                rtc_hand_tail_snapshot = None
            use_rtc_reference = (
                self.rtc_enabled
                and rtc_motion_tail_snapshot is not None
                and rtc_task_id_snapshot == task_id
            )
            use_rtc_hand_reference = (
                use_rtc_reference and self.hand_rtc_enabled
                and self.hand_control_mode != "continuous"
            )
            hand_filter_snapshot = self.hand_filter_state if continuing_parent else None
            history_feature_mask_snapshot = None
            next_observation_state_history = observation_state_history_snapshot
            next_observation_hand_closure_history = (
                observation_hand_closure_history_snapshot
            )
            motion_previous_local_rot_mat = previous_local_rot_mat_snapshot
            motion_previous_root_position = previous_root_position_snapshot

            if state_history_segment is None:
                raise ValueError(
                    "Real-world inference requires observation.state_history at "
                    "every replanning request"
                )
            segment = np.asarray(state_history_segment, dtype=np.float32)
            if segment.ndim == 1:
                segment = segment.reshape(1, -1)
            if segment.ndim != 2 or segment.shape[1] != 64 or segment.shape[0] == 0:
                raise ValueError("observation.state_history must have shape [T, 64]")
            if not np.isfinite(segment).all():
                raise ValueError("observation.state_history contains NaN or Inf")
            next_observation_state_history = (
                segment.copy() if deployment is not None else np.concatenate(
                    (observation_state_history_snapshot, segment), axis=0
                )
            )
            if self.hand_observation_mode == "continuous":
                assert hand_closure_history_segment is not None
                next_observation_hand_closure_history = (
                    hand_closure_history_segment.copy()
                    if deployment is not None else np.concatenate(
                        (observation_hand_closure_history_snapshot, hand_closure_history_segment),
                        axis=0,
                    )
                )
                if (
                    next_observation_hand_closure_history.shape[0]
                    != next_observation_state_history.shape[0]
                ):
                    raise RuntimeError(
                        "Measured hand and body histories lost frame alignment"
                    )
            (
                state_history_motion,
                state_history_feature_mask,
                state_local_rot_mats,
                state_root_positions,
            ) = _partial_real_world_state_history_to_motion(
                next_observation_state_history,
                decoder=self.action_codec,
                representation=self.state_motion_representation,
                source_fps=self.control_fps,
                target_fps=float(self.model.fps),
                source_timestamps=next_timestamps if deployment is not None else None,
                max_frames=self.model.config.action_history + 1,
            )
            # Before two model-rate samples exist, bootstrap conditioning with
            # the one measured pose held in history. Observation IDs/times stay
            # untouched; an empty condition would discard known pose/hand state.
            condition_stop = -1 if state_history_motion.shape[0] > 1 else None
            condition_history_motion = state_history_motion[:condition_stop]
            condition_history_feature_mask = state_history_feature_mask[:condition_stop]
            condition_history_motion = condition_history_motion[
                -self.model.config.action_history :
            ]
            history_motion_snapshot = condition_history_motion.unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            history_feature_mask_snapshot = condition_history_feature_mask[
                -self.model.config.action_history :
            ].unsqueeze(0).to(device=self.device)
            if self.hand_observation_mode == "continuous":
                if deployment is not None:
                    positions = _timestamp_source_positions(
                        next_timestamps, self.model_fps, self.model.config.action_history + 1,
                    )
                    measured_hand_history = np.stack([
                        np.interp(positions, np.arange(len(next_timestamps)), column)
                        for column in next_observation_hand_closure_history.T
                    ], axis=1).astype(np.float32)
                else:
                    measured_hand_history = resample_hand_continuous(
                        next_observation_hand_closure_history,
                        source_fps=self.control_fps,
                        target_fps=float(self.model.fps),
                    )
                measured_hand_condition = measured_hand_history[:condition_stop]
                measured_hand_condition = measured_hand_condition[
                    -self.model.config.action_history :
                ]
                model_hand_history_snapshot = torch.as_tensor(
                    measured_hand_condition,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                model_hand_history_snapshot = _align_hand_history(
                    model_hand_history_snapshot,
                    history_motion_snapshot.shape[1],
                )
            else:
                model_hand_history_snapshot = _align_hand_history(
                    history_hand_snapshot, history_motion_snapshot.shape[1]
                )
            if command_boundary is not None:
                motion_previous_local_rot_mat = command_boundary["local_rot_mats"][0].clone()
            elif motion_previous_local_rot_mat is None:
                motion_previous_local_rot_mat = state_local_rot_mats[-1].clone()
            if motion_previous_root_position is None:
                motion_previous_root_position = state_root_positions[-1].clone()

            generator = None
            noise_seed = None
            if self.deterministic_eval:
                if self.episode_seed is None:
                    raise RuntimeError("model server was not reset with an episode seed")
                noise_seed = _derive_inference_seed(
                    self.episode_seed, self.inference_index
                )
                generator = torch.Generator(device=self.device)
                generator.manual_seed(noise_seed)
            model_started = time.perf_counter()
            output = self.model.predict_future(
                instruction=instruction,
                egoview=image,
                history_motion=history_motion_snapshot,
                history_feature_mask=history_feature_mask_snapshot,
                diffusion_steps=self.diffusion_steps,
                squeeze_batch=True,
                text_feat=text_feature,
                hand_history=model_hand_history_snapshot,
                generator=generator,
                rtc_motion_reference=(
                    rtc_motion_tail_snapshot if use_rtc_reference else None
                ),
                rtc_hand_reference=(
                    rtc_hand_tail_snapshot if use_rtc_hand_reference else None
                ),
                rtc_overlap_frames=self.rtc_overlap_frames,
                rtc_frozen_frames=self.rtc_frozen_frames,
                rtc_ramp_power=self.rtc_ramp_power,
            )
            full_future_features = output["motion_features"].detach().to(
                device=self.device, dtype=torch.float32
            )
            full_source_local_rot_mats = (
                output["local_rot_mats"].detach().float().cpu()
            )
            full_source_root_positions = (
                output["root_positions"].detach().float().cpu()
            )
            model_finished = time.perf_counter()
            if not (
                full_future_features.ndim == 2
                and full_source_local_rot_mats.shape[0]
                == full_source_root_positions.shape[0]
                == full_future_features.shape[0]
            ):
                raise ValueError("Model returned inconsistent full prediction shapes")
            if not all(torch.isfinite(value).all() for value in (
                full_future_features, full_source_local_rot_mats, full_source_root_positions,
            )):
                raise ValueError("Model returned NaN/Inf prediction")

            next_rtc_motion_tail = None
            next_rtc_hand_tail = None
            next_rtc_task_id = None
            if self.rtc_enabled:
                rtc_tail_start = self.execution_frames
                rtc_tail_end = min(
                    full_future_features.shape[0],
                    rtc_tail_start + self.rtc_overlap_frames,
                )
                if rtc_tail_end > rtc_tail_start:
                    next_rtc_motion_tail = full_future_features[
                        rtc_tail_start:rtc_tail_end
                    ].detach().clone()
                    next_rtc_task_id = task_id
                    full_hand_clean = output.get("hand_clean")
                    if self.hand_control_mode == "continuous":
                        full_hand_clean = None
                    if full_hand_clean is not None:
                        full_hand_clean = (
                            full_hand_clean.detach().to(
                                device=self.device, dtype=torch.float32
                            )
                        )
                        if (
                            full_hand_clean.ndim != 2
                            or full_hand_clean.shape[1] != 2
                            or full_hand_clean.shape[0]
                            != full_future_features.shape[0]
                        ):
                            raise ValueError(
                                "Model returned inconsistent clean hand prediction shape"
                            )
                        if not torch.isfinite(full_hand_clean).all():
                            raise ValueError("Model returned NaN/Inf clean hand prediction")
                        next_rtc_hand_tail = full_hand_clean[
                            rtc_tail_start:rtc_tail_end
                        ].detach().clone()

            future_features = full_future_features
            source_local_rot_mats = full_source_local_rot_mats
            source_root_positions = full_source_root_positions
            history_last_root_position = output.get("history_last_root_position")
            if history_last_root_position is not None:
                history_last_root_position = (
                    history_last_root_position.detach().float().cpu().reshape(3)
                )
                if not torch.isfinite(history_last_root_position).all():
                    raise ValueError(
                        "Predicted history boundary root contains NaN or Inf"
                    )
                motion_previous_root_position = history_last_root_position
            elif deployment is not None:
                # Root XY deltas are encoded in this prediction window's gauge;
                # an earlier prediction's planar origin cannot anchor them.
                motion_previous_root_position = source_root_positions[0].clone()
            if command_boundary is not None:
                motion_previous_root_position = motion_previous_root_position.clone()
                # Kimodo uses Y up. The boundary action's XY entries are local
                # displacements, so only its absolute height is meaningful here.
                motion_previous_root_position[1] = command_boundary["root_positions"][0, 1]
            future_features, source_local_rot_mats, source_root_positions = (
                _select_execution_prefix(
                    future_features,
                    source_local_rot_mats,
                    source_root_positions,
                    self.execution_frames,
                )
            )
            hand_output_key = (
                "hand_closure"
                if self.hand_control_mode == "continuous"
                else "hand_binary"
            )
            source_hand = output.get(hand_output_key)
            if source_hand is None and self.hand_control_mode == "continuous":
                raise RuntimeError(
                    "Continuous real-world checkpoint did not return hand_closure"
                )
            if command_boundary is not None:
                previous_hand = torch.as_tensor(
                    deployment["boundary_action"][38:40],
                    dtype=torch.float32,
                )
                if self.hand_control_mode == "binary":
                    previous_hand = (previous_hand >= 0.5).float()
            elif previous_hand_snapshot is not None:
                previous_hand = previous_hand_snapshot
            elif self.hand_control_mode == "continuous":
                previous_hand = torch.as_tensor(
                    next_observation_hand_closure_history[-1], dtype=torch.float32,
                )
            else:
                previous_hand = (
                    (history_hand_snapshot[0, -1].detach().float().cpu() >= 0.5).float()
                    if history_hand_snapshot.shape[1] > 0 and self.hand_observation_mode == "binary"
                    else torch.zeros(2, dtype=torch.float32)
                )
            if source_hand is not None:
                source_hand = _select_hand_execution_prefix(
                    _prediction_hand_tensor(source_hand, hand_output_key),
                    self.execution_frames,
                )
            next_hand_filter_state = None
            hand_diagnostics = {"rtc_hand_reference_used": bool(use_rtc_hand_reference)}
            if self.hand_control_mode == "binary" and source_hand is not None:
                probabilities = _select_hand_execution_prefix(
                    _prediction_hand_tensor(output["hand_closure"], "hand_closure"),
                    self.execution_frames).detach().float().cpu().numpy()
                raw_binary = source_hand.detach().float().cpu().numpy()
                if self.hand_stabilization:
                    filtered, next_hand_filter_state = stabilize_binary_hand(
                        probabilities, previous_hand.detach().float().cpu().numpy(),
                        state=hand_filter_snapshot,
                        close_threshold=self.hand_close_threshold,
                        open_threshold=self.hand_open_threshold,
                        close_frames=self.hand_close_frames, open_frames=self.hand_open_frames)
                    source_hand = torch.as_tensor(filtered, dtype=source_hand.dtype,
                                                  device=source_hand.device)
                hand_diagnostics.update(
                    hand_closure_before_filter=probabilities.tolist(),
                    hand_binary_before_filter=raw_binary.tolist(),
                    hand_binary_after_filter=source_hand.detach().float().cpu().tolist())
            next_history_motion = history_motion_snapshot
            if self.hand_observation_mode == "continuous":
                # The next prediction must be conditioned on what the robot
                # measured, not on the closure just emitted by the model.
                next_history_hand = model_hand_history_snapshot.detach().clone()
            else:
                next_history_hand = history_hand_snapshot
            if source_hand is not None and self.hand_observation_mode != "continuous":
                next_history_hand = torch.cat(
                    (
                        history_hand_snapshot,
                        source_hand.to(self.device).unsqueeze(0),
                    ),
                    dim=1,
                )[:, -self.model.config.action_history :]

            local_rot_mats, root_positions = resample_motion_chunk(
                source_local_rot_mats,
                source_root_positions,
                source_fps=float(self.model.fps),
                target_fps=self.control_fps,
                previous_local_rot_mat=motion_previous_local_rot_mat,
                previous_root_position=motion_previous_root_position,
            )
            control_hand = None
            if source_hand is not None:
                if self.hand_control_mode == "continuous":
                    control_hand = resample_hand_continuous_chunk(
                        source_hand,
                        source_fps=float(self.model.fps),
                        target_fps=self.control_fps,
                        previous_hand_closure=previous_hand,
                    )
                else:
                    control_hand = resample_hand_binary_chunk(
                        source_hand,
                        source_fps=float(self.model.fps),
                        target_fps=self.control_fps,
                        previous_hand_binary=previous_hand,
                    )
                if control_hand.shape[0] != local_rot_mats.shape[0]:
                    raise RuntimeError(
                        "Motion and hand control chunks have different lengths: "
                        f"{local_rot_mats.shape[0]} and {control_hand.shape[0]}"
                    )
            action_chunk = self.action_codec.encode(
                local_rot_mats,
                root_positions,
                previous_root_position=motion_previous_root_position,
                # The codec field is named for the Arena action schema; in
                # continuous real-world mode these two dims carry closure.
                hand_binary=control_hand,
            )
            action_chunk_numpy = action_chunk.detach().cpu().numpy().astype(np.float32)
            if action_chunk_numpy.ndim != 2 or action_chunk_numpy.shape[1] != 40:
                raise RuntimeError(
                    f"Kimodo returned invalid action shape {action_chunk_numpy.shape}"
                )
            if not np.isfinite(action_chunk_numpy).all():
                raise RuntimeError("Kimodo returned NaN/Inf action")
            next_previous_local_rot_mat = source_local_rot_mats[-1].clone()
            next_previous_root_position = source_root_positions[-1].clone()
            next_previous_hand_command = source_hand[-1].clone() if source_hand is not None else None
            response = action_chunk_numpy
            next_deployment = None
            if deployment is not None:
                identity = deployment_identity(deployment)
                next_deployment = {**identity, "task_id": task_id}
                next_sequences = next_sequences.copy()
                next_timestamps = next_timestamps.copy()
                response = {
                    "action_chunk": action_chunk_numpy.tolist(),
                    "deployment": identity,
                    "diagnostics": {**hand_diagnostics, "rtc_reference_used": bool(use_rtc_reference),
                        "preprocess_s": model_started-infer_started,
                        "model_and_transfer_s": model_finished-model_started,
                        "postprocess_s": time.perf_counter()-model_finished,
                        "runtime_total_s": time.perf_counter()-infer_started},
                }
            if self.deterministic_eval:
                print(
                    "[deterministic_real_world_inference] "
                    f"episode_seed={self.episode_seed} "
                    f"inference_index={self.inference_index} "
                    f"noise_seed={noise_seed} "
                    f"rtc_reference_used={use_rtc_reference}",
                    flush=True,
                )

            self.observation_state_history = next_observation_state_history
            self.observation_hand_closure_history = (
                next_observation_hand_closure_history
            )
            self.history_motion = next_history_motion
            self.history_hand = next_history_hand
            self.previous_local_rot_mat = next_previous_local_rot_mat
            self.previous_root_position = next_previous_root_position
            self.previous_hand_command = next_previous_hand_command
            self.hand_filter_state = next_hand_filter_state
            self.rtc_motion_tail = next_rtc_motion_tail
            self.rtc_hand_tail = next_rtc_hand_tail
            self.rtc_task_id = next_rtc_task_id
            if deployment is not None:
                self.observation_state_sequences = next_sequences
                self.observation_state_timestamps = next_timestamps
                self.deployment_session_id = deployment["session_id"]
                self.last_deployment = next_deployment
            if self.deterministic_eval:
                self.inference_index += 1
            return response


class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    runtime: KimodoRealWorldRuntime | Any = None
    max_request_bytes = MAX_REQUEST_BYTES

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if status >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length <= 0:
            raise ValueError("request body is empty")
        if length > self.max_request_bytes:
            raise ValueError(
                f"request body exceeds {self.max_request_bytes} bytes"
            )
        raw = self.rfile.read(length)
        encoding = self.headers.get("Content-Encoding", "identity").lower()
        if encoding == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                    raw = stream.read(self.max_request_bytes + 1)
            except (OSError, EOFError) as exc:
                raise ValueError("invalid gzip request") from exc
            if len(raw) > self.max_request_bytes:
                raise ValueError("decompressed request exceeds byte limit")
        elif encoding != "identity":
            raise ValueError("unsupported Content-Encoding")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in {"/", "/health"}:
            runtime = self.runtime
            fps = getattr(runtime, "control_fps", None)
            hand_control_mode = getattr(runtime, "hand_control_mode", None)
            hand_observation_mode = getattr(
                runtime,
                "hand_observation_mode",
                hand_control_mode,
            )
            self._send_json(
                200,
                {
                    "status": "ok",
                    "mode": "real_world",
                    "control_fps": float(fps) if fps is not None else None,
                    "state_dim": 64,
                    "action_dim": 40,
                    "hand_control_mode": hand_control_mode,
                    "hand_observation_mode": hand_observation_mode,
                    "deployment_protocol_version": PROTOCOL_VERSION,
                    "served_task": getattr(runtime, "served_task", None),
                    "checkpoint": str(getattr(runtime, "checkpoint_dir", "")),
                    "model_fps": getattr(runtime, "model_fps", None),
                    "prediction_frames": getattr(runtime, "prediction_frames", None),
                    "diffusion_steps": getattr(runtime, "diffusion_steps", None),
                    "image_compression": ["none", "zlib"],
                    "execution_frames": getattr(runtime, "execution_frames", None),
                    "execution_control_frames": getattr(runtime, "execution_control_frames", None),
                    "history_control_frames": getattr(runtime, "history_control_frames", None),
                    "rtc_enabled": getattr(runtime, "rtc_enabled", None),
                    "hand_rtc_enabled": getattr(runtime, "hand_rtc_enabled", None),
                    "hand_stabilization": getattr(runtime, "hand_stabilization", None),
                    "hand_close_threshold": getattr(runtime, "hand_close_threshold", None),
                    "hand_open_threshold": getattr(runtime, "hand_open_threshold", None),
                    "hand_close_frames": getattr(runtime, "hand_close_frames", None),
                    "hand_open_frames": getattr(runtime, "hand_open_frames", None),
                    "rtc_overlap_frames": getattr(runtime, "rtc_overlap_frames", None),
                    "rtc_frozen_frames": getattr(runtime, "rtc_frozen_frames", None),
                    "rtc_ramp_power": getattr(runtime, "rtc_ramp_power", None),
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            read_started = time.perf_counter()
            payload = self._read_json()
            read_finished = time.perf_counter()
            if path == "/reset":
                if not isinstance(payload, dict):
                    raise ValueError("/reset payload must be a JSON object")
                self.runtime.reset(payload.get("seed"), session_id=payload.get("session_id"))
                self._send_json(200, {"status": "ok"})
                return
            if path == "/infer":
                # Runtime owns validation, including direct non-HTTP callers.
                action = self.runtime.infer(payload)
                if isinstance(action, dict):
                    action.setdefault("diagnostics", {}).update(read_decode_s=read_finished-read_started, handler_total_s=time.perf_counter()-read_started)
                self._send_json(
                    200, action if isinstance(action, dict) else {"action_chunk": action.tolist()},
                )
                return
            self._send_json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:  # never leak a traceback into the wire protocol
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--policy-path", dest="checkpoint", required=True)
    parser.add_argument("--served-task", default=None,
                        help="Task ID expected by the bridge, e.g. teleop-boxing")
    parser.add_argument(
        "--text-embedding-cache",
        default=os.environ.get(
            "KIMODO_TEXT_EMBEDDING_CACHE", str(DEFAULT_TEXT_EMBEDDING_CACHE)
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default=os.environ.get("KIMODO_DTYPE", "fp32"),
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.environ.get("KIMODO_DIFFUSION_STEPS", "10")),
    )
    parser.add_argument(
        "--execution-frames",
        type=int,
        default=int(os.environ.get("KIMODO_EXECUTION_FRAMES", "0")),
    )
    parser.add_argument(
        "--rtc", type=int, choices=(0, 1), default=int(os.environ.get("KIMODO_RTC", "1"))
    )
    parser.add_argument(
        "--rtc-overlap-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_OVERLAP_FRAMES", "12")),
    )
    parser.add_argument(
        "--rtc-frozen-frames",
        type=int,
        default=int(os.environ.get("KIMODO_RTC_FROZEN_FRAMES", "1")),
    )
    parser.add_argument(
        "--rtc-ramp-power",
        type=float,
        default=float(os.environ.get("KIMODO_RTC_RAMP_POWER", "1.0")),
    )
    parser.add_argument("--hand-rtc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hand-stabilization", type=int, choices=(0, 1), default=1)
    parser.add_argument("--hand-close-threshold", type=float, default=.65)
    parser.add_argument("--hand-open-threshold", type=float, default=.35)
    parser.add_argument("--hand-close-frames", type=int, default=6)
    parser.add_argument("--hand-open-frames", type=int, default=21)
    parser.add_argument("--control-fps", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= int(args.port) <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    RequestHandler.runtime = KimodoRealWorldRuntime(args)
    server = ThreadingHTTPServer((args.host, int(args.port)), RequestHandler)
    print(
        f"Serving Kimodo real-world server on http://{args.host}:{args.port} "
        f"(control_fps={args.control_fps:g})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[real-world-server] stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
