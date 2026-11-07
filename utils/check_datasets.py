#!/usr/bin/env python3
"""Audit data integrity and motion quality for a multisource training config.

This script intentionally reuses the discontinuity detectors used by
``data.datasetloader``.  It scans parquet motion/state columns only; it
does not initialize the model, decode training images, or modify the dataset.
Arena additionally validates file completeness, schema, timestamps, frame indices,
root rotations, and hand commands. Arena continuity flags are audit diagnostics;
they do not change which episodes the training loader selects.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from data.datasetloader import (
    HIW_TRIGGER_CLOSE_THRESHOLD,
    HIW_SQUEEZE_OPEN_THRESHOLD,
    UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
    UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
    UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
    UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
    UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
    EpisodeRecord,
    MultiSourceG1Dataset,
    ParquetEpisodeReader,
    SOURCE_HIW500,
    SOURCE_HUMANOID_EVERYDAY,
    SOURCE_HUMANOID_ARENA,
    SOURCE_UNIFOLM,
    _as_matrix,
    _hiw_joint_discontinuity,
    _unifolm_motion_discontinuity,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "pre_training"
    / "419h_gbs1024_100w_controlnet8_detach_false_mse.yaml"
)

_WORKER_READER: ParquetEpisodeReader | None = None
_WORKER_SELECTION: dict[str, dict[str, Any]] = {}
_WORKER_ACTION_CHUNK = 50


def _initialize_worker(
    selection_by_source: Mapping[str, Mapping[str, Any]],
    action_chunk: int,
) -> None:
    """Initialize lightweight process-local parquet state."""
    global _WORKER_READER, _WORKER_SELECTION, _WORKER_ACTION_CHUNK
    _WORKER_READER = ParquetEpisodeReader()
    _WORKER_SELECTION = {
        str(source): dict(selection)
        for source, selection in selection_by_source.items()
    }
    _WORKER_ACTION_CHUNK = int(action_chunk)


def _selection(source: str) -> dict[str, Any]:
    return _WORKER_SELECTION.get(source, {})


def _target_length_after_trim(episode: EpisodeRecord, trim_frames: int) -> int:
    remaining = int(episode.source_length) - int(trim_frames)
    if remaining <= 0:
        return 0
    return int(
        round((remaining - 1) * episode.target_fps / episode.source_fps)
    ) + 1


def _motion_violations(
    metrics: Mapping[str, float | int], selection: Mapping[str, Any]
) -> list[str]:
    violations = []
    if float(metrics["root_jump"]) > float(
        selection.get(
            "internal_root_jump_threshold",
            UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
        )
    ):
        violations.append("root_translation_jump")
    if float(metrics["joint_jump"]) > float(
        selection.get(
            "joint_jump_threshold_radians",
            UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
        )
    ):
        violations.append("joint_jump")
    if float(metrics["root_rotation_jump_degrees"]) > float(
        selection.get(
            "root_rotation_jump_threshold_degrees",
            UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
        )
    ):
        violations.append("root_rotation_jump")
    return violations


def _audit_unifolm_like(
    source: str, episode: EpisodeRecord
) -> dict[str, Any]:
    assert _WORKER_READER is not None
    selection = _selection(source)
    if source == SOURCE_UNIFOLM:
        table = _WORKER_READER.read(
            episode,
            [
                "observation.state.robot_q_current",
                "action.robot_q_desired",
            ],
        )
        current = _as_matrix(
            table["observation.state.robot_q_current"],
            36,
            "UnifoLM current q",
        )
        desired = _as_matrix(
            table["action.robot_q_desired"],
            36,
            "UnifoLM desired q",
        )
    else:
        table = _WORKER_READER.read(
            episode,
            [
                "observation.arm_joints",
                "observation.leg_joints",
                "observation.odometry.position",
                "observation.odometry.quat",
                "action",
            ],
        )
        arm = _as_matrix(
            table["observation.arm_joints"], 14, "HumanoidEveryday arm"
        )
        leg = _as_matrix(
            table["observation.leg_joints"], 15, "HumanoidEveryday leg"
        )
        root_position = _as_matrix(
            table["observation.odometry.position"],
            3,
            "HumanoidEveryday odometry position",
        )
        root_quaternion = _as_matrix(
            table["observation.odometry.quat"],
            4,
            "HumanoidEveryday odometry quaternion",
        )
        action = _as_matrix(table["action"], 28, "HumanoidEveryday action")
        observed_q = np.concatenate((leg, arm), axis=1)
        target_q = np.concatenate((leg, action[:, 14:28]), axis=1)
        current = np.concatenate(
            (root_position, root_quaternion, observed_q), axis=1
        )
        desired = np.concatenate(
            (root_position, root_quaternion, target_q), axis=1
        )

    frame_offset, issue = _unifolm_motion_discontinuity(
        current,
        desired,
        first_root_jump_threshold=selection.get(
            "first_root_jump_threshold",
            UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
        ),
        root_jump_threshold=selection.get(
            "internal_root_jump_threshold",
            UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
        ),
        joint_jump_threshold=selection.get(
            "joint_jump_threshold_radians",
            UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
        ),
        root_rotation_jump_threshold_degrees=selection.get(
            "root_rotation_jump_threshold_degrees",
            UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
        ),
        max_initial_trim_frames=selection.get(
            "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
        ),
    )
    if issue is not None:
        transition = int(issue["transition_frame"])
        return {
            "source": source,
            "status": "invalid",
            "reason_type": "motion_discontinuity",
            "reason": (
                f"motion discontinuity at frames {transition}->{transition + 1}: "
                f"root={float(issue['root_jump']):.3f} m, "
                f"joint={float(issue['joint_jump']):.3f} rad, "
                "root_rotation="
                f"{float(issue['root_rotation_jump_degrees']):.1f} deg"
            ),
            "violations": _motion_violations(issue, selection),
            "metrics": dict(issue),
        }

    if _target_length_after_trim(episode, frame_offset) < _WORKER_ACTION_CHUNK:
        return {
            "source": source,
            "status": "invalid",
            "reason_type": "too_short_after_initial_trim",
            "reason": (
                f"episode is shorter than action_chunk={_WORKER_ACTION_CHUNK} "
                f"after trimming {frame_offset} source frame(s)"
            ),
            "violations": ["too_short_after_initial_trim"],
            "trimmed_source_frames": int(frame_offset),
        }
    if frame_offset:
        return {
            "source": source,
            "status": "trimmed",
            "reason_type": "initial_reset_trim",
            "trimmed_source_frames": int(frame_offset),
        }
    return {"source": source, "status": "valid"}


def _audit_hiw(episode: EpisodeRecord) -> dict[str, Any]:
    assert _WORKER_READER is not None
    selection = _selection(SOURCE_HIW500)
    table = _WORKER_READER.read(episode, ["observation.state"])
    joint_q = _as_matrix(table["observation.state"], 29, "HIW joint state")
    frame_offset, issue = _hiw_joint_discontinuity(
        joint_q,
        joint_jump_threshold=selection.get(
            "joint_jump_threshold_radians",
            UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
        ),
        max_initial_trim_frames=selection.get(
            "max_initial_trim_frames", UNIFOLM_MAX_INITIAL_TRIM_FRAMES
        ),
    )
    if issue is not None:
        transition = int(issue["transition_frame"])
        return {
            "source": SOURCE_HIW500,
            "status": "invalid",
            "reason_type": "joint_discontinuity",
            "reason": (
                f"joint discontinuity at frames {transition}->{transition + 1}: "
                f"joint={float(issue['joint_jump']):.3f} rad"
            ),
            "violations": ["joint_jump"],
            "metrics": dict(issue),
        }
    if _target_length_after_trim(episode, frame_offset) < _WORKER_ACTION_CHUNK:
        return {
            "source": SOURCE_HIW500,
            "status": "invalid",
            "reason_type": "too_short_after_initial_trim",
            "reason": (
                f"episode is shorter than action_chunk={_WORKER_ACTION_CHUNK} "
                f"after trimming {frame_offset} source frame(s)"
            ),
            "violations": ["too_short_after_initial_trim"],
            "trimmed_source_frames": int(frame_offset),
        }
    if frame_offset:
        return {
            "source": SOURCE_HIW500,
            "status": "trimmed",
            "reason_type": "initial_reset_trim",
            "trimmed_source_frames": int(frame_offset),
        }
    return {"source": SOURCE_HIW500, "status": "valid"}


def _audit_arena(episode: EpisodeRecord) -> dict[str, Any]:
    """Validate Arena's action/state contract without loading a model or video."""
    from motion.g1_reference import HumanoidArenaActionDecoder, rot6d_row_to_matrix
    import torch

    assert _WORKER_READER is not None
    columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index"]
    table = _WORKER_READER.read(episode, columns)
    state = _as_matrix(table["observation.state"], 64, "HumanoidArena observation state")
    actions = _as_matrix(table["action"], 40, "HumanoidArena action")
    n = episode.source_length
    if state.shape[0] != n or actions.shape[0] != n:
        raise ValueError("HumanoidArena state/action length does not match episode metadata")
    if n < 2 or episode.target_length < _WORKER_ACTION_CHUNK:
        raise ValueError("HumanoidArena episode is too short for the configured action chunk")
    if not np.isfinite(episode.source_fps) or episode.source_fps <= 0:
        raise ValueError("HumanoidArena FPS must be finite and positive")
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    frames = np.asarray(table["frame_index"])
    episode_ids = np.asarray(table["episode_index"])
    for name, values in [("timestamp", timestamp), ("frame_index", frames), ("episode_index", episode_ids)]:
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"HumanoidArena {name} must contain {n} finite scalar values")
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError("HumanoidArena frame_index must be contiguous and start at zero")
    expected_id = int(episode.episode_id.rsplit(":", 1)[-1])
    if not np.all(episode_ids == expected_id):
        raise ValueError("HumanoidArena parquet slice contains incorrect episode_index values")
    if not np.allclose(np.diff(timestamp), 1.0 / episode.source_fps, rtol=1e-3, atol=1e-5):
        raise ValueError("HumanoidArena timestamps are not contiguous at the declared FPS")
    if np.any((actions[:, 38:40] < 0) | (actions[:, 38:40] > 1)):
        raise ValueError("HumanoidArena hand commands must be in [0, 1]")
    rotations = []
    for name, rot6d in [("state", state[:, :6]), ("action", actions[:, 3:9])]:
        if np.any(np.linalg.norm(rot6d[:, (0, 2, 4)], axis=1) < 1e-6) or np.any(
            np.linalg.norm(np.cross(rot6d[:, (0, 2, 4)], rot6d[:, (1, 3, 5)]), axis=1) < 1e-6
        ):
            raise ValueError(f"HumanoidArena {name} contains a degenerate 6D root rotation")
        rotations.append(rot6d_row_to_matrix(torch.from_numpy(rot6d)))
    positions = HumanoidArenaActionDecoder._integrate_root(torch.from_numpy(actions), rotations[1]).numpy()
    if not np.isfinite(positions).all():
        raise ValueError("HumanoidArena decoded root positions contain NaN or Inf")
    rotation_jumps = []
    for rotation in rotations:
        r = rotation.numpy()
        cosine = np.clip((np.einsum("tij,tij->t", r[1:], r[:-1]) - 1) / 2, -1, 1)
        rotation_jumps.append(float(np.rad2deg(np.arccos(cosine)).max()))
    metrics = {
        "root_jump": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()),
        "joint_jump": float(max(np.abs(np.diff(state[:, 6:35], axis=0)).max(), np.abs(np.diff(actions[:, 9:38], axis=0)).max())),
        "root_rotation_jump_degrees": max(rotation_jumps),
    }
    violations = _motion_violations(metrics, _selection(SOURCE_HUMANOID_ARENA))
    if violations:
        return {"source": SOURCE_HUMANOID_ARENA, "status": "invalid", "reason_type": "motion_discontinuity",
                "reason": "HumanoidArena motion exceeds audit thresholds: " + ", ".join(violations),
                "violations": violations, "metrics": metrics}
    return {"source": SOURCE_HUMANOID_ARENA, "status": "valid"}


def _audit_episode(item: tuple[str, EpisodeRecord]) -> dict[str, Any]:
    """Audit one episode, keeping data errors separate from quality rejects."""
    source, episode = item
    try:
        if source == SOURCE_HUMANOID_ARENA:
            return _audit_arena(episode)
        if source in {SOURCE_UNIFOLM, SOURCE_HUMANOID_EVERYDAY}:
            return _audit_unifolm_like(source, episode)
        if source == SOURCE_HIW500:
            return _audit_hiw(episode)
        raise ValueError(
            f"Motion filter audit does not support dataset source {source!r}"
        )
    except Exception as error:
        return {
            "source": source,
            "status": "error",
            "reason_type": type(error).__name__,
            "reason": f"{type(error).__name__}: {error}",
        }


def _build_dataset(config) -> MultiSourceG1Dataset:
    dataset_roots = config.main.get("dataset_roots", None)
    if dataset_roots is not None:
        dataset_roots = OmegaConf.to_container(dataset_roots, resolve=True)
    sampling = config.main.get("sampling", None)
    if sampling is not None:
        sampling = OmegaConf.to_container(sampling, resolve=True)
    dataset = MultiSourceG1Dataset(
        dataset_root=config.main.get("data_root", None),
        dataset_roots=dataset_roots,
        action_history=config.main.action_history,
        action_chunk=config.main.action_chunk,
        sample_stride=config.main.get("sample_stride", 1),
        episode_cache_size=0,
        video_cache_size=0,
        dataset_selection=OmegaConf.to_container(
            config.main.get("dataset_selection", {}), resolve=True
        ),
        sampling=sampling,
        target_fps=config.model.fps,
        sampling_seed=int(config.main.seed),
    )
    try:
        _check_arena_inventory(dataset)
    except Exception:
        dataset.close()
        raise
    return dataset


def _check_arena_inventory(dataset) -> None:
    """Do not let the training loader's missing-file skips hide partial downloads."""
    import pyarrow.parquet as pq
    from data.common import ARENA_TASK_KEY_BY_TASK_ID, _instruction

    adapter = dataset.adapters.get(SOURCE_HUMANOID_ARENA)
    if adapter is None:
        return
    selected = adapter._parse_selection(adapter.selection)
    merged = selected["mode"] == "merged"
    root = (adapter.root / "HumanoidArena_merged_datasets_v3_1" / selected["merged"]
            if merged else adapter.root / selected["task"] / f"{selected['backend']}_refpose_v3_1")
    expected = set()
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        metadata = pq.read_table(path, columns=["episode_index", "tasks"]).to_pydict()
        for index, task in zip(metadata["episode_index"], metadata["tasks"]):
            task_name = ARENA_TASK_KEY_BY_TASK_ID[_instruction(task)]
            if merged and adapter._skip_merged_training_task(task_name):
                continue
            expected.add(f"{root.name}:{int(index)}")
    actual = {episode.episode_id for episode in adapter.episodes}
    missing = expected - actual
    if missing:
        raise ValueError(
            f"Incomplete HumanoidArena dataset: {len(missing)} selected episodes were skipped "
            "because data/video files are missing or episodes are too short. "
            f"Check the loader warnings and download paths. Examples: {sorted(missing)[:5]}"
        )


def _post_trim_windows(episode: EpisodeRecord, trim_frames: int) -> int:
    target_length = _target_length_after_trim(episode, trim_frames)
    if target_length < _WORKER_ACTION_CHUNK:
        return 0
    stride = int(episode.metadata.get("sample_stride", 1))
    return (target_length - _WORKER_ACTION_CHUNK) // stride + 1


def _new_summary() -> dict[str, Any]:
    return {
        "episodes": Counter(),
        "windows": Counter(),
        "hours": Counter(),
        "reason_types": Counter(),
        "violations": Counter(),
        "trimmed_source_frames": 0,
    }


def _update_summary(
    summary: dict[str, Any],
    episode: EpisodeRecord,
    result: Mapping[str, Any],
) -> None:
    status = str(result["status"])
    original_windows = int(episode.sample_count)
    original_hours = float(episode.source_length) / float(episode.source_fps) / 3600.0
    trim_frames = int(result.get("trimmed_source_frames", 0))

    summary["episodes"]["total"] += 1
    summary["episodes"][status] += 1
    summary["windows"]["total"] += original_windows
    summary["hours"]["total"] += original_hours

    if status in {"valid", "trimmed"}:
        kept_windows = _post_trim_windows(episode, trim_frames)
        kept_frames = max(0, int(episode.source_length) - trim_frames)
        summary["episodes"]["kept"] += 1
        summary["windows"]["kept"] += kept_windows
        summary["hours"]["kept"] += (
            float(kept_frames) / float(episode.source_fps) / 3600.0
        )
        summary["windows"]["trimmed"] += original_windows - kept_windows
        summary["hours"]["trimmed"] += (
            float(trim_frames) / float(episode.source_fps) / 3600.0
        )
    elif status == "invalid":
        summary["windows"]["invalid"] += original_windows
        summary["hours"]["invalid"] += original_hours
    else:
        summary["windows"]["error"] += original_windows
        summary["hours"]["error"] += original_hours

    summary["trimmed_source_frames"] += trim_frames
    if result.get("reason_type"):
        summary["reason_types"][str(result["reason_type"])] += 1
    for violation in result.get("violations", []):
        summary["violations"][str(violation)] += 1


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _serialize_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    episodes = dict(summary["episodes"])
    windows = dict(summary["windows"])
    hours = dict(summary["hours"])
    total_episodes = int(episodes.get("total", 0))
    total_windows = int(windows.get("total", 0))
    invalid_windows = int(windows.get("invalid", 0))
    trimmed_windows = int(windows.get("trimmed", 0))
    error_windows = int(windows.get("error", 0))
    return {
        "episodes": episodes,
        "windows": windows,
        "hours": {key: round(float(value), 6) for key, value in hours.items()},
        "motion_filtered_episode_percent": round(
            _percent(int(episodes.get("invalid", 0)), total_episodes), 4
        ),
        "motion_filtered_window_percent": round(
            _percent(invalid_windows + trimmed_windows, total_windows), 4
        ),
        "read_error_episode_percent": round(
            _percent(int(episodes.get("error", 0)), total_episodes), 4
        ),
        "unusable_window_percent_including_errors": round(
            _percent(invalid_windows + trimmed_windows + error_windows, total_windows),
            4,
        ),
        "trimmed_source_frames": int(summary["trimmed_source_frames"]),
        "reason_types": dict(summary["reason_types"]),
        "violations": dict(summary["violations"]),
    }


def _result_iterator(
    items: list[tuple[str, EpisodeRecord]],
    selection_by_source: Mapping[str, Mapping[str, Any]],
    action_chunk: int,
    workers: int,
) -> Iterable[dict[str, Any]]:
    if workers <= 1:
        _initialize_worker(selection_by_source, action_chunk)
        return map(_audit_episode, items)
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(selection_by_source, action_chunk),
    )
    # map() preserves episode order, which lets the parent aggregate metadata
    # without copying it into every result dictionary.
    results = executor.map(_audit_episode, items, chunksize=8)

    def consume() -> Iterable[dict[str, Any]]:
        try:
            yield from results
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    return consume()


def _print_summary(source_summaries: Mapping[str, Mapping[str, Any]]) -> None:
    print()
    print(
        "Source | Episodes total/kept/trimmed/invalid/error | "
        "motion-filtered windows | kept hours"
    )
    print("-" * 112)
    for source, summary in source_summaries.items():
        episodes = summary["episodes"]
        windows = summary["windows"]
        hours = summary["hours"]
        filtered_windows = int(windows.get("invalid", 0)) + int(
            windows.get("trimmed", 0)
        )
        print(
            f"{source} | "
            f"{episodes.get('total', 0)}/{episodes.get('kept', 0)}/"
            f"{episodes.get('trimmed', 0)}/{episodes.get('invalid', 0)}/"
            f"{episodes.get('error', 0)} | "
            f"{filtered_windows:,}/{windows.get('total', 0):,} "
            f"({summary['motion_filtered_window_percent']:.3f}%) | "
            f"{hours.get('kept', 0.0):.3f}/{hours.get('total', 0.0):.3f} h"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit episode filtering using the exact motion discontinuity "
            "thresholds from a multisource training YAML."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Audit worker processes. Defaults to main.cpu_workers_num, capped "
            "by the visible CPU count. Use 1 for sequential debugging."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audit_invalid_episodes_report.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N selected episodes for a quick smoke test.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many episodes; 0 disables progress.",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Return a nonzero exit status if parquet/schema read errors occur.",
    )
    parser.add_argument("--fail-on-invalid", action="store_true",
                        help="Return a nonzero exit status if motion-quality checks reject episodes.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")

    config = OmegaConf.load(config_path)
    dataset = _build_dataset(config)
    try:
        items = [
            (source, episode)
            for source, records in dataset._episodes_by_source.items()
            for _, episode in records
        ]
        selection_by_source = {
            source: dict(adapter.selection)
            for source, adapter in dataset.adapters.items()
        }
    finally:
        dataset.close()

    if args.limit is not None:
        items = items[: args.limit]
    cpu_count = os.cpu_count() or 1
    configured_workers = int(config.main.get("cpu_workers_num", 1))
    workers = (
        min(configured_workers, cpu_count)
        if args.workers is None
        else int(args.workers)
    )
    if workers <= 0:
        raise ValueError("--workers must be positive")

    action_chunk = int(config.main.action_chunk)
    # Used by parent-side post-trim window accounting.
    global _WORKER_ACTION_CHUNK
    _WORKER_ACTION_CHUNK = action_chunk

    print(f"Config: {config_path}")
    print(f"Selected/discovered episodes: {len(items):,}")
    print(f"Audit workers: {workers}")
    print("Video decoding is not audited; no dataset files will be modified.")
    if SOURCE_HUMANOID_ARENA in selection_by_source:
        print("Arena motion-quality flags are diagnostics, not training-time filtering.")

    started = time.time()
    overall = _new_summary()
    by_source: dict[str, dict[str, Any]] = defaultdict(_new_summary)
    by_source_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(_new_summary)
    )
    issue_details = []
    results = _result_iterator(
        items, selection_by_source, action_chunk, workers
    )
    for index, ((source, episode), result) in enumerate(
        zip(items, results), start=1
    ):
        _update_summary(overall, episode, result)
        _update_summary(by_source[source], episode, result)
        _update_summary(by_source_task[source][episode.task_name], episode, result)
        if result["status"] != "valid":
            detail = {
                "source": source,
                "task": episode.task_name,
                "instruction": episode.instruction,
                "episode_id": episode.episode_id,
                "data_path": str(episode.data_path),
                **dict(result),
            }
            issue_details.append(detail)
        if args.progress_every and (
            index % args.progress_every == 0 or index == len(items)
        ):
            elapsed = time.time() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            print(
                f"Audited {index:,}/{len(items):,} episodes "
                f"({rate:.1f} episodes/s)",
                flush=True,
            )

    serialized_sources = {
        source: _serialize_summary(summary)
        for source, summary in sorted(by_source.items())
    }
    report = {
        "config": str(config_path),
        "scope": (
            "Task/camera-selected episodes discovered by MultiSourceG1Dataset; "
            "source-specific schema and motion continuity checks; video frames not decoded."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "workers": workers,
        "thresholds_by_source": selection_by_source,
        "defaults": {
            "first_root_jump_threshold_meters": UNIFOLM_FIRST_ROOT_JUMP_THRESHOLD_METERS,
            "internal_root_jump_threshold_meters": UNIFOLM_INTERNAL_ROOT_JUMP_THRESHOLD_METERS,
            "joint_jump_threshold_radians": UNIFOLM_JOINT_JUMP_THRESHOLD_RADIANS,
            "root_rotation_jump_threshold_degrees": UNIFOLM_ROOT_ROTATION_JUMP_THRESHOLD_DEGREES,
            "max_initial_trim_frames": UNIFOLM_MAX_INITIAL_TRIM_FRAMES,
            "hiw_hand_trigger_threshold": HIW_TRIGGER_CLOSE_THRESHOLD,
            "hiw_hand_squeeze_threshold": HIW_SQUEEZE_OPEN_THRESHOLD,
        },
        "overall": _serialize_summary(overall),
        "by_source": serialized_sources,
        "by_source_task": {
            source: {
                task: _serialize_summary(summary)
                for task, summary in sorted(tasks.items())
            }
            for source, tasks in sorted(by_source_task.items())
        },
        "issues": issue_details,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    _print_summary(serialized_sources)
    print()
    overall_report = report["overall"]
    print(
        "Overall motion-filtered windows: "
        f"{overall_report['motion_filtered_window_percent']:.4f}%"
    )
    print(
        "Overall read-error episodes: "
        f"{overall_report['episodes'].get('error', 0):,}"
    )
    print(f"Full JSON report: {output_path}")

    if args.fail_on_invalid and overall_report["episodes"].get("invalid", 0):
        return 2
    if args.fail_on_errors and overall_report["episodes"].get("error", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# cd /path/to/controlnet_v1.2
# export UNIFOLM_ROOT=/path/to/UnifoLM_WBT_Dataset \
# export HUMANOID_EVERYDAY_ROOT=/path/to/HumanoidEveryday \
# export HIW500_ROOT=/path/to/HIW500 \

# python utils/check_datasets.py \
#     --config scripts/experiments/pre_training/419h_gbs1024_100w_controlnet8_detach_false_mse.yaml \
#     --workers 24 \
#     --output audit_419h_filter_report.json
