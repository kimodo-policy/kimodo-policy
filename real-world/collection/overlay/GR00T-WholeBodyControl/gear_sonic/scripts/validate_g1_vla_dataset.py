#!/usr/bin/env python3
"""Strict read-only preflight validation for real G1 VLA datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.g1_reference import G1_REFERENCE_JOINT_NAMES

REQUIRED_COLUMNS = {
    "observation.joint_q": 29,
    "observation.joint_dq": 29,
    "observation.root_q_relative": 4,
    "observation.root_orientation": 4,
    "action.joint_q": 29,
    "action.root_p": 3,
    "action.root_q": 4,
    "action.root_z": 1,
    "action.hand_binary": 2,
    "control.joint_q_target": 29,
    "audit.gmr_joint_q_raw": 29,
    "audit.gmr_root_p_raw": 3,
    "audit.gmr_root_q_raw": 4,
    "audit.g1_joint_q_measured_raw": 29,
    "audit.g1_joint_dq_measured_raw": 29,
    "audit.g1_root_q_measured_raw": 4,
    "sync.state_sequence": 1,
    "sync.state_monotonic_timestamp": 1,
    "sync.reference_sequence": 1,
    "sync.reference_monotonic_timestamp": 1,
    "sync.control_sequence": 1,
    "sync.control_monotonic_timestamp": 1,
    "sync.video_sequence": 1,
    "sync.video_capture_sequence": 1,
    "sync.video_capture_source_monotonic_timestamp": 1,
    "sync.camera_resampled": 1,
    "sync.camera_source_gap_frames": 1,
    "sync.camera_source_skipped_captures": 1,
    "sync.video_monotonic_timestamp": 1,
    "sync.smpl_sequence": 1,
    "sync.smpl_monotonic_timestamp": 1,
    "sync.state_video_receive_skew_ms": 1,
    "sync.smpl_state_receive_skew_ms": 1,
    "sync.reference_video_receive_skew_ms": 1,
    "sync.control_video_receive_skew_ms": 1,
}
WRIST_ONLY_INDICES = {19, 20, 21, 26, 27, 28}
MAX_VIDEO_SEQUENCE_STEP = 3
MAX_VIDEO_DROP_RATIO = 0.05


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _matrix(table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].to_pylist())
    if width == 1 and values.shape == (table.num_rows,):
        values = values[:, None]
    if values.shape != (table.num_rows, width):
        raise ValueError(f"{key}: expected {(table.num_rows, width)}, got {values.shape}")
    return values


def _count_video_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _video_sequence_drop_stats(sequence: np.ndarray) -> tuple[int, float, int]:
    if sequence.size < 2:
        return 0, 0.0, 0
    delta = np.diff(sequence)
    dropped_frames = int(np.sum(np.maximum(delta - 1, 0)))
    source_frame_count = int(sequence[-1] - sequence[0] + 1)
    drop_ratio = dropped_frames / source_frame_count if source_frame_count > 0 else 0.0
    return dropped_frames, drop_ratio, int(np.max(delta))


def _video_fps(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return float(stream.average_rate)


def _video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def _heading_quaternion(q: np.ndarray) -> np.ndarray:
    yaw = R.from_quat(q, scalar_first=True).as_euler("xyz")[2]
    return R.from_euler("z", yaw).as_quat(scalar_first=True)


def _heading_relative_root(raw_p: np.ndarray, raw_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heading0 = R.from_quat(_heading_quaternion(raw_q[0]), scalar_first=True)
    p_rel = heading0.inv().apply(raw_p - raw_p[0])
    q_rel = (heading0.inv() * R.from_quat(raw_q, scalar_first=True)).as_quat(
        scalar_first=True
    )
    return p_rel, q_rel


def _longest_equal_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    longest = current = 1
    for previous, value in zip(values[:-1], values[1:]):
        if value == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _check_quaternion(name: str, values: np.ndarray, errors: list[str]) -> None:
    norms = np.linalg.norm(values, axis=1)
    bad = np.flatnonzero(~np.isclose(norms, 1.0, atol=1e-4))
    if bad.size:
        errors.append(f"{name}: {bad.size} non-normalized quaternion rows")


def validate_episode(
    dataset_root: Path,
    parquet_path: Path,
    task_indices: set[int],
    video_keys: list[str],
    max_receive_skew_ms: float,
) -> tuple[list[str], list[str], set[Path]]:
    table = pq.read_table(parquet_path)
    errors: list[str] = []
    warnings: list[str] = []
    expected_videos: set[Path] = set()
    episode_name = parquet_path.stem

    for key, width in REQUIRED_COLUMNS.items():
        if key not in table.column_names:
            errors.append(f"missing column {key}")
            continue
        try:
            values = _matrix(table, key, width)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not np.all(np.isfinite(values)):
            errors.append(f"{key}: contains NaN/Inf")

    if "action.joint_q" in table.column_names:
        try:
            joint_q = _matrix(table, "action.joint_q", 29)
            active = set(np.flatnonzero(np.max(np.abs(joint_q), axis=0) > 1e-5).tolist())
            non_wrist_energy = np.max(
                np.abs(joint_q[:, sorted(set(range(29)) - WRIST_ONLY_INDICES)])
            )
            if active.issubset(WRIST_ONLY_INDICES) or non_wrist_energy <= 1e-5:
                errors.append(
                    "action.joint_q is wrist-only/zero-filled outside indices "
                    f"{sorted(WRIST_ONLY_INDICES)}"
                )
            elif len(active) < 12:
                warnings.append(f"action.joint_q has only {len(active)}/29 active dimensions")
        except ValueError:
            joint_q = None
    else:
        joint_q = None

    if joint_q is not None and "audit.gmr_joint_q_raw" in table.column_names:
        raw_joint_q = _matrix(table, "audit.gmr_joint_q_raw", 29)
        mismatch = np.flatnonzero(
            np.max(np.abs(joint_q - raw_joint_q), axis=1) > 1e-9
        )
        if mismatch.size:
            errors.append(
                f"action.joint_q differs from matching raw GMR qpos[7:] in {mismatch.size} rows"
            )

    for saved_key, raw_key, width in (
        ("observation.joint_q", "audit.g1_joint_q_measured_raw", 29),
        ("observation.joint_dq", "audit.g1_joint_dq_measured_raw", 29),
        ("observation.root_orientation", "audit.g1_root_q_measured_raw", 4),
    ):
        if saved_key in table.column_names and raw_key in table.column_names:
            saved = _matrix(table, saved_key, width)
            raw = _matrix(table, raw_key, width)
            if not np.allclose(saved, raw, rtol=0.0, atol=1e-9):
                errors.append(f"{saved_key} differs from matching measured G1 raw data")

    if "action.root_p" not in table.column_names:
        return errors, warnings, expected_videos
    root_p = _matrix(table, "action.root_p", 3)
    if not np.allclose(root_p[0], 0.0, atol=1e-8):
        errors.append(f"action.root_p first row is {root_p[0].tolist()}, expected [0, 0, 0]")
    _check_quaternion(
        "observation.root_q_relative",
        _matrix(table, "observation.root_q_relative", 4),
        errors,
    )
    _check_quaternion("action.root_q", _matrix(table, "action.root_q", 4), errors)
    if {"audit.gmr_root_p_raw", "audit.gmr_root_q_raw", "action.root_q"}.issubset(
        table.column_names
    ):
        raw_root_p = _matrix(table, "audit.gmr_root_p_raw", 3)
        raw_root_q = _matrix(table, "audit.gmr_root_q_raw", 4)
        expected_root_p, expected_root_q = _heading_relative_root(raw_root_p, raw_root_q)
        if not np.allclose(root_p, expected_root_p, rtol=0.0, atol=1e-8):
            errors.append("action.root_p is not inverse(heading(q0)) * (p-p0)")
        saved_root_q = _matrix(table, "action.root_q", 4)
        quat_dot = np.abs(np.sum(saved_root_q * expected_root_q, axis=1))
        if not np.allclose(quat_dot, 1.0, rtol=0.0, atol=1e-8):
            errors.append("action.root_q is not inverse(heading(q0)) * q")
        first_yaw = R.from_quat(saved_root_q[0], scalar_first=True).as_euler("xyz")[2]
        if not np.isclose(first_yaw, 0.0, atol=1e-8):
            errors.append(f"action.root_q first-row yaw is {first_yaw}, expected 0")

    hand = _matrix(table, "action.hand_binary", 2)
    if not np.all(np.isin(hand, [0.0, 1.0])):
        errors.append("action.hand_binary contains non-binary values")

    if "timestamp" in table.column_names:
        timestamps = _matrix(table, "timestamp", 1)[:, 0]
        if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            errors.append("timestamp: must be finite and strictly increasing")
        if not np.isclose(timestamps[0], 0.0, atol=1e-4):
            errors.append(f"timestamp first value={timestamps[0]:.6f}, expected 0")
    else:
        errors.append("missing authoritative timestamp column")

    for key in (
        "sync.state_sequence",
        "sync.video_sequence",
        "sync.video_capture_sequence",
        "sync.reference_sequence",
    ):
        if key not in table.column_names:
            continue
        sequence = _matrix(table, key, 1)[:, 0].astype(np.int64)
        delta = np.diff(sequence)
        if np.any(delta < 0):
            errors.append(f"{key}: sequence rollback detected")
        if key == "sync.video_sequence" and np.any(delta == 0):
            errors.append(f"{key}: repeated frame detected")
        if key == "sync.video_capture_sequence" and np.any(delta < 0):
            errors.append(f"{key}: capture sequence rollback detected")
        if key == "sync.video_capture_sequence":
            stats_sequence = sequence
            if "sync.camera_resampled" in table.column_names:
                fresh = _matrix(table, "sync.camera_resampled", 1)[:, 0] == 0.0
                stats_sequence = sequence[fresh]
            dropped_frames, drop_ratio, max_step = _video_sequence_drop_stats(stats_sequence)
            if dropped_frames:
                warnings.append(
                    f"{key}: dropped {dropped_frames} source frames "
                    f"({100.0 * drop_ratio:.2f}%, max step={max_step})"
                )
            if max_step > MAX_VIDEO_SEQUENCE_STEP:
                errors.append(
                    f"{key}: sequence step {max_step} exceeds "
                    f"{MAX_VIDEO_SEQUENCE_STEP}"
                )
            if drop_ratio > MAX_VIDEO_DROP_RATIO:
                errors.append(
                    f"{key}: source frame drop ratio {100.0 * drop_ratio:.2f}% "
                    f"exceeds {100.0 * MAX_VIDEO_DROP_RATIO:.1f}%"
                )
        if key == "sync.reference_sequence" and _longest_equal_run(sequence) > 3:
            errors.append(f"{key}: repeated reference run exceeds 3 rows")

    for key in (
        "sync.state_monotonic_timestamp",
        "sync.video_monotonic_timestamp",
        "sync.reference_monotonic_timestamp",
        "sync.control_monotonic_timestamp",
    ):
        if key not in table.column_names:
            continue
        timestamp = _matrix(table, key, 1)[:, 0]
        if np.any(timestamp <= 0.0):
            errors.append(f"{key}: missing/non-positive timestamps")
        if np.any(np.diff(timestamp) < 0.0):
            errors.append(f"{key}: timestamp rollback detected")

    if "sync.video_source_monotonic_timestamp" in table.column_names:
        source_ts = _matrix(table, "sync.video_source_monotonic_timestamp", 1)[:, 0]
        if np.any(source_ts <= 0.0) or np.any(np.diff(source_ts) < 0.0):
            errors.append("sync.video_source_monotonic_timestamp: rollback detected")
        elif "sync.camera_resampled" in table.column_names:
            fresh = _matrix(table, "sync.camera_resampled", 1)[:, 0] == 0.0
            fresh_source_ts = source_ts[fresh]
            if fresh_source_ts.size > 1 and np.any(np.diff(fresh_source_ts) <= 0.0):
                errors.append(
                    "sync.video_source_monotonic_timestamp: fresh captures not increasing"
                )
    if "action.root_z" in table.column_names:
        root_z = _matrix(table, "action.root_z", 1)[:, 0]
        if np.any((root_z < 0.5) | (root_z > 1.2)):
            errors.append(
                "action.root_z: outside validated G1 pelvis-height range [0.5, 1.2] m"
            )
    for key, label in (
        ("sync.camera_resampled", "camera resampled output rows"),
        ("sync.camera_source_gap_frames", "camera missing grid frames"),
        ("sync.camera_source_skipped_captures", "camera same-grid captures skipped"),
    ):
        if key not in table.column_names:
            continue
        values = _matrix(table, key, 1)[:, 0]
        if np.any(values < 0):
            errors.append(f"{key}: contains negative values")
        total = int(np.sum(values))
        if total:
            warnings.append(f"{label}: {total}")

    for key, label in (
        ("sync.state_video_receive_skew_ms", "state/video"),
        ("sync.reference_video_receive_skew_ms", "reference/video"),
        ("sync.control_video_receive_skew_ms", "control/video"),
    ):
        skew = (
            np.abs(_matrix(table, key, 1)[:, 0])
            if key in table.column_names
            else np.array([])
        )
        if skew.size and "sync.camera_resampled" in table.column_names:
            fresh = _matrix(table, "sync.camera_resampled", 1)[:, 0] == 0.0
            skew = skew[fresh]
        if skew.size:
            warnings.append(
                f"{label} receive skew ms: mean={float(np.mean(skew)):.2f}, "
                f"p95={float(np.percentile(skew, 95)):.2f}, max={float(np.max(skew)):.2f}"
            )
        if skew.size and np.any(skew > max_receive_skew_ms):
            errors.append(
                f"{label} receive skew exceeds {max_receive_skew_ms:.1f} ms "
                f"(max={float(np.max(skew)):.1f} ms)"
            )

    joint_step = (
        np.max(np.abs(np.diff(joint_q, axis=0)), axis=1)
        if joint_q is not None and table.num_rows > 1
        else np.array([])
    )
    if joint_step.size and np.max(joint_step) > 1.0:
        errors.append(f"action.joint_q jump exceeds 1.0 rad (max={float(np.max(joint_step)):.3f})")
    root_step = np.linalg.norm(np.diff(root_p, axis=0), axis=1) if table.num_rows > 1 else np.array([])
    if root_step.size and np.max(root_step) > 0.2:
        errors.append(f"action.root_p jump exceeds 0.2 m (max={float(np.max(root_step)):.3f})")

    if "task_index" not in table.column_names:
        errors.append("missing task_index")
    else:
        episode_task_indices = set(np.asarray(table["task_index"].to_pylist(), dtype=np.int64).tolist())
        if len(episode_task_indices) != 1:
            errors.append(f"task_index changes within episode: {sorted(episode_task_indices)}")
        if not episode_task_indices.issubset(task_indices):
            errors.append(f"task_index not present in meta/tasks.jsonl: {sorted(episode_task_indices)}")

    video_root = dataset_root / "videos"
    for video_key in video_keys:
        candidates = list(video_root.glob(f"chunk-*/{video_key}/{episode_name}.mp4"))
        if len(candidates) != 1:
            errors.append(f"{video_key}: expected one video for {episode_name}, found {len(candidates)}")
            continue
        video_path = candidates[0]
        expected_videos.add(video_path.resolve())
        try:
            frame_count = _count_video_frames(video_path)
        except Exception as exc:
            errors.append(f"{video_key}: unreadable video: {exc}")
            continue
        if frame_count != table.num_rows:
            errors.append(
                f"{video_key}: video frames={frame_count}, motion rows={table.num_rows}"
            )
        fps = _video_fps(video_path)
        if not np.isclose(fps, 30.0, atol=1e-6):
            errors.append(f"{video_key}: encoded fps={fps}, expected 30")
        if "timestamp" in table.column_names:
            timestamps = _matrix(table, "timestamp", 1)[:, 0]
            motion_duration = float(timestamps[-1] - timestamps[0])
            video_duration = _video_duration(video_path)
            frame_period = 1.0 / fps
            expected_video_duration = motion_duration + frame_period
            if abs(expected_video_duration - video_duration) > 1e-3:
                errors.append(
                    f"{video_key}: duration mismatch expected={expected_video_duration:.4f}s "
                    f"video={video_duration:.4f}s"
                )
            has_camera_holds = bool(
                "sync.camera_resampled" in table.column_names
                and np.any(_matrix(table, "sync.camera_resampled", 1)[:, 0] > 0.0)
            )
            if (
                "sync.video_capture_source_monotonic_timestamp" in table.column_names
                and not has_camera_holds
            ):
                source_ts = _matrix(
                    table, "sync.video_capture_source_monotonic_timestamp", 1
                )[:, 0]
                source_duration = float(source_ts[-1] - source_ts[0])
                if abs(source_duration - video_duration) > frame_period:
                    errors.append(
                        f"{video_key}: authoritative source duration mismatch "
                        f"source={source_duration:.4f}s video={video_duration:.4f}s"
                    )

    return errors, warnings, expected_videos


def validate_dataset(dataset_root: Path, max_receive_skew_ms: float = 80.0) -> bool:
    info_path = dataset_root / "meta/info.json"
    tasks_path = dataset_root / "meta/tasks.jsonl"
    if not info_path.is_file() or not tasks_path.is_file():
        print(f"FAIL {dataset_root}: missing meta/info.json or meta/tasks.jsonl")
        return False

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("fps", -1)) != 30:
        print(f"FAIL {dataset_root}: metadata fps={info.get('fps')}, expected 30")
        return False
    action_names = info.get("features", {}).get("action.joint_q", {}).get("names")
    raw_names = info.get("features", {}).get("audit.gmr_joint_q_raw", {}).get("names")
    expected_names = list(G1_REFERENCE_JOINT_NAMES)
    if action_names != expected_names or raw_names != expected_names:
        print("FAIL dataset metadata: action/raw GMR q29 names do not match canonical order")
        return False
    tasks = _load_jsonl(tasks_path)
    task_indices = {int(item["task_index"]) for item in tasks}
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    parquet_paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_paths:
        print(f"FAIL {dataset_root}: no episode parquet files")
        return False

    ok = True
    expected_videos: set[Path] = set()
    for parquet_path in parquet_paths:
        errors, warnings, episode_videos = validate_episode(
            dataset_root, parquet_path, task_indices, video_keys, max_receive_skew_ms
        )
        expected_videos.update(episode_videos)
        label = parquet_path.stem
        if errors:
            ok = False
            print(f"FAIL {label}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {label}: {pq.read_metadata(parquet_path).num_rows} rows")
        for warning in warnings:
            print(f"  ! {warning}")

    actual_videos = {path.resolve() for path in dataset_root.glob("videos/chunk-*/*/episode_*.mp4")}
    orphan_videos = sorted(actual_videos - expected_videos)
    if orphan_videos:
        ok = False
        print("FAIL orphan video files")
        for path in orphan_videos:
            print(f"  - {path.relative_to(dataset_root.resolve())}")

    print(f"{'PASS' if ok else 'FAIL'} dataset {dataset_root}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-receive-skew-ms", type=float, default=80.0)
    args = parser.parse_args()
    return 0 if validate_dataset(args.dataset_root, args.max_receive_skew_ms) else 1


if __name__ == "__main__":
    sys.exit(main())
