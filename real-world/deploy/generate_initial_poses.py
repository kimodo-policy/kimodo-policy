"""Select an actual demonstration first frame as each task's deployment pose.

Run on the H20 training host, where the merged real-world datasets live. The
commanded pose comes from action.*, and convergence targets come from
observation.*; neither is synthesized by averaging episode poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TASKS = ('boxing', 'pushing', 'transportation', 'sitting', 'football', 'pnp', 'long')
COLUMNS = (
    'action.joint_q', 'observation.joint_q', 'action.root_q', 'action.root_z',
    'action.hand_binary', 'action.root_target_valid',
    'action.root_target_discontinuous', 'frame_index',
)
POSE_SETTINGS = {
    'transition_min_s': 5.0,
    'joint_speed_limit_rad_s': 0.3,
    'root_speed_limit_m_s': 0.02,
    'settle_s': 1.0,
    'history_s': 3.5,
    'arm_rms_tolerance_rad': 0.18,
    'arm_max_tolerance_rad': 0.35,
    'body_rms_tolerance_rad': 0.25,
    'velocity_rms_tolerance_rad_s': 0.2,
    'orientation_tolerance_rad': 0.26,
}


def first_frame(path: Path) -> dict:
    table = pq.read_table(path, columns=list(COLUMNS))
    if table.num_rows == 0:
        raise ValueError(f'Empty episode: {path}')
    return {key: table[key][0].as_py() for key in COLUMNS}


def select_pose(data_root: Path, task: str) -> tuple[dict, dict]:
    dataset = data_root / f'teleop-{task}'
    info = json.loads((dataset / 'meta/info.json').read_text())
    names = info['features']['action.joint_q']['names']
    if len(names) != 29 or names[18] != 'left_elbow_joint' or names[25] != 'right_elbow_joint':
        raise ValueError(f'Unexpected joint order for {task}')
    candidates = []
    excluded = []
    for path in sorted((dataset / 'data').glob('chunk-*/episode_*.parquet')):
        row = first_frame(path)
        q = np.asarray(row['action.joint_q'], dtype=float)
        measured = np.asarray(row['observation.joint_q'], dtype=float)
        quat = np.asarray(row['action.root_q'], dtype=float)
        hands = np.asarray(row['action.hand_binary'], dtype=float)
        height = float(row['action.root_z'])
        valid = (
            q.shape == (29,) and measured.shape == (29,) and quat.shape == (4,)
            and hands.shape == (2,) and np.isfinite(q).all()
            and np.isfinite(measured).all() and np.isfinite(quat).all()
            and np.isfinite(hands).all() and np.isfinite(height)
            and 0.65 <= height <= 1.15 and 0.9 <= np.linalg.norm(quat) <= 1.1
            and np.max(np.abs(q)) <= 3.2
            and np.all((hands >= 0) & (hands <= 1))
            and float(row['action.root_target_valid']) >= 0.5
            and float(row['action.root_target_discontinuous']) < 0.5
        )
        if valid:
            candidates.append((path, row, q))
        else:
            excluded.append(path.name)
    if not candidates:
        raise ValueError(f'No valid first frames for {task}; excluded={excluded}')
    arms = np.stack([q[15:29] for _, _, q in candidates])
    median = np.median(arms, axis=0)
    distances = np.sqrt(np.mean((arms - median) ** 2, axis=1))
    selected_index = int(np.argmin(distances))
    path, row, q = candidates[selected_index]
    pose = {
        'version': 1,
        'task': f'teleop-{task}',
        'source_episode': path.name,
        'source_frame': int(row['frame_index']),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'selection': (
            f'Actual first reference frame closest to the component-wise median '
            f'first-frame arm reference across {len(candidates)} valid episodes; '
            'no averaging of commanded poses.'
        ),
        'joint_order': names,
        'reference_joint_q_mujoco': q.tolist(),
        'expected_observed_joint_q_mujoco': row['observation.joint_q'],
        'root_quat_wxyz': row['action.root_q'],
        'root_height_m': float(row['action.root_z']),
        'hand_closure': row['action.hand_binary'],
        **POSE_SETTINGS,
    }
    report = {
        'task': task,
        'episodes_total': int(info['total_episodes']),
        'first_frames_valid': len(candidates),
        'first_frames_excluded': excluded,
        'selected_episode': path.name,
        'selected_frame': pose['source_frame'],
        'arm_distance_to_median_rms_rad': round(float(distances[selected_index]), 5),
        'median_arm_distance_rms_rad': round(float(np.median(distances)), 5),
        'root_height_m': round(float(pose['root_height_m']), 4),
        'left_elbow_rad': round(float(q[18]), 4),
        'right_elbow_rad': round(float(q[25]), 4),
        'hand_closure': pose['hand_closure'],
    }
    return pose, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for task in args.tasks:
        pose, report = select_pose(args.data_root, task)
        (args.output_dir / f'{task}_initial_pose.json').write_text(
            json.dumps(pose, indent=2) + '\n'
        )
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    (args.output_dir / 'initial_pose_selection_report.json').write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + '\n'
    )


if __name__ == '__main__':
    main()
