#!/usr/bin/env python3
"""Read-only recursive audit of finalized G1 LeRobot v2 datasets."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import validate_g1_vla_dataset as validator


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root, skew):
    result = dict(path=str(root), errors=[], episodes=[])
    errors = result['errors']
    info = json.loads((root / 'meta/info.json').read_text())
    tasks = jsonl(root / 'meta/tasks.jsonl')
    records = jsonl(root / 'meta/episodes.jsonl')
    task_ids = {int(t['task_index']) for t in tasks}
    videos = [k for k, v in info['features'].items() if v.get('dtype') == 'video']
    if not videos:
        errors.append('metadata contains no video features')
    if info.get('fps') != 30:
        errors.append('metadata fps must be 30')
    for key in ('action.joint_q', 'audit.gmr_joint_q_raw'):
        if info['features'].get(key, {}).get('names') != list(validator.G1_REFERENCE_JOINT_NAMES):
            errors.append(f'{key}: incorrect joint order in metadata')
    paths = sorted(root.glob('data/chunk-*/episode_*.parquet'))
    ids = [int(p.stem.split('_')[-1]) for p in paths]
    record_ids = [int(r['episode_index']) for r in records]
    lengths = {int(r['episode_index']): int(r['length']) for r in records}
    if not paths:
        errors.append('no episode parquet files')
    if sorted(ids) != list(range(len(ids))):
        errors.append('episode indices are missing, duplicated, or do not start at zero')
    if sorted(record_ids) != sorted(ids):
        errors.append('meta/episodes.jsonl indices do not match parquet files')
    if info.get('total_episodes') != len(paths):
        errors.append('total_episodes mismatch')
    if info.get('total_tasks') != len(tasks) or len(task_ids) != len(tasks):
        errors.append('task metadata count mismatch or duplicate task index')
    expected_videos = set()
    total_rows = 0
    offset = 0
    for path in paths:
        ep = dict(episode=path.stem, rows=0, errors=[], warnings=[], metrics={})
        result['episodes'].append(ep)
        try:
            table = pq.read_table(path)
            n = table.num_rows
            ep['rows'] = n
            total_rows += n
            eid = int(path.stem.split('_')[-1])
            if n == 0:
                raise ValueError('empty episode')
            if lengths.get(eid) != n:
                ep['errors'].append('episode metadata length differs from parquet rows')
            for key, expected in [('frame_index', np.arange(n)), ('episode_index', np.full(n, eid)), ('index', np.arange(offset, offset+n))]:
                if key not in table.column_names or not np.array_equal(np.asarray(table[key].to_pylist()), expected):
                    ep['errors'].append(f'{key}: missing or non-contiguous/incorrect')
            # Check every numeric field, including hand_q and observation.state.
            for key in table.column_names:
                values = np.asarray(table[key].to_pylist())
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    ep['errors'].append(f'{key}: NaN/Inf')
                spec = info['features'].get(key, {})
                shape = spec.get('shape')
                if spec.get('dtype') != 'video' and shape:
                    actual = list(values.shape[1:]) or [1]
                    if actual != shape:
                        ep['errors'].append(f'{key}: shape {actual} differs from metadata {shape}')
            for key, spec in info['features'].items():
                if spec.get('dtype') != 'video' and key not in table.column_names:
                    ep['errors'].append(f'missing metadata-declared field: {key}')
            ts = np.asarray(table['timestamp'].to_pylist())
            if not np.allclose(ts, np.arange(n)/30, rtol=0, atol=1e-4):
                ep['errors'].append('timestamp is not a continuous 30 Hz grid')
            ep['metrics']['duration_seconds'] = n/30
            for key in ('sync.camera_resampled',):
                if key in table.column_names:
                    vals = np.asarray(table[key].to_pylist()).reshape(-1)
                    ep['metrics']['camera_hold_ratio'] = float(np.mean(vals > 0))
            for key in table.column_names:
                if key.startswith('sync.') and (key.endswith('_sequence') or key.endswith('_monotonic_timestamp')):
                    vals = np.asarray(table[key].to_pylist()).reshape(-1)
                    if np.any(np.diff(vals) < 0):
                        ep['errors'].append(f'{key}: rollback')
            if 'action.hand_binary' in table.column_names:
                hand = np.asarray(table['action.hand_binary'].to_pylist())
                ep['metrics']['hand_switches_left_right'] = np.sum(np.diff(hand, axis=0) != 0, axis=0).tolist()
            try:
                err, warn, seen = validator.validate_episode(root, path, task_ids, videos, skew)
                ep['errors'].extend(err)
                ep['warnings'].extend(warn)
                expected_videos.update(seen)
            except Exception as exc:
                ep['errors'].append(f'detailed validation failed: {type(exc).__name__}: {exc}')
        except Exception as exc:
            ep['errors'].append(f'unreadable/incomplete episode: {type(exc).__name__}: {exc}')
        offset += ep['rows']
        ep['status'] = 'FAIL' if ep['errors'] else 'PASS'
        print(f"  {ep['status']} {ep['episode']}: {ep['rows']} rows", flush=True)
        for e in ep['errors']:
            print('    ERROR: ' + e, flush=True)
        for w in ep['warnings']:
            print('    INFO: ' + w, flush=True)
    actual_videos = {p.resolve() for p in root.glob('videos/**/*.mp4')}
    if actual_videos - expected_videos:
        errors.append('orphan or unchecked videos: ' + ', '.join(str(p.relative_to(root)) for p in sorted(actual_videos - expected_videos)))
    if info.get('total_frames') != total_rows:
        errors.append(f'total_frames mismatch: metadata={info.get("total_frames")}, actual={total_rows}')
    if info.get('total_videos') != len(actual_videos):
        errors.append('total_videos mismatch')
    result['status'] = 'FAIL' if errors or any(e['errors'] for e in result['episodes']) else 'PASS'
    result['total_rows'] = total_rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path, help='Dataset root or parent directory; recursively scanned')
    parser.add_argument('--report', type=Path, help='Output JSON path (default: timestamped file in current directory)')
    parser.add_argument('--max-receive-skew-ms', type=float, default=80)
    args = parser.parse_args()
    parent = args.parent.expanduser().resolve()
    if not parent.is_dir():
        parser.error(f'not a directory: {parent}')
    # Also discover partially written datasets with missing info.json.
    roots = set()
    for p in parent.rglob('*'):
        if p.is_dir() and p.name == 'meta':
            roots.add(p.parent)
        elif p.is_file() and p.suffix == '.parquet' and p.parent.parent.name == 'data':
            roots.add(p.parent.parent.parent)
    report = dict(parent=str(parent), checked_at=datetime.now().astimezone().isoformat(),
                  thresholds=dict(max_receive_skew_ms=args.max_receive_skew_ms, max_camera_drop_ratio=.05, max_camera_sequence_step=3), datasets=[])
    for root in sorted(roots):
        print(f'\nDATASET {root}', flush=True)
        try:
            result = audit(root, args.max_receive_skew_ms)
        except Exception as exc:
            result = dict(path=str(root), status='FAIL', errors=[f'{type(exc).__name__}: {exc}'], episodes=[])
        report['datasets'].append(result)
        for error in result['errors']:
            print('  ERROR: ' + error, flush=True)
        print(f"{result['status']} dataset {root.name}", flush=True)
    passed = sum(r['status'] == 'PASS' for r in report['datasets'])
    report['summary'] = dict(datasets=len(roots), passed=passed, failed=len(roots)-passed)
    output = args.report or Path.cwd()/datetime.now().strftime('dataset-audit-%Y%m%d-%H%M%S.json')
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'\nSUMMARY datasets={len(roots)} PASS={passed} FAIL={len(roots)-passed}\nJSON: {output}')
    print('PASS means automated checks passed, not task success or visual semantic correctness. Scan after stopping and saving recording.')
    return 2 if not roots else (1 if passed != len(roots) else 0)

if __name__ == '__main__':
    sys.exit(main())
