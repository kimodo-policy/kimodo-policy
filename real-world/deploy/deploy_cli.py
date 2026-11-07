"""Task-selectable H20 server and laptop bridge launcher for real-world Kimodo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("KIMODO_PROJECT_ROOT", DEPLOY_DIR.parent.parent)).expanduser().resolve()
CONFIG_PATH = Path(os.environ.get('KIMODO_DEPLOY_CONFIG', DEPLOY_DIR / 'deploy_config.json'))
MODES = ('server', 'check', 'dry-run', 'run', 'show')


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise ValueError('Create deploy_config.json from deploy_config.example.json and configure this host first')
    config = json.loads(CONFIG_PATH.read_text())
    if config.get('version') != 1 or not isinstance(config.get('tasks'), dict):
        raise ValueError(f'Unsupported deployment configuration: {CONFIG_PATH}')
    if os.environ.get('KIMODO_SERVER_PYTHON'):
        config['server']['python'] = os.environ['KIMODO_SERVER_PYTHON']
    if os.environ.get('COLLECTION_PYTHON'):
        config['bridge']['python'] = os.environ['COLLECTION_PYTHON']
    if os.environ.get('CAMERA_HOST'):
        config['bridge']['camera_host'] = os.environ['CAMERA_HOST']
    port = int(os.environ.get('KIMODO_PORT', config['server']['port']))
    if not 1 <= port <= 65535:
        raise ValueError('Model HTTP port must be between 1 and 65535')
    config['server']['port'] = port
    return config


def require_task(config: dict, task: str) -> dict:
    if task not in config['tasks']:
        raise ValueError(f'Unknown task {task!r}; choose: {", ".join(sorted(config["tasks"]))}')
    return config['tasks'][task]


def dataset_name(config: dict, task: str) -> str:
    return require_task(config, task).get('dataset', f'teleop-{task}')


def resolve_checkpoint(task: str, setting: str | int, dataset: str | None = None) -> Path:
    dataset = dataset or f'teleop-{task}'
    if setting == 'latest' or isinstance(setting, int) or str(setting).isdigit():
        root = PROJECT_ROOT / 'log/Real_World' / (
            'ft_real_world_single_gbs64_5w_controlnet4_detach_true_mse_' + dataset
        )
        choices = []
        for path in root.glob('*/checkpoint_*'):
            match = re.fullmatch(r'checkpoint_(\d+)', path.name)
            if match and (path / 'training_state.pt').is_file() and (path / 'config.json').is_file():
                if setting != 'latest' and int(match.group(1)) != int(setting):
                    continue
                if all((path / f'rng_state_rank_{rank:05d}.pt').is_file() for rank in range(8)):
                    choices.append((int(match.group(1)), path.stat().st_mtime, path))
        if not choices:
            raise ValueError(f'No complete checkpoint for {task} at {setting!r}; '
                             f'wait for it or set tasks.{task}.checkpoint to another step/path')
        path = max(choices)[2]
    else:
        path = Path(setting).expanduser().resolve()
    if not (path / 'training_state.pt').is_file() or not (path / 'config.json').is_file():
        raise ValueError(f'Checkpoint is missing training_state.pt or config.json: {path}')
    checkpoint_config = json.loads((path / 'config.json').read_text())
    actual_task = checkpoint_config['main']['dataset_selection']['RealWorld']['dataset']
    expected_task = dataset
    if actual_task != expected_task:
        raise ValueError(f'Checkpoint dataset {actual_task!r} does not match {expected_task!r}: {path}')
    return path


def get_pose(config: dict, task: str) -> Path:
    setting = require_task(config, task)['initial_pose']
    path = Path(setting)
    if not path.is_absolute():
        path = DEPLOY_DIR / path
    pose = json.loads(path.read_text())
    if pose.get('version') != 1 or pose.get('task') != dataset_name(config, task):
        raise ValueError(f'Initial pose does not belong to teleop-{task}: {path}')
    if len(pose.get('reference_joint_q_mujoco', [])) != 29:
        raise ValueError(f'Initial pose must have 29 reference joints: {path}')
    return path


def health(port: int) -> dict:
    url = f'http://127.0.0.1:{port}/health'
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f'Cannot reach Kimodo server at {url}; check server and SSH tunnel') from exc


def server_command(config: dict, task: str, args: argparse.Namespace) -> tuple[list[str], dict]:
    cfg = config['server']
    task_cfg = require_task(config, task)
    checkpoint = resolve_checkpoint(task, args.checkpoint or task_cfg['checkpoint'], dataset_name(config, task))
    gpu = cfg['gpu'] if args.gpu is None else args.gpu
    port = cfg['port'] if args.port is None else args.port
    diffusion = cfg['diffusion_steps'] if args.diffusion_steps is None else args.diffusion_steps
    rtc = cfg['rtc'] if args.rtc is None else bool(args.rtc)
    execution = cfg['execution_frames'] if args.execution_frames is None else args.execution_frames
    overlap = cfg['rtc_overlap_frames'] if args.rtc_overlap_frames is None else args.rtc_overlap_frames
    frozen = cfg['rtc_frozen_frames'] if args.rtc_frozen_frames is None else args.rtc_frozen_frames
    ramp = cfg['rtc_ramp_power'] if args.rtc_ramp_power is None else args.rtc_ramp_power
    if not 1 <= int(port) <= 65535 or int(gpu) < 0 or int(diffusion) < 1:
        raise ValueError('Invalid port, GPU index, or diffusion steps')
    if not 1 <= int(execution) <= 50 or not 0 <= int(frozen) <= int(overlap) <= 50 - int(execution):
        raise ValueError('Require 1 <= execution_frames <= 50 and 0 <= frozen <= overlap <= 50 - execution')
    if float(ramp) <= 0:
        raise ValueError('rtc_ramp_power must be positive')
    with socket.socket() as sock:
        sock.settimeout(0.4)
        if sock.connect_ex(('127.0.0.1', int(port))) == 0 and not args.print_only:
            raise RuntimeError(f'Port {port} is already in use; stop the old server before switching tasks')
    command = [
        cfg['python'], '-u', str(DEPLOY_DIR / 'real_world_server.py'),
        '--checkpoint', str(checkpoint),
        '--served-task', dataset_name(config, task),
        '--text-embedding-cache', str(Path(os.environ.get('KIMODO_TEXT_CACHE', PROJECT_ROOT / 'data/cache/RealWorld'))),
        '--host', '127.0.0.1', '--port', str(port), '--device', 'cuda:0',
        '--dtype', cfg['dtype'], '--diffusion-steps', str(diffusion),
        '--control-fps', str(cfg['control_fps']), '--rtc', str(int(rtc)),
        '--execution-frames', str(execution), '--rtc-overlap-frames', str(overlap),
        '--rtc-frozen-frames', str(frozen), '--rtc-ramp-power', str(ramp),
        '--hand-rtc', str(int(cfg.get('hand_rtc', True))),
        '--hand-stabilization', str(int(cfg.get('hand_stabilization', True))),
        '--hand-close-threshold', str(cfg.get('hand_close_threshold', .65)),
        '--hand-open-threshold', str(cfg.get('hand_open_threshold', .35)),
        '--hand-close-frames', str(cfg.get('hand_close_frames', 6)),
        '--hand-open-frames', str(config['tasks'][task].get('hand_open_frames', cfg.get('hand_open_frames', 21))),
    ]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
    env.pop('PYTHONPATH', None)
    print(f'[SERVER] task={dataset_name(config, task)} checkpoint={checkpoint} gpu={gpu} '
          f'diffusion={diffusion} rtc={int(rtc)} execution={execution} '
          f'overlap={overlap} frozen={frozen}', flush=True)
    return command, env


def bridge_command(config: dict, task: str, mode: str, args: argparse.Namespace) -> tuple[list[str], dict, Path]:
    cfg = config['bridge']
    server_cfg = config['server']
    port = server_cfg['port'] if args.port is None else args.port
    pose = get_pose(config, task)
    if not args.print_only:
        status = health(port)
        if status.get('served_task') != dataset_name(config, task):
            raise RuntimeError(f'Server reports task {status.get("served_task")!r}; expected {dataset_name(config, task)}. '
                               'Restart the H20 server with the selected task.')
        print(f'[SERVER] {status["served_task"]} checkpoint={status.get("checkpoint")} '
              f'diffusion={status.get("diffusion_steps")} rtc={status.get("rtc_enabled")} '
              f'execution={status.get("execution_frames")}', flush=True)
        import subprocess
        if subprocess.run(['pgrep', '-f', '[r]un_kimodo_sonic.py'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('Another Kimodo bridge is running or holding the last pose; exit it before switching tasks')
        if mode != 'dry-run':
            result = subprocess.run(['pgrep', '-f', '[p]ico_manager_thread_server.py'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                raise RuntimeError('Stop PICO manager before model deployment; both use port 5556')
    command = [
        cfg['python'], '-u', str(DEPLOY_DIR / 'run_kimodo_sonic.py'),
        '--server-url', f'http://127.0.0.1:{port}', '--task', dataset_name(config, task),
        '--state-kp-host', '127.0.0.1', '--state-kp-port', str(cfg.get('state_kp_port', cfg.get('state_zmq_port', 5557))),
        '--camera-host', cfg['camera_host'], '--camera-port', str(cfg['camera_port']),
        '--action-kp-host', '127.0.0.1', '--action-kp-port', str(cfg.get('action_kp_port', cfg.get('action_zmq_port', 5556))),
        '--request-when-queue-below', str(cfg['request_when_queue_below']),
        '--initial-inference-latency-s', str(cfg['initial_inference_latency_s']),
        '--inference-lead-margin-s', str(cfg['inference_lead_margin_s']),
        '--sonic-lookahead-frames', str(cfg['sonic_lookahead_frames']),
        '--max-reference-xy-radius', str(cfg['max_reference_xy_radius']),
        '--hand-control-mode', cfg['hand_control_mode'], '--verbose',
    ]
    if mode == 'run':
        command += ['--auto-start', '--initial-pose', str(pose), '--hold-on-interrupt']
    elif mode == 'check':
        command += ['--no-publish', '--max-steps', str(args.max_steps or cfg['check_steps'])]
    else:
        command += ['--dry-run', '--max-steps', str(args.max_steps or cfg['dry_run_steps'])]
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_root = Path(os.environ.get('KIMODO_DIAGNOSTICS_ROOT', DEPLOY_DIR / 'logs'))
    run_dir = log_root / f'{stamp}-{task}-{mode}-{os.getpid()}'
    command += ['--diagnostics-dir', str(run_dir)]
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env.pop('PYTHONPATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    print(f'[BRIDGE] task={dataset_name(config, task)} mode={mode} initial_pose={pose} logs={run_dir}', flush=True)
    return command, env, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('task')
    parser.add_argument('--checkpoint', help='H20 server: override configured checkpoint path or latest')
    parser.add_argument('--gpu', type=int, help='H20 server: physical GPU index')
    parser.add_argument('--port', type=int, help='Server and bridge port')
    parser.add_argument('--diffusion-steps', type=int)
    parser.add_argument('--rtc', type=int, choices=(0, 1))
    parser.add_argument('--execution-frames', type=int)
    parser.add_argument('--rtc-overlap-frames', type=int)
    parser.add_argument('--rtc-frozen-frames', type=int)
    parser.add_argument('--rtc-ramp-power', type=float)
    parser.add_argument('--max-steps', type=int, help='Bridge check/dry-run steps')
    parser.add_argument('--print-only', action='store_true', help='Validate and print command; do not connect or execute')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    require_task(config, args.task)
    if args.mode == 'show':
        print(json.dumps({'task': args.task, 'server': config['server'],
                          'bridge': config['bridge'], 'task_config': config['tasks'][args.task]}, indent=2))
        return
    if args.mode == 'server':
        command, env = server_command(config, args.task, args)
        if args.print_only:
            print(shlex.join(command))
            return
        os.chdir(PROJECT_ROOT)
        os.execvpe(command[0], command, env)
    command, env, run_dir = bridge_command(config, args.task, args.mode, args)
    if args.print_only:
        print(shlex.join(command))
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    script = 'exec > >(tee -a ' + shlex.quote(str(run_dir / 'console.log')) + \
             ') 2>&1; exec ' + shlex.join(command)
    os.execvpe('/bin/bash', ['/bin/bash', '-c', script], env)


if __name__ == '__main__':
    try:
        main()
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f'deploy: {exc}', file=sys.stderr)
        raise SystemExit(2)
