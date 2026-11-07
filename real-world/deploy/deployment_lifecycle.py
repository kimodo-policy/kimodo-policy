"""Data-sourced preparation and a live last-reference hold; no robot I/O here."""
from __future__ import annotations

import copy
import json
import select
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kimodo_sonic_bridge import MUJOCO_TO_ISAACLAB, SafetyLimits, normalize_quat_wxyz


class InitialPoseController:
    def __init__(self, path, *, task='teleop-banana'):
        self.config = json.loads(Path(path).read_text())
        c = self.config
        if c['version'] != 1 or c['task'] != task:
            raise ValueError(f'Initial pose task mismatch: expected {task}, got {c.get("task")}')
        names = c['joint_order']
        if len(names) != 29 or names[18] != 'left_elbow_joint' or names[25] != 'right_elbow_joint':
            raise ValueError('Initial pose must use the documented MuJoCo joint order')
        self.target_q = np.asarray(c['reference_joint_q_mujoco'], dtype=np.float32)[MUJOCO_TO_ISAACLAB]
        self.expected_q = np.asarray(c['expected_observed_joint_q_mujoco'], dtype=np.float32)[MUJOCO_TO_ISAACLAB]
        self.settle_basis = c.get('settle_basis', 'recorded_observation')
        if self.settle_basis not in ('recorded_observation', 'reference_tracking'):
            raise ValueError('Invalid initial pose settle_basis')
        self.arms = np.flatnonzero(MUJOCO_TO_ISAACLAB >= 15)
        self.target_quat = normalize_quat_wxyz(c['root_quat_wxyz'])
        self.target_root = np.array([0, 0, c['root_height_m']], dtype=np.float32)
        self.hands = np.asarray(c['hand_closure'], dtype=np.float32)
        limits = SafetyLimits()
        if (self.target_q.shape != (29,) or self.expected_q.shape != (29,)
                or not np.isfinite([self.target_q, self.expected_q]).all()
                or np.max(np.abs(self.target_q)) > limits.max_joint_abs
                or not limits.min_root_z <= self.target_root[2] <= limits.max_root_z
                or self.hands.shape != (2,) or not np.isfinite(self.hands).all()
                or np.any((self.hands < 0) | (self.hands > 1))):
            raise ValueError('Invalid initial pose values')
        for name in ('transition_min_s', 'joint_speed_limit_rad_s', 'root_speed_limit_m_s',
                     'settle_s', 'history_s', 'arm_rms_tolerance_rad', 'arm_max_tolerance_rad',
                     'body_rms_tolerance_rad', 'velocity_rms_tolerance_rad_s', 'orientation_tolerance_rad'):
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f'Invalid initial pose setting: {name}')
        self.phase = 'waiting_for_state'
        self.reference = None
        self.last_time = None
        self.elapsed = self.stable = self.history_elapsed = 0.0
        self.reset_history = False
        self.errors = {}

    @property
    def ready(self):
        return self.phase == 'ready'

    def step(self, now, q, dq, quat, *, valid, root_height=.793):
        self.reset_history = False
        dt = 0 if self.last_time is None else np.clip(now-self.last_time, 0, .04)
        self.last_time = now
        if self.ready:
            return self.reference
        if not valid:
            self.stable = self.history_elapsed = 0
            if self.phase == 'history':
                self.phase = 'settling'
            return self.reference
        q, dq = np.asarray(q), np.asarray(dq)
        if self.reference is None:
            self.start_q = q.copy()
            self.start_root = np.array([0, 0, root_height], dtype=np.float32)
            self.start_quat = normalize_quat_wxyz(quat)
            rotations = Rotation.from_quat(np.stack([self.start_quat, self.target_quat])[:, [1,2,3,0]])
            self.slerp = Slerp([0,1], rotations)
            self.duration = max(self.config['transition_min_s'],
                                1.5*float(np.max(abs(self.target_q-q)))/self.config['joint_speed_limit_rad_s'],
                                1.5*abs(float(self.target_root[2]-root_height))/self.config['root_speed_limit_m_s'],
                                1.5*(rotations[0].inv()*rotations[1]).magnitude()/.15)
            self.phase = 'transition'
        if self.phase == 'transition':
            self.elapsed += dt
            x = min(1., self.elapsed/self.duration)
            alpha = x*x*(3-2*x)
            velocity = 6*x*(1-x)/self.duration
            self.reference = dict(joint_pos=self.start_q+alpha*(self.target_q-self.start_q),
                                  joint_vel=velocity*(self.target_q-self.start_q),
                                  body_quat=self.slerp([alpha]).as_quat()[0][[3,0,1,2]].astype(np.float32),
                                  root_position=self.start_root+alpha*(self.target_root-self.start_root),
                                  hand_closure=self.hands.copy(), hand_binary=(self.hands>=.5).astype(np.float32), frame_index=0)
            if x == 1:
                self.reference['joint_pos'] = self.target_q.copy()
                self.reference['body_quat'] = self.target_quat.copy()
                self.reference['root_position'] = self.target_root.copy()
                self.phase = 'settling'
            return self.reference
        recorded_error = q-self.expected_q
        error = q-(self.target_q if self.settle_basis == "reference_tracking" else self.expected_q)
        orientation_error = (Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).inv()*Rotation.from_quat(self.target_quat[[1,2,3,0]])).magnitude()
        self.errors = dict(recorded_arm_rms=float(np.sqrt(np.mean(recorded_error[self.arms]**2))), orientation=float(orientation_error), arm_rms=float(np.sqrt(np.mean(error[self.arms]**2))),
                           arm_max=float(np.max(abs(error[self.arms]))),
                           body_rms=float(np.sqrt(np.mean(error**2))),
                           velocity_rms=float(np.sqrt(np.mean(dq**2))))
        stable = all(self.errors[key] <= self.config[setting] for key,setting in
                     [('arm_rms','arm_rms_tolerance_rad'),('arm_max','arm_max_tolerance_rad'),
                      ('body_rms','body_rms_tolerance_rad'),('velocity_rms','velocity_rms_tolerance_rad_s'),
                      ('orientation','orientation_tolerance_rad')])
        if not stable:
            self.stable = self.history_elapsed = 0
            self.phase = 'settling'
        elif self.phase == 'settling':
            self.stable += dt
            if self.stable >= self.config['settle_s']:
                self.phase = 'history'
                self.reset_history = True
        elif self.phase == 'history':
            self.history_elapsed += dt
            if self.history_elapsed >= self.config['history_s']:
                self.phase = 'ready'
        return self.reference


def frozen_reference(reference):
    """Preserve pose/hand commands; a static hold has zero reference velocity."""
    result = copy.deepcopy(reference)
    result['joint_vel'] = np.zeros(29, dtype=np.float32)
    return result


def _operator_finished_hold():
    if not sys.stdin.isatty():
        return False
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower() == 'exit'
    return False


def _hold_notice(message, **kwargs):
    # A terminal pipeline (tee) may close on Ctrl+C. Logging must not stop holding.
    try:
        print(message, **kwargs)
    except (BrokenPipeError, OSError):
        pass


def hold_last_reference(reference, hands, *, window, next_frame, send, recorder,
                        rate=50., should_exit=None, clock=time.monotonic, sleep=time.sleep):
    """Keep SONIC alive with poses only. Never send start/stop or open the hand.

    The caller must keep the same publisher, window and monotonically increasing
    frame counter. This cannot protect against loss of power/SONIC/network.
    """
    reference = frozen_reference(reference)
    left, right = (None if hand is None else np.array(hand, copy=True) for hand in hands)
    should_exit = should_exit or _operator_finished_hold
    recorder.emit('hold_last_started', reference=reference, next_frame=next_frame)
    _hold_notice('[HOLD] Task inference stopped. Streaming the last published pose and hand targets at 50 Hz. '
          'Keep this terminal and SONIC running. After securing the robot and stopping SONIC, '
          'type exit + Enter here. Ctrl+C keeps holding.', flush=True)
    interrupted = [False]
    def keep_holding(signum, frame):
        interrupted[0] = True
    previous_handler = signal.signal(signal.SIGINT, keep_holding)
    try:
        while True:
            try:
                if interrupted[0]:
                    _hold_notice('[HOLD] Still holding. Stop SONIC safely first; then type exit + Enter.', flush=True)
                    interrupted[0] = False
                if should_exit():
                    recorder.emit('hold_last_operator_exit', next_frame=next_frame)
                    return next_frame
                start = clock()
                packet = window.append(joint_pos_isaaclab=reference['joint_pos'],
                                       joint_vel_isaaclab=reference['joint_vel'],
                                       body_quat_wxyz=reference['body_quat'],
                                       root_position=reference['root_position'], frame_index=next_frame,
                                       timestamp_monotonic=start, left_hand_joints=left, right_hand_joints=right)
                next_frame += 1
                if packet is not None:
                    send(packet)
                if next_frame % int(rate) == 0:
                    recorder.emit('hold_last_heartbeat', next_frame=next_frame)
                sleep(max(0, 1/rate-(clock()-start)))
            except KeyboardInterrupt:
                _hold_notice('[HOLD] Still holding. Stop SONIC safely first; then type exit + Enter.', flush=True)
            except Exception as exc:
                recorder.emit('hold_last_error', error=str(exc))
                _hold_notice(f'[HOLD] Publish failed: {exc}; check SONIC/connection immediately.', flush=True)
                sleep(.1)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
