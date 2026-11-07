import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from deployment_lifecycle import InitialPoseController, hold_last_reference
from kimodo_sonic_bridge import SonicPoseWindow, SONIC_HEADER_SIZE


def unpack(packet):
    fields=json.loads(packet[4:4+SONIC_HEADER_SIZE].rstrip(b'\x00'))['fields']
    offset=4+SONIC_HEADER_SIZE; out={}
    for f in fields:
        dtype={'f32':'<f4','f64':'<f8','i64':'<i8','bool':'?'}[f['dtype']]
        a=np.frombuffer(packet,dtype=dtype,count=int(np.prod(f['shape'])),offset=offset).reshape(f['shape'])
        out[f['name']]=a;offset+=a.nbytes
    return out

class LifecycleTests(unittest.TestCase):
    def controller(self):
        return InitialPoseController(Path(__file__).with_name('banana_initial_pose.json'))

    def test_each_task_uses_its_own_data_pose(self):
        for task in ('banana', 'boxing', 'pushing', 'transportation',
                     'sitting', 'football', 'pnp', 'long'):
            path = Path(__file__).with_name(f'{task}_initial_pose.json')
            controller = InitialPoseController(path, task=f'teleop-{task}')
            self.assertEqual(controller.target_q.shape, (29,))
            with self.assertRaisesRegex(ValueError, 'task mismatch'):
                InitialPoseController(path, task='teleop-wrong')

    def test_requires_measured_pose_and_new_stable_history(self):
        p=self.controller(); q=p.target_q.copy();q[p.arms]+=.7
        previous=None; reset_count=0
        for k in range(1500):
            ref=p.step(k*.02,q,np.zeros(29),[1,0,0,0],valid=True)
            if previous is not None:
                self.assertLessEqual(np.max(abs(ref['joint_pos']-previous))/.02,.301)
            previous=ref['joint_pos'].copy()
            self.assertFalse(p.ready)
        self.assertEqual(p.phase,'settling')
        for k in range(1500,1800):
            p.step(k*.02,p.expected_q,np.zeros(29),[1,0,0,0],valid=True)
            reset_count+=p.reset_history
            if k<1720:self.assertFalse(p.ready)
        self.assertTrue(p.ready);self.assertEqual(reset_count,1)
        np.testing.assert_allclose(p.reference['joint_pos'],p.target_q)
        np.testing.assert_array_equal(p.reference['joint_vel'],np.zeros(29))

    def test_invalid_sensor_resets_settling_time(self):
        p=self.controller()
        for k in range(1500):p.step(k*.02,p.expected_q,np.zeros(29),[1,0,0,0],valid=(k%30!=0))
        self.assertFalse(p.ready)

    def test_hold_preserves_last_pose_hands_and_stream_indices(self):
        sent=[];events=[];window=SonicPoseWindow(5)
        pose=dict(joint_pos=np.arange(29,dtype=float)*.01,joint_vel=np.ones(29),
                  body_quat=np.array([.8,0,0,-.6]),root_position=np.array([.4,-.2,.86]),
                  hand_closure=np.array([0.,1.]))
        left=np.zeros(7);right=np.array([0,.7,.7,-1,-1.5,-1,-1.5])
        for f in range(5):
            window.append(joint_pos_isaaclab=pose['joint_pos'],joint_vel_isaaclab=pose['joint_vel'],body_quat_wxyz=pose['body_quat'],root_position=pose['root_position'],frame_index=f,timestamp_monotonic=f*.02,left_hand_joints=left,right_hand_joints=right)
        class Recorder:
            def emit(self,*args,**kw):events.append((args,kw))
        calls=[0]
        def done():
            calls[0]+=1
            if calls[0]==3:raise KeyboardInterrupt()
            return len(sent)>=7
        with patch("builtins.print", side_effect=BrokenPipeError("closed tee")):
            end=hold_last_reference(pose,(left,right),window=window,next_frame=5,send=sent.append,recorder=Recorder(),should_exit=done,sleep=lambda _:None)
        self.assertEqual(end,12);self.assertEqual(len(sent),7)
        arrays=unpack(sent[-1]);np.testing.assert_array_equal(arrays['frame_index'],np.arange(7,12))
        np.testing.assert_allclose(arrays['joint_pos'],np.tile(pose['joint_pos'],(5,1)))
        np.testing.assert_allclose(arrays['joint_vel'],np.zeros((5,29)))
        np.testing.assert_allclose(arrays['body_quat'][:,0],np.tile(pose['body_quat'],(5,1)))
        np.testing.assert_allclose(arrays['root_position'],np.tile(pose['root_position'],(5,1)))
        np.testing.assert_allclose(arrays['right_hand_joints'],right[None,:])
        self.assertTrue(all(packet.startswith(b'pose') for packet in sent))
        np.testing.assert_array_equal(pose['joint_vel'],np.ones(29))

if __name__=='__main__':unittest.main()
