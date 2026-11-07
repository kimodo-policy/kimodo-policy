import unittest
from collections import deque
import numpy as np
from kimodo_sonic_bridge import SonicPoseWindow, KimodoActionRuntime
from test_deployment_lifecycle import unpack
from run_kimodo_sonic import _preview_reference_queue

class LookaheadTests(unittest.TestCase):
    def test_future_is_committed_then_hold_and_interrupt_replaces_future(self):
        window=SonicPoseWindow(5,46)
        pose=dict(joint_pos_isaaclab=np.zeros(29),joint_vel_isaaclab=np.zeros(29),body_quat_wxyz=[1,0,0,0],root_position=[0,0,.8],left_hand_joints=np.zeros(7),right_hand_joints=np.ones(7)*.1)
        future=[dict(joint_pos=np.ones(29)*k*.01,joint_vel=np.ones(29)*.5,body_quat=[1,0,0,0],root_position=[k*.001,0,.8]) for k in (1,2,3)]
        for frame in range(5):packet=window.append(**pose,frame_index=frame,timestamp_monotonic=frame*.02,future_references=future)
        arrays=unpack(packet)
        np.testing.assert_array_equal(arrays['frame_index'],np.arange(51))
        np.testing.assert_allclose(arrays['joint_pos'][5:8,0],[.01,.02,.03])
        np.testing.assert_allclose(arrays['joint_pos'][8:],.03)
        np.testing.assert_allclose(arrays['joint_vel'][8:],0)
        # Same operation used by Ctrl+C hold: no future argument means hold.
        pose['joint_pos_isaaclab']=np.ones(29)*.1
        packet=window.append(**pose,frame_index=5,timestamp_monotonic=.1)
        arrays=unpack(packet)
        np.testing.assert_array_equal(arrays['frame_index'],np.arange(1,52))
        np.testing.assert_allclose(arrays['joint_pos'][4:],.1)
        np.testing.assert_allclose(arrays['joint_vel'][4:],0)
        np.testing.assert_allclose(arrays['reference_timestamp_monotonic'],np.arange(1,52)*.02)
        # History retains actual emitted frames only, never speculative futures.
        self.assertEqual(len(window.frames),5)
        self.assertEqual(int(window.frames[-1]['frame_index'][0]),5)

    def test_ctrl_c_hold_overwrites_lookahead_and_keeps_indices(self):
        from deployment_lifecycle import hold_last_reference
        window=SonicPoseWindow(5,46);sent=[]
        pose=dict(joint_pos=np.ones(29)*.1,joint_vel=np.ones(29),body_quat=np.array([1,0,0,0]),root_position=np.array([0,0,.8]),hand_closure=np.array([0,1]))
        moving=dict(pose,joint_pos=np.ones(29)*.2)
        for k in range(5):window.append(joint_pos_isaaclab=pose['joint_pos'],joint_vel_isaaclab=pose['joint_vel'],body_quat_wxyz=pose['body_quat'],root_position=pose['root_position'],frame_index=k,timestamp_monotonic=k*.02,future_references=[moving]*46)
        class Recorder:
            def emit(self,*args,**kwargs):pass
        end=hold_last_reference(pose,(np.zeros(7),np.ones(7)),window=window,next_frame=5,send=sent.append,recorder=Recorder(),should_exit=lambda:len(sent)==3,sleep=lambda _:None)
        self.assertEqual(end,8)
        for k,packet in enumerate(sent):
            arrays=unpack(packet)
            np.testing.assert_array_equal(arrays['frame_index'],np.arange(1+k,52+k))
            np.testing.assert_allclose(arrays['joint_pos'][4:],.1)
            np.testing.assert_allclose(arrays['joint_vel'][4:],0)
            np.testing.assert_allclose(arrays['right_hand_joints'],1)

    def test_preview_does_not_advance_live_runtime_or_consume_actions(self):
        runtime=KimodoActionRuntime();actions=deque()
        for k in range(1,10):
            a=np.zeros(40,np.float32);a[:3]=[.001,0,.8];a[3:9]=[1,0,0,1,0,0];a[9:38]=.001*k;actions.append(a)
        refs=_preview_reference_queue(runtime,actions,np.zeros(29),[1,0,0,0])
        self.assertEqual(runtime.frame_index,0);self.assertIsNone(runtime.previous_joint_q)
        np.testing.assert_array_equal(runtime.reference_root_position,[0,0,0]);self.assertEqual(len(actions),9)
        runtime.prime_joints(np.zeros(29))
        for action,expected in zip(actions,refs):
            actual=runtime.decode(action,current_root_quat_wxyz=[1,0,0,0])
            for key in ('joint_pos','joint_vel','body_quat','root_position'):np.testing.assert_array_equal(actual[key],expected[key])

if __name__=='__main__':unittest.main()
