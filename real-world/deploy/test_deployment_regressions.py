import gzip,json,threading,unittest
from http.client import HTTPConnection
from collections import deque
import numpy as np
import real_world_server as server
from kimodo_rtc import BridgePlanTracker
from kimodo_sonic_bridge import KimodoActionRuntime
from run_kimodo_sonic import _admit_result

class Runtime:
    control_fps=50
    def reset(self,seed=None,session_id=None):pass

class RegressionTests(unittest.TestCase):
    def test_pending_child_survives_queue_exhaustion_but_stale_does_not(self):
        for age,expected in [(.65,True),(1.1,False)]:
            t=BridgePlanTracker(control_fps=50,chunk_frames=63,queue_cap=40)
            parent=t.begin();t.accept(parent);child=t.begin()
            self.assertFalse(t.exhausted_queue_needs_replan())
            runtime=KimodoActionRuntime();actions=np.zeros((63,40),np.float32)
            actions[:,2]=.793;actions[:,3:9]=[1,0,0,1,0,0]
            result=dict(deployment=child,submitted_monotonic=10,end_state_ts=10,queue_at_submission=25,ok=True,action=actions)
            q=deque()
            ok,details=_admit_result(result,t,runtime,q,now=10+age,latest_state_ts=10+age,measured_joints=np.zeros(29),root_quat=np.array([1,0,0,0]),max_state_lag_s=1,max_result_age_s=1,max_start_age_s=1)
            self.assertEqual(ok,expected,details)
            self.assertEqual(len(q),63 if expected else 0)
            if expected:self.assertTrue(t.exhausted_queue_needs_replan())
            else:self.assertEqual(details['reason'],'stale Kimodo result')
    def test_sonic_window_keeps_skipped_frames_and_original_targets(self):
        from kimodo_sonic_bridge import SonicPoseWindow, SONIC_HEADER_SIZE
        window = SonicPoseWindow(5)
        def unpack(packet):
            header = json.loads(packet[4:4+SONIC_HEADER_SIZE].rstrip(b"\x00"))
            offset = 4+SONIC_HEADER_SIZE
            values = {}
            for field in header["fields"]:
                dtype = {"f32":"<f4", "f64":"<f8", "i64":"<i8", "bool":"?"}[field["dtype"]]
                array = np.frombuffer(packet,dtype=dtype,count=int(np.prod(field["shape"])),offset=offset).reshape(field["shape"])
                values[field["name"]] = array
                offset += array.nbytes
            self.assertEqual(offset,len(packet))
            return values
        received = []
        for frame in range(10):
            theta = -frame*.1
            packet = window.append(joint_pos_isaaclab=np.full(29,frame*.01),joint_vel_isaaclab=np.zeros(29),body_quat_wxyz=[np.cos(theta/2),0,0,np.sin(theta/2)],root_position=[0,0,.793],frame_index=frame,timestamp_monotonic=10+frame*.02,left_hand_joints=np.zeros(7),right_hand_joints=np.full(7,frame*.02))
            if frame<4:
                self.assertIsNone(packet)
            elif frame != 7:  # Simulate overwritten transport packet.
                received.append(unpack(packet))
        after_skip = received[-2]
        np.testing.assert_array_equal(after_skip["frame_index"],np.arange(4,9))
        np.testing.assert_allclose(after_skip["joint_pos"][:,0],np.arange(4,9)*.01)
        np.testing.assert_allclose(after_skip["body_quat"][:,0,3],np.sin(-np.arange(4,9)*.05),atol=1e-7)
        np.testing.assert_allclose(after_skip["reference_timestamp_monotonic"],10+np.arange(4,9)*.02)
        np.testing.assert_allclose(after_skip["right_hand_joints"],np.full((1,7),.16))
        self.assertEqual(after_skip["catch_up"].tolist(),[True])

    def test_optional_reference_radius_keeps_local_limits(self):
        from kimodo_sonic_bridge import SafetyLimits, BridgeSafetyError
        action=np.zeros(40,np.float32);action[0]=.01;action[2]=.8;action[3:9]=[1,0,0,1,0,0]
        for radius in (10.,0.):
            runtime=KimodoActionRuntime(limits=SafetyLimits(max_root_xy_abs=radius))
            runtime.reference_root_position[:2]=[9.999,0]
            if radius:
                with self.assertRaisesRegex(BridgeSafetyError,'XY workspace'):
                    runtime.decode(action,current_root_quat_wxyz=[1,0,0,0])
            else:
                ref=runtime.decode(action,current_root_quat_wxyz=[1,0,0,0])
                self.assertGreater(ref['root_position'][0],10)
                bad=action.copy();bad[0]=.16
                with self.assertRaisesRegex(BridgeSafetyError,'one-frame XY jump'):
                    runtime.decode(bad,current_root_quat_wxyz=[1,0,0,0])
                bad=action.copy();bad[9]=.71
                with self.assertRaisesRegex(BridgeSafetyError,'joint velocity'):
                    runtime.decode(bad,current_root_quat_wxyz=[1,0,0,0])

    def test_client_reuses_http_connection(self):
        from run_kimodo_sonic import KimodoHttpClient, _inference_is_stale
        h=type('H',(server.RequestHandler,),dict(runtime=Runtime(),max_request_bytes=1024))
        http=server.ThreadingHTTPServer(('127.0.0.1',0),h)
        worker=threading.Thread(target=http.serve_forever);worker.start()
        client=KimodoHttpClient(f'http://127.0.0.1:{http.server_port}')
        try:
            self.assertEqual(client._post('/reset',{})['status'],'ok')
            sock=client._connections.connection.sock
            self.assertEqual(client._post('/reset',{})['status'],'ok')
            self.assertIs(client._connections.connection.sock,sock)
            self.assertFalse(_inference_is_stale(True,10,10.6,.25))
            self.assertTrue(_inference_is_stale(False,10,10.6,.25))
        finally:
            client._connections.connection.close()
            http.shutdown();http.server_close();worker.join()

    def test_gzip_identity_and_decompressed_limit(self):
        h=type('H',(server.RequestHandler,),dict(runtime=Runtime(),max_request_bytes=1024))
        http=server.ThreadingHTTPServer(('127.0.0.1',0),h);worker=threading.Thread(target=http.serve_forever);worker.start()
        try:
            for body,encoding,status in [(b'{}','identity',200),(gzip.compress(b'{}'),'gzip',200),(gzip.compress(b' '*2048),'gzip',400),(b'bad','gzip',400)]:
                c=HTTPConnection('127.0.0.1',http.server_port,timeout=2)
                c.request('POST','/reset',body,{'Content-Encoding':encoding,'Content-Type':'application/json'})
                r=c.getresponse();self.assertEqual(r.status,status,r.read());c.close()
        finally:http.shutdown();http.server_close();worker.join()
if __name__=='__main__':unittest.main()
