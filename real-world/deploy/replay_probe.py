"""Offline demonstration-window probe; HTTP inference only, never robot I/O."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av
from scipy.spatial.transform import Rotation as R,Slerp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.realworld_loader import project_realworld_hand_closure
from run_kimodo_sonic import KimodoHttpClient
from kimodo_sonic_bridge import state_msg_to_kimodo_state, MUJOCO_TO_ISAACLAB, KimodoActionRuntime

def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--episodes',default='5,14');p.add_argument('--server',default='http://127.0.0.1:18080');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 info=json.loads((a.dataset/'meta/info.json').read_text());client=KimodoHttpClient(a.server,30);report=[]
 for eid in map(int,a.episodes.split(',')):
  fmt=dict(episode_index=eid,episode_chunk=eid//info['chunks_size'])
  table=pq.read_table(a.dataset/info['data_path'].format(**fmt))
  def ar(k):return np.array(table[k].to_pylist())
  q=ar('observation.joint_q');dq=ar('observation.joint_dq');quat=ar('observation.root_orientation');target=ar('action.joint_q')[:,MUJOCO_TO_ISAACLAB];hand=ar('action.hand_binary');rootq=ar('action.root_q');n=len(q);t=np.arange(n)/30
  observed_hand=project_realworld_hand_closure(ar('observation.state'),info['features']['observation.state']['names'],name='replay measured hand')
  if hasattr(observed_hand,'numpy'):observed_hand=observed_hand.numpy()
  yaw=np.unwrap(R.from_quat(rootq[:,[1,2,3,0]]).as_euler('xyz')[:,2])
  changes=np.flatnonzero(np.any(np.diff(hand,axis=0)>0,axis=1))+1
  close=int(changes[0]) if len(changes) else n//3
  turn=int(np.argmax(np.abs(yaw[38:]-yaw[:-38])))
  candidates=[('approach',max(110,close-15)),('after_close',max(110,close+15)),('turn',max(110,turn))]
  candidates=[(tag,min(k,n-42)) for tag,k in candidates]
  vp=a.dataset/info['video_path'].format(video_key='observation.images.ego_view',**fmt)
  want={k for _,k in candidates};images={}
  with av.open(str(vp)) as video:
   for k,frame in enumerate(video.decode(video=0)):
    if k in want:images[k]=frame.to_ndarray(format='rgb24')
    if k>=max(want):break
  for tag,k in candidates:
   ht=np.arange(max(0,t[k]-3.4),t[k]+.00001,1/50);ht=ht[-172:]
   def interp(v,at):return np.stack([np.interp(at,t,v[:,i]) for i in range(v.shape[1])],axis=1)
   hq=interp(q,ht);hdq=interp(dq,ht);hrot=Slerp(t,R.from_quat(quat[:,[1,2,3,0]]))(ht).as_quat()[:,[3,0,1,2]]
   hist=[]
   for j in range(len(ht)):
    state,*_=state_msg_to_kimodo_state(dict(body_q=hq[j],body_dq=hdq[j],base_quat=hrot[j],state_monotonic_timestamp=ht[j]+1),initial_root_quat_wxyz=quat[0])
    hist.append(state)
   hist=np.array(hist);hh=interp(observed_hand,ht)
   client.reset();start=time.monotonic();pred=client.infer_chunk(images[k],hist[-1],hist,'teleop-banana',hh);elapsed=time.monotonic()-start
   ft=t[k]+np.arange(1,len(pred)+1)/50;mask=ft<=t[-1];pred=pred[mask];ft=ft[mask]
   gtq=interp(target,ft);gth=hand[np.minimum(np.rint(ft*30).astype(int),n-1)]
   # Rotation columns use the project's row-stacked 6D convention.
   from kimodo_sonic_bridge import rot6d_row_to_quat_wxyz
   pqrot=np.array([rot6d_row_to_quat_wxyz(row[3:9]) for row in pred])
   pyaw=np.unwrap(R.from_quat(pqrot[:,[1,2,3,0]]).as_euler('xyz')[:,2]);gyaw=np.interp(ft,t,yaw)
   row=dict(episode=eid,stage=tag,frame=k,inference_s=elapsed,joint_rmse_rad=float(np.sqrt(np.mean((pred[:,9:38]-gtq)**2))),hand_agreement=float(np.mean((pred[:,38:40]>=.5)==gth)),pred_hand_first_last=pred[[0,-1],38:40].tolist(),target_hand_first_last=gth[[0,-1]].tolist(),pred_yaw_change_deg=float(np.rad2deg(pyaw[-1]-pyaw[0])),target_yaw_change_deg=float(np.rad2deg(gyaw[-1]-gyaw[0])),prediction=pred.tolist(),target_joint=gtq.tolist())
   measured_yaw=np.unwrap(R.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
   myaw=np.interp(ft,t,measured_yaw)
   row['measured_yaw_change_deg']=float(np.rad2deg(myaw[-1]-myaw[0]))
   row['reference_measured_joint_rmse_rad']=float(np.sqrt(np.mean((gtq-interp(q[:,MUJOCO_TO_ISAACLAB],ft))**2)))
   runtime=KimodoActionRuntime();runtime.prime_joints(q[k][MUJOCO_TO_ISAACLAB]); runtime.previous_reference_quat=rootq[k].astype(np.float32); decoded=[]
   # A mid-episode probe seeds the preceding reference orientation; identity would simulate an artificial restart.
   try:
    for act in pred:decoded.append(runtime.decode(act,current_root_quat_wxyz=quat[k])['body_quat'])
    dyaw=np.unwrap(R.from_quat(np.array(decoded)[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    row['bridge_yaw_change_deg']=float(np.rad2deg(dyaw[-1]-dyaw[0]))
    row['bridge_error']=None
   except Exception as exc:row['bridge_error']=str(exc)
   report.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['prediction','target_joint']}),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(note='Teacher-forced windows from training data, reset per window; not held-out evaluation or full RTC rollout. Stage labels are heuristic, not proof of successful grasp.',results=report),indent=2))
if __name__=='__main__':main()
