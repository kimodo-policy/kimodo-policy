"""Summarize one deployment diagnostics directory without robot access."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def yaw(q):return R.from_quat(np.asarray(q)[[1,2,3,0]]).as_euler('xyz')[2]
def stats(a):
 a=np.asarray(a,dtype=float)
 return None if not a.size else dict(count=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
 rows=[json.loads(s) for s in (a.run/'events.jsonl').read_text().splitlines() if s.strip()]
 results=[r for r in rows if r['event']=='result'];accepted=[r for r in results if r['accepted']]
 control=[r for r in rows if r['event']=='control' and r.get('state_valid') and r.get('reference')]
 yawerr=[];joint=[];hand=[];target=[]
 for r in control:
  ref=r['reference'];initial=r.get('initial_root_quat')
  if initial:
   err=yaw(ref['body_quat'])-(yaw(r['measured_root_quat'])-yaw(initial))
   yawerr.append(abs(np.rad2deg(np.arctan2(np.sin(err),np.cos(err)))))
  joint.append(float(np.sqrt(np.mean((np.array(ref['joint_pos'])-r['measured_q'])**2))))
  if r.get('measured_hand') is not None:
   hand.append(r['measured_hand']);target.append(ref.get('hand_closure',ref.get('hand_binary')))
 report=dict(events=dict(Counter(r['event'] for r in rows)),closed_cleanly=rows[-1]['event']=='recorder_end',recorder_dropped=rows[-1].get('dropped'),accepted=len(accepted),rejected=dict(Counter(str(r['details'].get('reason')) for r in results if not r['accepted'])),request_elapsed_s=stats([r['elapsed_s'] for r in results if r.get('elapsed_s') is not None]),accepted_age_s=stats([r['details']['age_s'] for r in accepted]),holds=dict(Counter(r['reason'] for r in rows if r['event']=='hold')),replans=dict(Counter(r['reason'] for r in rows if r['event']=='invalidate')),control_sample_gap_s=stats(np.diff([r['monotonic'] for r in control])),empty_queue_fraction=None if not control else float(np.mean([r['queue']==0 for r in control])),reference_measured_joint_rmse_rad=stats(joint),reference_measured_yaw_abs_error_deg=stats(yawerr))
 report['boundary_joint_step_rad']=stats([r['details']['boundary_joint_step_rad'] for r in accepted if 'boundary_joint_step_rad' in r['details']])
 report['stage_timing_s']={key:stats([r['timing'][key] for r in results if r.get('timing') and key in r['timing']]) for key in ('client_encode_s','read_decode_s','preprocess_s','model_and_transfer_s','postprocess_s','runtime_total_s','handler_total_s')}
 if target:
  report['reference_hand_switches']=(np.diff(np.array(target)>=.5,axis=0)!=0).sum(axis=0).tolist()
  report['measured_hand_switches']=(np.diff(np.array(hand)>=.5,axis=0)!=0).sum(axis=0).tolist()
 report['note']='10Hz samples, not all 50Hz cycles. Tracking errors in no-publish/dry-run do not measure robot tracking performance. No automatic pass/fail or causal attribution.'
 (a.run/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
