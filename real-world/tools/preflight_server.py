#!/usr/bin/env python3
"""No model construction/CUDA allocation: validate checkpoint, base assets and text cache."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--checkpoint');a=p.parse_args()
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'deploy'))
import deploy_cli as d
c=d.load_config();t=d.require_task(c,a.task);ck=d.resolve_checkpoint(a.task,a.checkpoint or t['checkpoint'],d.dataset_name(c,a.task));conf=json.loads((ck/'config.json').read_text());project=d.PROJECT_ROOT
for rel in ['train.py','model/kimodo_policy.py','motion/g1_reference.py','skeleton/assets/g1skel34/xml/g1.xml','checkpoints/Kimodo-G1-RP-v1/config.yaml','checkpoints/Kimodo-G1-RP-v1/model.safetensors']:
 if not (project/rel).is_file():raise SystemExit('Missing model asset/source: '+rel)
vision=conf['model'].get('dinov3_checkpoint') or str(project/'checkpoints'/conf['model']['dinov3_model_name'])
if not Path(vision).is_absolute():vision=str(project/vision)
if not Path(vision).exists():raise SystemExit('Missing vision model: '+vision)
from real_world_server import _load_text_embedding_cache,_resolve_cached_task
cache=Path(os.environ.get('KIMODO_TEXT_CACHE',project/'data/cache/RealWorld'))
embeddings,instructions,aliases=_load_text_embedding_cache(cache)
key=_resolve_cached_task(d.dataset_name(c,a.task),aliases)
d.get_pose(c,a.task)
print(json.dumps({'checkpoint':str(ck),'dataset':d.dataset_name(c,a.task),'text_task_id':key,'instruction':instructions[key],'embedding_shape':list(embeddings[key].shape),'status':'PASS metadata/assets/text; GPU/model not loaded'},indent=2))
