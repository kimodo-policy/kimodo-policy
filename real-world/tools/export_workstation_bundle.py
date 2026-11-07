#!/usr/bin/env python3
"""Maintainer: export known-working non-conda dependencies; never launches hardware."""
import argparse,hashlib,json,tarfile,time
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=a.source_root.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise SystemExit('Refusing to overwrite existing archive')
 mappings=[(r/'controlnet_v1.2/real-world/GR00T-WholeBodyControl','external/GR00T-WholeBodyControl'),(r/'external/GMR-source','external/GMR'),(r/'.native','runtime/native')]
 skip={'.git','__pycache__','.cache','build','dist','logs','outputs','target','.pytest_cache'}
 records={}
 with tarfile.open(a.output,'w:gz',compresslevel=1,dereference=True) as tar:
  for src,dst in mappings:
   if not src.is_dir():raise RuntimeError(f'Missing source: {src.name}')
   for f in sorted(src.rglob('*')):
    rel=f.relative_to(src)
    if any(x in skip or x.endswith('.egg-info') for x in rel.parts):continue
    if not f.is_file() or f.suffix in ('.pyc','.trt','.engine','.mp4','.parquet','.log'):continue
    # Exclude training checkpoints. SONIC ONNX/native libraries and reference assets are retained.
    if f.suffix in ('.pt','.pth','.ckpt'):continue
    name=str(Path(dst)/rel)
    records[name]={'bytes':f.stat().st_size,'sha256':digest(f)}
    tar.add(f,arcname=name,recursive=False)
 meta={'format':1,'kind':'workstation-runtime','archive_sha256':digest(a.output),'archive_bytes':a.output.stat().st_size,'files':records}
 a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({'archive':a.output.name,'bytes':meta['archive_bytes'],'sha256':meta['archive_sha256'],'files':len(records)}),flush=True)
if __name__=='__main__':main()
