#!/usr/bin/env python3
"""Verify and extract a trusted, maintainer-produced runtime bundle into a fresh clone."""
import argparse,hashlib,json,tarfile,os,shutil,tempfile
from pathlib import Path, PurePosixPath

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
 manifest=json.loads(a.manifest.read_text())
 if digest(a.archive)!=manifest['archive_sha256']:raise ValueError('Archive hash mismatch')
 destinations=['external/GR00T-WholeBodyControl','external/GMR','runtime/native']
 root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 for d in destinations:
  if (root/d).exists():raise SystemExit(f'Refusing to overwrite {d}; use a fresh runtime directory')
 with tempfile.TemporaryDirectory(prefix='.runtime-stage-',dir=root) as td:
  stage=Path(td)
  with tarfile.open(a.archive,'r:gz') as tar:
   members=tar.getmembers()
   for m in members:
    name=PurePosixPath(m.name)
    if name.is_absolute() or '..' in name.parts or str(name) not in manifest['files']:raise ValueError('Unexpected archive member')
    if not (m.isfile() or m.islnk()):raise ValueError('Only regular files and internal hardlinks are supported')
    if m.islnk():
     link=PurePosixPath(m.linkname)
     if link.is_absolute() or '..' in link.parts or str(link) not in manifest['files']:raise ValueError('Invalid hardlink')
   # Every member and link was confined to the staging tree above.
   tar.extractall(stage,members=members)
  for rel,expected in manifest['files'].items():
   f=stage/rel
   if not f.is_file() or digest(f)!=expected['sha256']:raise ValueError('File hash mismatch: '+rel)
  for d in destinations:
   target=root/d;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(stage/d),str(target))
 print('Runtime verified and extracted. Apply collection overlay and build SONIC next.')
if __name__=='__main__':main()
