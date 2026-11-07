#!/usr/bin/env python3
"""Check the selected interpreter against observed deployment-critical package versions."""
import argparse,importlib.metadata as meta,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile',choices=['teleop','collection','server']);p.add_argument('--all-packages',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1];expected=json.loads((r/'environments'/f'{a.profile}.observed.json').read_text())
core={'numpy','scipy','torch','torchvision','pyzmq','opencv-python','opencv-python-headless','av','pyarrow','pin','xrobotoolkit_sdk','lerobot','accelerate','transformers','omegaconf'}
errors=[]
for n,v in expected.items():
 if not a.all_packages and n not in core:continue
 try:actual=meta.version(n)
 except meta.PackageNotFoundError:actual='MISSING'
 if actual!=v:errors.append({'package':n,'expected':v,'actual':actual})
if sys.version_info[:2]!=(3,10):errors.append({'python_expected':'3.10','actual':sys.version.split()[0]})
print(json.dumps({'profile':a.profile,'python':sys.version.split()[0],'differences':errors},indent=2))
raise SystemExit(bool(errors))
