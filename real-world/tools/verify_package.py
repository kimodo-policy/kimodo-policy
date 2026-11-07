#!/usr/bin/env python3
"""Static packaging checks; never contacts a server or robot."""
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
counts = {'python': 0, 'shell': 0, 'json': 0, 'links': 0, 'tasks': 0}
for p in ROOT.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:
        continue
    rel = str(p.relative_to(ROOT))
    if p.suffix in (".pt", ".gz", ".zip", ".safetensors"):
        continue
    text = p.read_text()
    try:
        if p.suffix == '.py':
            ast.parse(text, filename=rel)
            counts['python'] += 1
        if p.suffix == '.sh' or text.startswith('#!/usr/bin/env bash'):
            subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
            counts['shell'] += 1
        if p.suffix == '.json':
            json.loads(text)
            counts['json'] += 1
        if p.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' not in target and not target.startswith('#'):
                    assert (p.parent / target.split('#')[0]).exists(), f'broken link: {target}'
                    counts['links'] += 1
        if rel != 'tools/verify_package.py':
            # Reject private machine paths and private-network addresses.
            assert not re.search(r'/(?:home|Users)/[\w.-]+|/data/local-data/|/ai/Yichi|\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b', text.replace('10.13.0.35', 'TENSORRT_VERSION').replace('10.3.7.77', 'CURAND_VERSION')), 'private path/address'
            assert not re.search(r'(?i)(?:api[_-]?key|password|passwd)\s*[:=]\s*[\"\'][^\"\']{5,}[\"\']', text), 'possible embedded credential'
    except Exception as exc:
        errors.append(f'{rel}: {exc}')

spec = importlib.util.spec_from_file_location('packaged_deploy_cli', ROOT/'deploy/deploy_cli.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = json.loads((ROOT/'deploy/deploy_config.example.json').read_text())
for task in config['tasks']:
    try:
        module.get_pose(config, task)
        counts['tasks'] += 1
    except Exception as exc:
        errors.append(f'task {task}: {exc}')
manifest = ROOT/'PACKAGE_MANIFEST.json'
if manifest.exists():
    for rel, expected in json.loads(manifest.read_text()).items():
        p = ROOT/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(f'hash mismatch: {rel}')
print(json.dumps({'checks': counts, 'errors': errors}, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
