"""Offline tests for relocatable paths and archive confinement; no robot/network."""
import hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 def test_env_preserves_explicit_roots(self):
  with tempfile.TemporaryDirectory(prefix='kp paths ') as td:
   env=os.environ.copy();env.update(KIMODO_PROJECT_ROOT=td+'/model',WBC_ROOT=td+'/sonic',TELEOP_PYTHON=td+'/teleop/bin/python')
   p=subprocess.run(['bash','-c','source "$1/env.sh"; printf "%s\\n" "$KIMODO_PROJECT_ROOT" "$WBC_ROOT" "$TELEOP_PYTHON"','check',str(ROOT)],env=env,text=True,capture_output=True,check=True)
   self.assertEqual(p.stdout.splitlines(),[env[k] for k in ['KIMODO_PROJECT_ROOT','WBC_ROOT','TELEOP_PYTHON']])
 def bundle(self,d,bad=False):
  arc=d/'runtime.tar.gz';records={}
  names=['external/GR00T-WholeBodyControl/a','external/GMR/a','runtime/native/a']
  if bad:names.append('../escape')
  with tarfile.open(arc,'w:gz') as t:
   for name in names:
    content=b'test';info=tarfile.TarInfo(name);info.size=len(content);t.addfile(info,io.BytesIO(content));records[name]={'sha256':hashlib.sha256(content).hexdigest()}
  manifest=d/'manifest.json';manifest.write_text(json.dumps({'archive_sha256':hashlib.sha256(arc.read_bytes()).hexdigest(),'files':records}))
  return [sys.executable,str(ROOT/'tools/install_runtime_bundle.py'),str(arc),'--manifest',str(manifest),'--root',str(d/'dest')]
 def test_bundle_verified_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);cmd=self.bundle(d);p=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
   self.assertEqual((d/'dest/runtime/native/a').read_bytes(),b'test')
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
 def test_bundle_rejects_path_traversal(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);p=subprocess.run(self.bundle(d,True),capture_output=True);self.assertNotEqual(p.returncode,0);self.assertFalse((d/'escape').exists())
 def test_bundle_rejects_wrong_digest(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);cmd=self.bundle(d);m=d/'manifest.json';j=json.loads(m.read_text());j['archive_sha256']='0'*64;m.write_text(json.dumps(j))
   self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
 def test_tunnel_and_launcher_share_port(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);config=json.loads((ROOT/'deploy/deploy_config.example.json').read_text())
   config['server']['port']=18123
   cfg=d/'config.json';cfg.write_text(json.dumps(config))
   ssh=d/'ssh';ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');ssh.chmod(0o755)
   env=os.environ.copy();env.pop('KIMODO_PORT',None)
   env.update(KIMODO_DEPLOY_CONFIG=str(cfg),KIMODO_SERVER_SSH='test-host',PATH=str(d)+os.pathsep+env['PATH'])
   for override,expected in [(None,'18123'),('18124','18124')]:
    if override:env['KIMODO_PORT']=override
    result=subprocess.run(['bash',str(ROOT/'deploy/tunnel.sh')],env=env,text=True,capture_output=True)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn('127.0.0.1:'+expected+':127.0.0.1:'+expected,result.stdout)
   env['KIMODO_PORT']='70000'
   result=subprocess.run(['bash',str(ROOT/'deploy/tunnel.sh')],env=env,text=True,capture_output=True)
   self.assertNotEqual(result.returncode,0)
   self.assertNotIn('test-host',result.stdout)
 def test_source_uses_environment_roots(self):
  for filename in ['deploy_cli.py','real_world_server.py','run_kimodo_sonic.py']:
   self.assertIn('KIMODO_PROJECT_ROOT',(ROOT/'deploy'/filename).read_text())
  self.assertIn('os.environ.get("WBC_ROOT"',(ROOT/'deploy/run_kimodo_sonic.py').read_text())
if __name__=='__main__':unittest.main()
