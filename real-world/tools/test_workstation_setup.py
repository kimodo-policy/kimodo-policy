"""Offline regression checks for the optional workstation installer."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import prepare_workstation as prepare


class RuntimePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='workstation test ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'real-world'
        self.root.mkdir()
        (self.root / 'tools').mkdir()
        shutil.copy2(Path(__file__).with_name('install_runtime_bundle.py'), self.root / 'tools')
        self.archive = Path(self.temporary.name) / 'runtime.tar.gz'
        self.files = {'external/GR00T-WholeBodyControl/example.txt': b'original',
                      'external/GMR/example.txt': b'gmr', 'runtime/native/example.txt': b'native'}
        self.sdk_output = next(iter(prepare.BUILD_OUTPUTS))
        self.files[self.sdk_output] = b'original SDK binary'
        with tarfile.open(self.archive, 'w:gz') as bundle:
            for name, data in self.files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
        self.manifest = {'archive_sha256': prepare.digest(self.archive),
                         'archive_bytes': self.archive.stat().st_size,
                         'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                   for name, data in self.files.items()}}
        (self.root / 'manifests').mkdir()
        (self.root / 'manifests/workstation-runtime.json').write_text(json.dumps(self.manifest))
        overlay = self.root / 'collection/overlay/GR00T-WholeBodyControl'
        overlay.mkdir(parents=True)
        (overlay / 'example.txt').write_text('reviewed overlay')

    def test_offline_install_and_rerun_verify_overlay(self):
        prepare.prepare(self.root, self.archive, False)
        self.assertEqual((self.root / 'external/GR00T-WholeBodyControl/example.txt').read_text(), 'reviewed overlay')
        prepare.prepare(self.root, None, False)

    def test_rebuilt_sdk_does_not_prevent_rerun_but_native_changes_do(self):
        prepare.prepare(self.root, self.archive, False)
        (self.root / self.sdk_output).write_bytes(b'rebuilt on this machine')
        prepare.prepare(self.root, None, False)
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare.verify_installed(self.root, self.manifest)
        (self.root / 'runtime/native/example.txt').write_bytes(b'changed native library')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare.prepare(self.root, None, False)

    def test_existing_checkout_is_not_overwritten(self):
        path = self.root / 'external/GMR'
        path.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Unmanaged'):
            prepare.prepare(self.root, self.archive, False)
        self.assertTrue(path.is_dir())

    def test_modified_installation_is_not_silently_replaced(self):
        prepare.prepare(self.root, self.archive, False)
        (self.root / 'external/GMR/example.txt').write_text('user change')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare.prepare(self.root, None, False)
        self.assertEqual((self.root / 'external/GMR/example.txt').read_text(), 'user change')

    def test_wrong_archive_rejected_before_extraction(self):
        self.archive.write_bytes(b'wrong archive')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            prepare.prepare(self.root, self.archive, False)
        self.assertFalse((self.root / 'external').exists())

    def test_download_is_verified_before_publication(self):
        data = self.archive.read_bytes()
        target = self.root / 'downloads/runtime.tar.gz'
        with patch.object(prepare.urllib.request, 'urlopen', return_value=io.BytesIO(data)):
            prepare.download_archive('https://example.invalid/runtime', target, self.manifest)
        self.assertEqual(target.read_bytes(), data)
        self.assertFalse(target.with_suffix('.gz.part').exists())

    def test_corrupt_download_not_published(self):
        target = self.root / 'downloads/runtime.tar.gz'
        with patch.object(prepare.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'wrong')), patch.object(prepare.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'verification'):
                prepare.download_archive('https://example.invalid/runtime', target, self.manifest)
        self.assertFalse(target.exists())

    def test_partial_download_resumes_at_verified_range(self):
        data = self.archive.read_bytes()
        target = self.root / 'downloads/runtime.tar.gz'
        target.parent.mkdir()
        target.with_suffix('.gz.part').write_bytes(data[:20])
        response = io.BytesIO(data[20:])
        response.status = 206
        response.headers = {'Content-Range': f'bytes 20-{len(data)-1}/{len(data)}'}
        with patch.object(prepare.urllib.request, 'urlopen', return_value=response) as call:
            prepare.download_archive('https://example.invalid/runtime', target, self.manifest)
        self.assertEqual(call.call_args.args[0].get_header('Range'), 'bytes=20-')
        self.assertEqual(target.read_bytes(), data)


class SetupRoutingTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[2]

    def test_default_excludes_workstation(self):
        result = subprocess.run(['bash', str(self.repo / 'setup.sh'), '--dry-run'], capture_output=True, text=True, check=True)
        self.assertIn('unitree_sim_env', result.stdout)
        self.assertNotIn('[workstation]', result.stdout)

    def test_workstation_plan_does_not_include_simulators(self):
        result = subprocess.run(['bash', str(self.repo / 'setup.sh'), '--only', 'workstation', '--dry-run'], capture_output=True, text=True, check=True)
        self.assertIn('[workstation]', result.stdout)
        self.assertNotIn('unitree_sim_env', result.stdout)
        self.assertNotIn('SIMPLE/.venv', result.stdout)

    def test_installation_does_not_redirect_into_user_environment(self):
        with tempfile.TemporaryDirectory(prefix='workstation config ') as directory:
            root = Path(directory)
            shutil.copy2(self.repo / 'real-world/env.sh', root / 'env.sh')
            (root / 'local.env').write_text('export TELEOP_PYTHON=/unrelated/environment/python\n')
            (root / '.state').mkdir()
            (root / '.state/workstation.env').write_text('export TELEOP_PYTHON=/old/checkout/python\n')
            result = subprocess.run(['bash', '-c', 'source "$1"; printf "%s" "$TELEOP_PYTHON"',
                                     'bash', str(root / 'env.sh')],
                                    env=dict(os.environ, KIMODO_WORKSTATION_INSTALLING='1', TELEOP_PYTHON='/managed/environment/python'),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, '/managed/environment/python')


if __name__ == '__main__':
    unittest.main()
