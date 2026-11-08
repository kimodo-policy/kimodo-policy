#!/usr/bin/env python3
"""Download, verify and install the versioned robot-side runtime; no hardware I/O."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from install_runtime_bundle import digest

# Rebuilt by the PICO editable-source installation on every setup run.
BUILD_OUTPUTS = frozenset({
    "external/GR00T-WholeBodyControl/external_dependencies/"
    "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/"
    "xrobotoolkit_sdk.cpython-310-x86_64-linux-gnu.so",
})


def overlay_files(root: Path) -> dict[str, Path]:
    base = root / 'collection/overlay/GR00T-WholeBodyControl'
    return {'external/GR00T-WholeBodyControl/' + p.relative_to(base).as_posix(): p
            for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def verify_installed(root: Path, manifest: dict, *, after_build: bool = False) -> None:
    overrides = overlay_files(root)
    for name, record in manifest['files'].items():
        if after_build and name in BUILD_OUTPUTS:
            continue
        expected = digest(overrides[name]) if name in overrides else record['sha256']
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Installed runtime changed or is incomplete: ' + name)
    for name, source in overrides.items():
        path = root / name
        if not path.is_file() or digest(path) != digest(source):
            raise ValueError('Installed overlay differs: ' + name)


def download_archive(url: str, target: Path, expected: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    print('Downloading workstation runtime (approximately 3.81 GB)...', flush=True)
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == expected['archive_bytes']:
            break
        if offset > expected['archive_bytes']:
            temporary.unlink()
            offset = 0
        headers = {'User-Agent': 'Kimodo-Policy-workstation-setup'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and getattr(response, 'status', 200) == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected range response from runtime download.')
                with temporary.open('ab' if append else 'wb') as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if temporary.stat().st_size == expected['archive_bytes']:
                break
        except OSError:
            if attempt == 2:
                raise
        if attempt < 2:
            print('Retrying runtime download; keeping the partial file for resumption.', flush=True)
            time.sleep(2)
    if temporary.stat().st_size != expected['archive_bytes'] or digest(temporary) != expected['archive_sha256']:
        temporary.unlink()
        raise ValueError('Downloaded runtime failed size/SHA-256 verification; rerun to download again.')
    os.replace(temporary, target)


def prepare(root: Path, archive: Path | None, download: bool) -> None:
    manifest_path = root / 'manifests/workstation-runtime.json'
    manifest = json.loads(manifest_path.read_text())
    marker = root / '.state/runtime-installed.json'
    if marker.exists():
        installed = json.loads(marker.read_text())
        if installed.get('archive_sha256') != manifest['archive_sha256']:
            raise ValueError('A different runtime release is installed; use a fresh checkout.')
        verify_installed(root, manifest, after_build=True)
        print('Existing runtime and collection overlay verified; reusing them.', flush=True)
        return
    destinations = ['external/GR00T-WholeBodyControl', 'external/GMR', 'runtime/native']
    if any((root / name).exists() for name in destinations):
        raise ValueError('Unmanaged runtime directories already exist; use a fresh checkout. No files were overwritten.')
    archive = archive or root / 'downloads/workstation-runtime.tar.gz'
    if not archive.is_file():
        if not download:
            raise ValueError('Provide --download-resources or --runtime-archive /path/to/workstation-runtime.tar.gz.')
        release = json.loads((root / 'manifests/workstation-download.json').read_text())
        if release['archive_sha256'] != manifest['archive_sha256']:
            raise ValueError('Download metadata does not match the runtime manifest.')
        download_archive(release['url'], archive, manifest)
    if archive.stat().st_size != manifest['archive_bytes'] or digest(archive) != manifest['archive_sha256']:
        raise ValueError('Runtime archive does not match this code release.')
    subprocess.run([sys.executable, str(root / 'tools/install_runtime_bundle.py'), str(archive),
                    '--manifest', str(manifest_path), '--root', str(root)], check=True)
    for name, source in overlay_files(root).items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    verify_installed(root, manifest)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps({'archive_sha256': manifest['archive_sha256']}, indent=2) + '\n')
    os.replace(temporary, marker)
    print('Runtime and collection overlay installed and verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args.root.resolve(), args.archive.resolve() if args.archive else None, args.download)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
