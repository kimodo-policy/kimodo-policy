#!/usr/bin/env python3
"""Download the versioned SIMPLE release resources; no simulator is required."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPO = "YunhengWang/kimodo-policy"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_text_cache(release):
    """Check the small task embeddings bundled with the Git checkout, offline."""
    for relative, entry in release["text_cache"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise RuntimeError(
                f"Bundled text embedding is missing or modified: {relative}. "
                "Restore this file from the matching Git checkout."
            )
    print(f"Verified {len(release['text_cache'])} bundled SIMPLE text embeddings (no download).")


def extract_archive(archive, destination, strip_prefix=""):
    """Extract only regular files/directories, rejecting links and path escapes."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def target(name):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive member: {name}")
        if strip_prefix:
            if not name.startswith(strip_prefix):
                raise ValueError(f"Unexpected archive root: {name}")
            name = name[len(strip_prefix):]
        out = destination / name
        if not out.resolve().is_relative_to(destination):
            raise ValueError(f"Archive path escapes destination: {name}")
        return out

    def write(out, stream):
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as temp:
            temporary = Path(temp.name)
            try:
                shutil.copyfileobj(stream, temp)
                temp.close()
                os.replace(temporary, out)
            finally:
                temporary.unlink(missing_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                out = target(entry.filename)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Archive symlink is not supported: {entry.filename}")
                if entry.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    with bundle.open(entry) as stream:
                        write(out, stream)
    else:
        with tarfile.open(archive) as bundle:
            for entry in bundle:
                out = target(entry.name)
                if entry.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    with bundle.extractfile(entry) as stream:
                        write(out, stream)
                else:
                    raise ValueError(f"Archive special file is not supported: {entry.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true", help="Download robots, objects, rooms, materials and fixed evaluation episodes")
    parser.add_argument("--text-cache", action="store_true", help="Verify the task embeddings bundled in Git (no download)")
    parser.add_argument("--training-task", metavar="TASK", help="Download one frozen training task (or all)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SIMPLE/data")
    parser.add_argument("--training-root", type=Path, default=ROOT / "datasets/Simple")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/simple-release")
    args = parser.parse_args()
    if not (args.assets or args.text_cache or args.training_task):
        parser.error("Select --assets, --text-cache and/or --training-task TASK")
    release = json.loads(Path(__file__).with_name("simple_release_manifest.json").read_text())

    if args.text_cache:
        verify_text_cache(release)

    def download(path, entry):
        from huggingface_hub import hf_hub_download

        cached = args.cache_dir / release["revision"] / path
        if not cached.is_file() or digest(cached) != entry["sha256"]:
            cached = Path(hf_hub_download(
                REPO, path, repo_type="dataset", revision=release["revision"],
                local_dir=args.cache_dir / release["revision"],
                headers={"Accept-Encoding": "identity"},
                force_download=cached.exists(),
            ))
        if digest(cached) != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}; retry the download")
        return cached

    if args.assets:
        for path, entry in release["assets"].items():
            print(f"Preparing {path}", flush=True)
            extract_archive(download(path, entry), args.data_dir)
        print(f"SIMPLE evaluation resources: {args.data_dir.resolve()}")
    if args.training_task:
        tasks = release["training"]
        if args.training_task != "all" and args.training_task not in tasks:
            parser.error(f"Unreleased training task: {args.training_task}. Available: {', '.join(tasks)}")
        for task, entry in tasks.items():
            if args.training_task not in ("all", task):
                continue
            print(f"Preparing training data: {task}", flush=True)
            extract_archive(download(entry["path"], entry), args.training_root, "simple/")
        print(f"Set KIMODO_SIMPLE_ROOT={args.training_root.resolve()}")


if __name__ == "__main__":
    main()
