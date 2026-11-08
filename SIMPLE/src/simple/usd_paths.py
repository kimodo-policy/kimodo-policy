"""Resolve HSSD exporter paths without creating machine-wide asset symlinks."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def prepare_hssd_usd(usd_path: str) -> str:
    """Create cached USD layers with local references, leaving source files intact.

    Called after Isaac has initialized USD. Each scene has its own cache, so
    concurrent rooms cannot redirect one another's props/textures. No root
    permissions or directories such as /ai and /data/local-data are needed.
    """
    import fcntl
    from pxr import Sdf, UsdUtils

    root = Path(usd_path).resolve()
    if not root.is_file():
        raise FileNotFoundError(root)
    scene = root.parent
    layers = sorted(p for p in scene.rglob("*")
                    if p.suffix.lower() in {".usd", ".usda", ".usdc"} and p.is_file())
    signature = "hssd-local-v3\n" + str(scene) + "\n" + "\n".join(
        f"{p.relative_to(scene)}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in layers)
    key = hashlib.sha256(signature.encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kimodo-policy/usd" / key
    cache.mkdir(parents=True, exist_ok=True)
    destinations = {p.resolve(): cache / p.relative_to(scene) for p in layers}
    result = destinations[root]
    with (cache / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (cache / ".complete").is_file() and all(p.is_file() for p in destinations.values()):
            return str(result)
        for source, destination in destinations.items():
            original = Sdf.Layer.FindOrOpen(str(source))
            if original is None:
                raise RuntimeError(f"Cannot open HSSD USD layer: {source}")
            layer = Sdf.Layer.CreateAnonymous()
            layer.TransferContent(original)

            def relocate(asset):
                if not asset or "://" in asset:
                    return asset
                path = Path(asset)
                if not path.is_absolute():
                    path = source.parent / path
                # Some exporters spell the same old /tmp path as a long
                # relative traversal (../../../../props), not an absolute path.
                parts = Path(asset).parts
                for folder in ("props", "textures"):
                    if folder in parts:
                        candidate = scene.joinpath(*parts[parts.index(folder):])
                        if not candidate.is_file() and candidate.parent.is_dir():
                            matches = [p for p in candidate.parent.iterdir()
                                       if p.name.lower() == candidate.name.lower() and p.is_file()]
                            if len(matches) == 1:
                                candidate = matches[0]
                        if candidate.is_file():
                            path = candidate
                            break
                path = path.resolve()
                if path in destinations:
                    return str(destinations[path])
                # Preserve MDL module names and unresolved resolver identifiers.
                return str(path) if path.exists() else asset

            UsdUtils.ModifyAssetPaths(layer, relocate)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.stem + f".{os.getpid()}.tmp" + destination.suffix)
            if not layer.Export(str(temporary)):
                raise RuntimeError(f"Cannot export portable HSSD layer: {destination}")
            os.replace(temporary, destination)
        (cache / ".complete").write_text(signature)
    return str(result)
