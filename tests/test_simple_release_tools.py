import importlib.util
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from utils.prepare_simple import extract_archive


class ArchiveTest(unittest.TestCase):
    def test_training_root_and_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(root / "data.tar.gz", "w:gz") as archive:
                entry = tarfile.TarInfo("simple/task/meta/info.json")
                entry.size = 2
                archive.addfile(entry, io.BytesIO(b"{}"))
            extract_archive(root / "data.tar.gz", root / "training", "simple/")
            self.assertEqual((root / "training/task/meta/info.json").read_text(), "{}")

    def test_zip_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "data.zip", "w") as archive:
                archive.writestr("../outside", "bad")
            with self.assertRaises(ValueError):
                extract_archive(root / "data.zip", root / "data")
            self.assertFalse((root / "outside").exists())

    def test_tar_symlink_and_existing_directory_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with tarfile.open(root / "link.tar", "w") as archive:
                entry = tarfile.TarInfo("link")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "../outside"
                archive.addfile(entry)
            with self.assertRaises(ValueError):
                extract_archive(root / "link.tar", root / "data")
            (root / "outside").mkdir()
            (root / "data/link").symlink_to(root / "outside", target_is_directory=True)
            with zipfile.ZipFile(root / "data.zip", "w") as archive:
                archive.writestr("link/file", "bad")
            with self.assertRaises(ValueError):
                extract_archive(root / "data.zip", root / "data")
            self.assertFalse((root / "outside/file").exists())


try:
    from pxr import Sdf, Usd, UsdUtils
except ImportError:
    Sdf = None


@unittest.skipIf(Sdf is None, "Run with USD/Isaac available to verify USD references")
class PortableUsdTest(unittest.TestCase):
    def test_concurrent_scenes_resolve_their_own_assets_without_modifying_sources(self):
        path = Path(__file__).resolve().parents[1] / "SIMPLE/src/simple/usd_paths.py"
        spec = importlib.util.spec_from_file_location("release_usd_paths", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"XDG_CACHE_HOME": tmp + "/cache"}):
            roots = []
            for name in ("a", "b"):
                scene = Path(tmp) / name
                (scene / "props").mkdir(parents=True)
                (scene / "textures").mkdir()
                (scene / "textures/color.png").write_bytes(b"image")
                stage = Usd.Stage.CreateNew(str(scene / "props/Mesh.usda"))
                prim = stage.DefinePrim("/Mesh", "Xform")
                stage.SetDefaultPrim(prim)
                prim.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("/tmp/exporter/textures/color.png"))
                stage.GetRootLayer().Save()
                root = scene / "scene.usda"
                reference = "/ai/props/mesh.usda" if name == "a" else "./../../../../../../props/mesh.usda"
                root.write_text('#usda 1.0\ndef Xform "Room" (references = @' + reference + '@) {}\n')
                roots.append(root)
            before = [p.read_bytes() for p in roots]
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(module.prepare_hssd_usd, map(str, roots * 2)))
            self.assertEqual(results[:2], results[2:])
            self.assertNotEqual(results[0], results[1])
            for source, result in zip(roots, results):
                stage = Usd.Stage.Open(result)
                texture = stage.GetPrimAtPath("/Room").GetAttribute("texture").Get()
                self.assertEqual(Path(texture.path), source.parent / "textures/color.png")
                _, _, unresolved = UsdUtils.ComputeAllDependencies(result)
                self.assertEqual(unresolved, [])
            self.assertEqual(before, [p.read_bytes() for p in roots])


if __name__ == "__main__":
    unittest.main()
