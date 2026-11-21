"""Voice store layout and lifecycle: import, list, get, delete, names, duplicates, atomicity and broken folders.

Each test points S2S_DATA_DIR at its own temp folder, so the store and its incoming folder are hermetic. Nothing reaches the network.
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import rvc_store
from tests_seedvc.rvc_fixtures import SaveCheckpoint, TinyConfig

INDEX_BYTES = b"IwFl" + bytes(64)


class _Hostile:
    """Pickles to a call of os.system. A loader that runs pickles would write the marker file."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.system, (f'echo x > "{self.marker}"',))


class _StoreCase(unittest.TestCase):
    """Each test gets its own data folder, so the voice store and its incoming folder are hermetic."""

    @classmethod
    def setUpClass(cls):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(folder.cleanup)
        cls.fixture = SaveCheckpoint(pathlib.Path(folder.name) / "tiny_v2.pth")
        cls.fixture_bytes = cls.fixture.read_bytes()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"S2S_DATA_DIR": str(self.root / "data")})
        env.start()
        self.addCleanup(env.stop)

    def _AssertNothingStored(self, name):
        store = rvc_store.StoreDir()
        if store.is_dir():
            self.assertEqual([p.name for p in store.iterdir() if p.name.startswith(".tmp-")], [])
        self.assertFalse((store / name).exists())
        self.assertEqual(list(rvc_store.IncomingDir().iterdir()), [])


class StoreLifecycleTests(_StoreCase):
    def test_import_writes_a_complete_folder_and_reports_its_facts(self):
        result = rvc_store.ImportPth(self.fixture)
        folder = rvc_store.StoreDir() / "tiny_v2"
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["meta.json", "model.pth"])
        self.assertEqual(result["name"], "tiny_v2")
        self.assertEqual(result["version"], "v2")
        self.assertEqual(result["sr"], 40000)
        self.assertIs(result["f0"], True)
        self.assertIs(result["has_index"], False)
        self.assertEqual(result["source"], "upload")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(len(meta["model_sha256"]), 64)
        self.assertEqual(list(rvc_store.IncomingDir().iterdir()), [])

    def test_get_and_list_see_the_voice_and_delete_removes_it(self):
        rvc_store.ImportPth(self.fixture)
        voice = rvc_store.Get("tiny_v2")
        self.assertTrue(os.path.isabs(voice["model_path"]))
        self.assertTrue(pathlib.Path(voice["model_path"]).is_file())
        self.assertIsNone(voice["index_path"])
        listed = rvc_store.List()
        self.assertEqual([v["name"] for v in listed], ["tiny_v2"])
        self.assertNotIn("model_path", listed[0])
        rvc_store.Delete("tiny_v2")
        self.assertFalse((rvc_store.StoreDir() / "tiny_v2").exists())
        self.assertEqual(rvc_store.List(), [])
        with self.assertRaisesRegex(rvc_store.RvcImportError, "no imported voice named tiny_v2"):
            rvc_store.Get("tiny_v2")

    def test_a_valid_index_is_stored_beside_the_model(self):
        index = self.root / "tiny_v2.index"
        index.write_bytes(INDEX_BYTES)
        result = rvc_store.ImportPth(self.fixture, index_path=index)
        self.assertTrue(result["has_index"])
        folder = rvc_store.StoreDir() / "tiny_v2"
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["meta.json", "model.index", "model.pth"])
        self.assertEqual((folder / "model.index").read_bytes(), INDEX_BYTES)
        self.assertEqual(rvc_store.Get("tiny_v2")["index_path"], str((folder / "model.index").resolve()))
        self.assertTrue(index.is_file())
        self.assertTrue(self.fixture.is_file())

    def test_a_path_like_name_stays_inside_the_store(self):
        result = rvc_store.ImportPth(self.fixture, name="../../x")
        name = result["name"]
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn("..", name)
        folder = (rvc_store.StoreDir() / name).resolve()
        self.assertEqual(folder.parent, rvc_store.StoreDir().resolve())
        self.assertTrue(folder.is_dir())

    def test_other_characters_become_underscores(self):
        self.assertEqual(rvc_store.ImportPth(self.fixture, name="my voice?!")["name"], "my voice__")

    def test_a_windows_reserved_name_is_refused(self):
        with self.assertRaisesRegex(rvc_store.RvcImportError, "reserved by Windows"):
            rvc_store.ImportPth(self.fixture, name="CON")

    def test_an_empty_name_falls_back_to_the_pth_stem(self):
        self.assertEqual(rvc_store.ImportPth(self.fixture, name="")["name"], "tiny_v2")

    def test_a_long_name_is_truncated_to_64_characters(self):
        result = rvc_store.ImportPth(self.fixture, name="a" * 70)
        self.assertEqual(result["name"], "a" * 64)
        self.assertTrue((rvc_store.StoreDir() / ("a" * 64)).is_dir())

    def test_a_duplicate_name_is_refused_without_overwrite(self):
        rvc_store.ImportPth(self.fixture, name="Dup")
        with self.assertRaisesRegex(rvc_store.RvcImportError, "already imported"):
            rvc_store.ImportPth(self.fixture, name="Dup")
        self.assertEqual([v["name"] for v in rvc_store.List()], ["Dup"])

    def test_a_hostile_import_leaves_no_temp_folder_or_voice(self):
        marker = self.root / "marker.txt"
        hostile = self.root / "hostile.pth"
        torch.save({"config": TinyConfig(40000), "weight": _Hostile(marker), "version": "v2", "f0": 1}, hostile)
        with self.assertRaises(rvc_store.RvcImportError):
            rvc_store.ImportPth(hostile, name="partial")
        self.assertFalse(marker.exists())
        self._AssertNothingStored("partial")

    def test_a_failed_final_rename_leaves_no_temp_folder_or_voice(self):
        real_replace = os.replace

        def FailFinalRename(source, target):
            if pathlib.Path(target).name == "partial":
                raise OSError("disk full")
            return real_replace(source, target)

        with mock.patch.object(rvc_store.os, "replace", side_effect=FailFinalRename):
            with self.assertRaisesRegex(rvc_store.RvcImportError, "could not save the voice on disk"):
                rvc_store.ImportPth(self.fixture, name="partial")
        self._AssertNothingStored("partial")

    def test_a_zip_with_a_model_index_and_readme_imports_with_the_zip_source(self):
        zip_path = self.root / "voice.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("model.pth", self.fixture_bytes)
            archive.writestr("model.index", INDEX_BYTES)
            archive.writestr("readme.txt", "hello")
        result = rvc_store.ImportZip(zip_path)
        self.assertEqual(result["source"], "zip")
        self.assertTrue(result["has_index"])
        self.assertEqual(result["name"], "model")
        self.assertEqual(sorted(p.name for p in (rvc_store.StoreDir() / "model").iterdir()), ["meta.json", "model.index", "model.pth"])
        self.assertTrue(zip_path.is_file())

    def test_list_skips_broken_folders_and_keeps_the_good_ones(self):
        rvc_store.ImportPth(self.fixture, name="good")
        store = rvc_store.StoreDir()
        (store / "garbage").mkdir()
        (store / "garbage" / "meta.json").write_text("{not json", encoding="utf-8")
        (store / "empty").mkdir()
        with self.assertLogs("seedvc.rvc_store", level="WARNING") as logs:
            voices = rvc_store.List()
        self.assertEqual([v["name"] for v in voices], ["good"])
        self.assertEqual(len(logs.output), 2)

    def test_delete_refuses_a_traversal_name_and_an_unknown_name(self):
        rvc_store.ImportPth(self.fixture, name="keep")
        for name in ("..", "../keep", "missing"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(rvc_store.RvcImportError, "no imported voice named"):
                    rvc_store.Delete(name)
        self.assertEqual([v["name"] for v in rvc_store.List()], ["keep"])


if __name__ == "__main__":
    unittest.main()
