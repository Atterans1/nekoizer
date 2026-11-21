"""Endpoint tests for the RVC routes in ui/server.py, against FakePipeline and a hermetic S2S_DATA_DIR."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from engine import rvc_store  # noqa: E402
from tests_seedvc.rvc_fixtures import SaveCheckpoint  # noqa: E402
from ui.fake_pipeline import FakePipeline  # noqa: E402
from ui.server import CreateApp  # noqa: E402

MODEL_SENTENCE = "send one .zip, or one .pth with an optional .index"
WEIGHTS_MISSING = {"state": "missing", "file": None, "done": 0, "total": 0, "message": None}
MiB = 1024 * 1024
FAKE_INDEX = b"IwF" + bytes(64)  # the store accepts only the FAISS IVF magic


class RvcServerTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"S2S_DATA_DIR": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.fake = FakePipeline()
        self.client = TestClient(CreateApp(self.fake), base_url="http://127.0.0.1", headers={"Host": "127.0.0.1"})
        self.addCleanup(self.client.close)

    def PthBytes(self) -> bytes:
        return SaveCheckpoint(self.root / "voice.pth").read_bytes()

    def ZipBytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("model.pth", self.PthBytes())
            archive.writestr("model.index", FAKE_INDEX)
        return buf.getvalue()

    def ImportPth(self, name: str) -> dict:
        return rvc_store.ImportPth(SaveCheckpoint(self.root / f"{name}.pth"), name=name)

    def UploadZip(self, name: str | None = "Zip Voice", slot: int = 0):
        data = {"name": name} if name is not None else {}
        return self.client.post(f"/api/slot/{slot}/rvc/import",
                                files={"zip": ("voice.zip", self.ZipBytes(), "application/zip")}, data=data)

    def Use(self, slot: int, name: str):
        return self.client.post(f"/api/slot/{slot}/rvc/use", json={"name": name})

    def test_list_is_empty_before_any_import(self):
        r = self.client.get("/api/rvc/models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"models": [], "weights": WEIGHTS_MISSING})

    def test_zip_import_loads_the_slot_and_records_the_assignment(self):
        r = self.UploadZip()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIs(body["ok"], True)
        self.assertEqual(body["slot"]["kind"], "rvc")
        self.assertEqual(body["slot"]["model"], "Zip Voice")
        self.assertEqual(body["model"]["name"], "Zip Voice")
        self.assertEqual(body["model"]["slots"], [0])
        self.assertIn(("SetSlotFromRvc", 0, "Zip Voice"), self.fake.calls)

    def test_pth_with_index_import_loads_the_slot(self):
        r = self.client.post("/api/slot/1/rvc/import",
                             files={"pth": ("voice.pth", self.PthBytes(), "application/octet-stream"),
                                    "index": ("voice.index", FAKE_INDEX, "application/octet-stream")},
                             data={"name": "Pth Voice"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slot"]["kind"], "rvc")
        self.assertEqual(r.json()["slot"]["idx"], 1)
        self.assertIn(("SetSlotFromRvc", 1, "Pth Voice"), self.fake.calls)

    def test_pth_only_import_names_the_model_from_the_file(self):
        r = self.client.post("/api/slot/2/rvc/import",
                             files={"pth": ("Lily.pth", self.PthBytes(), "application/octet-stream")})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["model"]["name"])
        self.assertEqual(r.json()["slot"]["model"], r.json()["model"]["name"])

    def test_use_assigns_an_imported_voice(self):
        self.ImportPth("Use Voice")
        r = self.Use(2, "Use Voice")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slot"]["kind"], "rvc")
        self.assertEqual(r.json()["slot"]["idx"], 2)
        self.assertEqual(r.json()["model"]["slots"], [2])
        self.assertIn(("SetSlotFromRvc", 2, "Use Voice"), self.fake.calls)

    def test_use_refuses_an_unknown_voice(self):
        self.assertEqual(self.Use(0, "Nobody").status_code, 400)

    def test_delete_clears_the_slot_and_removes_the_model(self):
        self.ImportPth("Gone Voice")
        self.Use(0, "Gone Voice")
        r = self.client.delete("/api/rvc/models/" + quote("Gone Voice"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "cleared_slots": [0]})
        self.assertIs(self.client.get("/api/state").json()["slots"][0]["loaded"], False)
        self.assertEqual(self.client.get("/api/rvc/models").json()["models"], [])

    def test_missing_parts_are_refused_with_the_expected_parts(self):
        r = self.client.post("/api/slot/0/rvc/import", files={"name": (None, "Only Name")})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], MODEL_SENTENCE)

    def test_zip_and_pth_together_are_refused(self):
        files = {"zip": ("voice.zip", self.ZipBytes(), "application/zip"),
                 "pth": ("voice.pth", self.PthBytes(), "application/octet-stream")}
        r = self.client.post("/api/slot/0/rvc/import", files=files)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], MODEL_SENTENCE)

    def test_unknown_part_is_refused(self):
        files = {"zip": ("voice.zip", self.ZipBytes(), "application/zip"),
                 "extra": ("x.txt", b"hi", "text/plain")}
        r = self.client.post("/api/slot/0/rvc/import", files=files)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "unknown form part 'extra'; send zip, pth, index or name")

    def test_wrong_slot_index_is_404(self):
        self.assertEqual(self.UploadZip(slot=10).status_code, 404)
        self.assertEqual(self.client.post("/api/slot/10/rvc/use", json={"name": "x"}).status_code, 404)
        self.assertEqual(self.client.post("/api/slot/10/rvc/link", json={"url": "https://x.example/a.zip"}).status_code, 404)

    def test_oversize_pth_is_413_and_leaves_no_temp_file(self):
        with mock.patch.object(rvc_store, "MAX_MODEL_BYTES", 1000):
            r = self.client.post("/api/slot/0/rvc/import",
                                 files={"pth": ("voice.pth", self.PthBytes(), "application/octet-stream")})
        self.assertEqual(r.status_code, 413)
        self.assertTrue(r.json()["detail"].startswith("the pth file is larger than"))
        self.assertEqual(os.listdir(rvc_store.IncomingDir()), [])
        self.assertFalse(any(call[0] == "SetSlotFromRvc" for call in self.fake.calls))

    def test_oversize_zip_is_413_and_leaves_no_temp_file(self):
        with mock.patch.object(rvc_store, "MAX_ZIP_BYTES", 1000):
            r = self.UploadZip()
        self.assertEqual(r.status_code, 413)
        self.assertTrue(r.json()["detail"].startswith("the zip file is larger than"))
        self.assertEqual(os.listdir(rvc_store.IncomingDir()), [])

    def test_declared_body_over_the_cap_is_413_before_any_part_is_read(self):
        with mock.patch.object(rvc_store, "MAX_ZIP_BYTES", 0):
            r = self.client.post("/api/slot/0/rvc/import",
                                 files={"pth": ("voice.pth", b"x" * (2 * MiB), "application/octet-stream")})
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.json()["detail"], "the upload is larger than 1 MB")
        self.assertEqual(os.listdir(rvc_store.IncomingDir()), [])

    def test_hostile_pickle_is_refused_and_runs_nothing(self):
        marker = self.root / "pwned.txt"

        class Hostile:
            def __reduce__(self):
                return (os.system, (f'echo pwned > "{marker}"',))

        evil = self.root / "evil.pth"
        torch.save(Hostile(), evil)
        r = self.client.post("/api/slot/0/rvc/import",
                             files={"pth": ("evil.pth", evil.read_bytes(), "application/octet-stream")})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(marker.exists())
        self.assertEqual(rvc_store.List(), [])

    def test_link_import_loads_the_slot_and_refusals_pass_through(self):
        model = self.ImportPth("Linked Voice")
        with mock.patch.object(rvc_store, "ImportLink", return_value=model) as link:
            r = self.client.post("/api/slot/1/rvc/link", json={"url": "https://example.com/voice.zip"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slot"]["model"], "Linked Voice")
        self.assertEqual(r.json()["model"]["slots"], [1])
        link.assert_called_once_with("https://example.com/voice.zip", index_url=None, name=None)
        self.assertNotIn("model_path", r.text)
        with mock.patch.object(rvc_store, "ImportLink",
                               side_effect=rvc_store.RvcImportError("the link must start with https://")):
            r = self.client.post("/api/slot/1/rvc/link", json={"url": "http://example.com/voice.zip"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "the link must start with https://")

    def test_link_body_fields_must_be_text(self):
        self.assertEqual(self.client.post("/api/slot/0/rvc/link", json={"url": 5}).status_code, 400)
        r = self.client.post("/api/slot/0/rvc/link", json={"url": "https://x.example/a.zip", "name": 7})
        self.assertEqual(r.status_code, 400)

    def test_name_longer_than_200_characters_is_refused(self):
        r = self.UploadZip(name="n" * 201)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "the name is longer than 200 characters")

    def test_state_carries_rvc_and_slot_kind_and_model(self):
        self.ImportPth("State Voice")
        self.Use(0, "State Voice")
        r = self.client.get("/api/state")
        state = r.json()
        self.assertEqual(state["rvc"]["weights"], WEIGHTS_MISSING)
        self.assertEqual([m["name"] for m in state["rvc"]["models"]], ["State Voice"])
        self.assertEqual(state["rvc"]["models"][0]["slots"], [0])
        self.assertEqual((state["slots"][0]["kind"], state["slots"][0]["model"]), ("rvc", "State Voice"))
        self.assertEqual((state["slots"][1]["kind"], state["slots"][1]["model"]), (None, None))

    def test_responses_never_expose_model_or_index_paths(self):
        self.ImportPth("Path Voice")
        self.Use(0, "Path Voice")
        for path in ("/api/state", "/api/rvc/models"):
            text = self.client.get(path).text
            self.assertNotIn("model_path", text, path)
            self.assertNotIn("index_path", text, path)


if __name__ == "__main__":
    unittest.main()
