"""RVC base-model registry: pinned sources, the first-use download state machine, and the list in installer/model_sources.json."""
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import rvc_weights  # noqa: E402


class RvcWeightsTests(unittest.TestCase):
    def test_pinned_sources_are_https_and_pinned_to_the_revision(self):
        self.assertEqual(len(rvc_weights.HF_REVISION), 40)
        for model in rvc_weights.MODELS:
            self.assertTrue(model["url"].startswith("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/" + rvc_weights.HF_REVISION + "/"))
            self.assertIsInstance(model["size"], int)
            self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$")

    def test_recorded_list_matches_the_json_record(self):
        record = json.loads((Path(ROOT) / "installer" / "model_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(record["rvc_models"]), rvc_weights.MODELS)
        self.assertEqual([m["name"] for m in record["models"]][:1], ["Seed-VC DiT (xlsr-tiny)"])

    def test_present_verified_files_become_ready_without_a_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "w"
            models = []
            for rel, body in (("hubert_base/config.json", b"{}"), ("rmvpe.pt", b"x" * 64)):
                (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                (folder / rel).write_bytes(body)
                models.append({"name": rel, "relpath": rel, "url": "https://example.invalid/" + rel, "size": len(body),
                               "sha256": hashlib.sha256(body).hexdigest()})
            rvc_weights.Reset()
            self.addCleanup(rvc_weights.Reset)
            rvc_weights.Ensure(tuple(models), folder)
            self.assertEqual(_Wait(), "ready")

    def test_a_damaged_file_is_an_error_and_is_never_called_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "w"
            (folder).mkdir()
            (folder / "rmvpe.pt").write_bytes(b"damaged")
            models = ({"name": "rmvpe", "relpath": "rmvpe.pt", "url": "https://example.invalid/rmvpe.pt", "size": 7,
                       "sha256": "0" * 64},)
            rvc_weights.Reset()
            self.addCleanup(rvc_weights.Reset)
            rvc_weights.Ensure(models, folder)
            self.assertEqual(_Wait(), "error")
            self.assertIn("rmvpe.pt", rvc_weights.Status()["message"])


def _Wait(timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = rvc_weights.Status()["state"]
        if state != "downloading":
            return state
        time.sleep(0.05)
    return rvc_weights.Status()["state"]


if __name__ == "__main__":
    unittest.main()
