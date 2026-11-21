"""download_weights.Fetch against a local HTTP server that honours Range: resume from a partial file, reject a damaged file."""
import hashlib
import http.server
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import download_weights
import integrity

PAYLOAD = os.urandom(20 * (1 << 20) + 123)  # three 8 MB chunks, the last one short


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        first, last = (int(v) for v in self.headers["Range"].split("=")[1].split("-"))
        body = PAYLOAD[first : last + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/m.bin"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _Model(self, sha=None):
        return {"name": "m", "relpath": "sub/m.bin", "url": self.url, "size": len(PAYLOAD), "sha256": sha or hashlib.sha256(PAYLOAD).hexdigest()}

    def test_resumes_a_partial_file_and_verifies_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, cache = Path(tmp) / "models", Path(tmp) / "cache.json"
            (models / "sub").mkdir(parents=True)
            part = models / "sub" / "m.bin.part"
            with open(part, "wb") as handle:  # a previous run got chunk 0 only, and the rest of the file is still empty
                handle.write(PAYLOAD[: download_weights.CHUNK_BYTES])
                handle.truncate(len(PAYLOAD))
            (models / "sub" / "m.bin.part.json").write_text('{"done": [0]}')
            seen = []
            dest = download_weights.Fetch(self._Model(), models, cache, lambda n, d, t: seen.append(d))
            self.assertEqual(seen[-1], len(PAYLOAD))
            self.assertGreater(seen[0], download_weights.CHUNK_BYTES)  # chunk 0 was not fetched again
            self.assertEqual(dest.read_bytes(), PAYLOAD)
            self.assertEqual(sorted(p.name for p in (models / "sub").iterdir()), ["m.bin"])
            manifest = integrity.Manifest("t", "x", {}, (self._Model(),))
            verifier = integrity.ModelVerifier(manifest, models, cache)
            verifier.Start()
            verifier.Wait()
            self.assertEqual(verifier.Stats()["cached"], 1)  # Fetch recorded the verified hash, so start-up only stats the file

    def test_a_damaged_download_is_refused_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            with self.assertRaisesRegex(download_weights.DownloadError, "damaged"):
                download_weights.Fetch(self._Model(sha="0" * 64), models, Path(tmp) / "cache.json")
            self.assertEqual(list((models / "sub").iterdir()), [])  # neither the part nor its sidecar is left behind


if __name__ == "__main__":
    unittest.main()
