"""runtime_libs and tools/make_torch_manifest.py against a local HTTP server that honours Range.

The fake wheel is a real zip with deflated members, served with Content-Range like the PyTorch CDN. Covered: manifest build and its mismatch
refusals, resume of a partial member, a damaged chunk, a wrong hash, a server that ignores Range, damage detection on start-up, and every
junction case of Link().
"""
import hashlib
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import make_torch_manifest  # noqa: E402
import runtime_libs  # noqa: E402

MEMBERS = {
    "torch/lib/a.dll": os.urandom(300 * 1024) + b"\0" * (700 * 1024),
    "torch/lib/sub/b.bin": b"hello torch\n" * 400,
    "torch/lib/c.dll": os.urandom(200 * 1024),
}


def _Wheel() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in MEMBERS.items():
            zf.writestr(name, data)
        zf.writestr("torch/__init__.py", b"# not a library\n")
    return buffer.getvalue()


WHEEL = _Wheel()


class Server:
    """Serves WHEEL with Range support and records every request start. Tests flip the switches below."""

    def __init__(self):
        self.starts = []
        self.corrupt_starts = set()  # first request starting at one of these offsets gets one byte flipped
        self.ignore_range = False

        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                if outer.ignore_range or not header:
                    body = WHEEL
                    self.send_response(200)
                else:
                    first, last = (int(v) for v in header.split("=")[1].split("-"))
                    outer.starts.append(first)
                    body = bytearray(WHEEL[first:last + 1])
                    if first in outer.corrupt_starts:
                        outer.corrupt_starts.discard(first)
                        body[len(body) // 2] ^= 0xFF
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first}-{last}/{len(WHEEL)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(bytes(body))

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/torch.whl"

    def Close(self):
        self.httpd.shutdown()


class LibsTests(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.venv_lib = self.root / "venv_lib"
        for name, data in MEMBERS.items():
            path = self.venv_lib / name[len("torch/lib/"):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.saved = (runtime_libs.CHUNK_BYTES, runtime_libs.RETRY_BACKOFF_SECONDS, runtime_libs.RETRIES)
        runtime_libs.CHUNK_BYTES = 64 * 1024  # several chunks per member
        runtime_libs.RETRY_BACKOFF_SECONDS = 0
        runtime_libs.RETRIES = 2
        self.manifest_path = self.root / "torch_lib_manifest.json"
        self.runtime = self.root / "runtime"
        self.bundle_lib = self.root / "bundle" / "torch" / "lib"

    def tearDown(self):
        runtime_libs.CHUNK_BYTES, runtime_libs.RETRY_BACKOFF_SECONDS, runtime_libs.RETRIES = self.saved
        self.server.Close()
        self.tmp.cleanup()

    def _Manifest(self) -> dict:
        doc = make_torch_manifest.Build(self.server.url, self.venv_lib)
        self.manifest_path.write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def _Entry(self, relpath: str) -> dict:
        doc = self._Manifest()
        entry = next(f for f in doc["files"] if f["relpath"] == relpath)
        return runtime_libs._WithWheel(entry, doc["wheel"])

    # ------------------------------------------------------------ manifest tool
    def test_tool_builds_entries_that_locate_each_member(self):
        doc = self._Manifest()
        self.assertEqual(len(doc["files"]), 3)
        for entry in doc["files"]:
            raw = WHEEL[entry["data_offset"]: entry["data_offset"] + entry["compress_size"]]
            inflated = zlib.decompressobj(-15).decompress(raw)
            self.assertEqual(inflated, MEMBERS[entry["member"]])
            self.assertEqual(entry["crc32"], zlib.crc32(MEMBERS[entry["member"]]))
            self.assertEqual(entry["sha256"], hashlib.sha256(MEMBERS[entry["member"]]).hexdigest())

    def test_tool_refuses_a_local_file_that_differs_from_the_wheel(self):
        (self.venv_lib / "a.dll").write_bytes(b"x" * len(MEMBERS["torch/lib/a.dll"]))
        with self.assertRaisesRegex(make_torch_manifest.ManifestError, "CRC-32"):
            make_torch_manifest.Build(self.server.url, self.venv_lib)

    def test_tool_refuses_an_extra_local_file(self):
        (self.venv_lib / "extra.dll").write_bytes(b"extra")
        with self.assertRaisesRegex(make_torch_manifest.ManifestError, "file lists differ"):
            make_torch_manifest.Build(self.server.url, self.venv_lib)

    # ------------------------------------------------------------ download
    def test_a_partial_member_resumes_from_its_finished_chunks(self):
        entry = self._Entry("a.dll")
        folder = self.runtime
        folder.mkdir(parents=True)
        zpart = folder / "a.dll.z.part"
        raw_first = WHEEL[entry["data_offset"]: entry["data_offset"] + runtime_libs.CHUNK_BYTES]
        with open(zpart, "wb") as handle:
            handle.write(raw_first)
            handle.truncate(entry["compress_size"])
        (folder / "a.dll.z.part.json").write_text(json.dumps({"done": [0]}))
        self.server.starts.clear()
        runtime_libs.FetchMember(entry, folder)
        self.assertEqual(len(self.server.starts), -(-entry["compress_size"] // runtime_libs.CHUNK_BYTES) - 1)  # chunk 0 not fetched again
        self.assertNotIn(entry["data_offset"], self.server.starts)
        self.assertEqual((folder / "a.dll").read_bytes(), MEMBERS["torch/lib/a.dll"])
        self.assertFalse(zpart.exists())
        self.assertFalse((folder / "a.dll.z.part.json").exists())

    def test_a_damaged_chunk_is_refused_and_leaves_no_file(self):
        entry = self._Entry("a.dll")
        self.server.corrupt_starts.add(entry["data_offset"] + runtime_libs.CHUNK_BYTES)
        with self.assertRaisesRegex(runtime_libs.RuntimeLibsError, "damaged"):
            runtime_libs.FetchMember(entry, self.runtime)
        self.assertEqual(sorted(p.name for p in self.runtime.iterdir()), [])

    def test_a_wrong_sha256_is_refused(self):
        entry = dict(self._Entry("c.dll"), sha256="0" * 64)
        with self.assertRaisesRegex(runtime_libs.RuntimeLibsError, "damaged"):
            runtime_libs.FetchMember(entry, self.runtime)
        self.assertEqual(sorted(p.name for p in self.runtime.iterdir()), [])

    def test_a_server_that_ignores_range_is_refused(self):
        entry = self._Entry("c.dll")
        self.server.ignore_range = True
        with self.assertRaisesRegex(runtime_libs.RuntimeLibsError, "ignored the byte range"):
            runtime_libs.FetchMember(entry, self.runtime)
        self.assertFalse((self.runtime / "c.dll").exists())

    def test_a_wheel_of_another_size_is_refused(self):
        entry = dict(self._Entry("c.dll"), wheel_size=len(WHEEL) + 1)
        with self.assertRaises(RuntimeError):
            runtime_libs.FetchMember(entry, self.runtime)

    # ------------------------------------------------------------ start-up check
    def test_ensure_fetches_once_then_only_stats_and_flags_damage(self):
        self._Manifest()
        runtime_libs.Ensure(None, self.manifest_path, self.runtime, self.bundle_lib)
        for relpath, data in (("a.dll", MEMBERS["torch/lib/a.dll"]), ("sub/b.bin", MEMBERS["torch/lib/sub/b.bin"]), ("c.dll", MEMBERS["torch/lib/c.dll"])):
            self.assertEqual((self.bundle_lib / relpath).read_bytes(), data)
        self.assertEqual(runtime_libs.Missing(self.manifest_path, self.runtime), [])
        self.server.starts.clear()
        runtime_libs.Ensure(None, self.manifest_path, self.runtime, self.bundle_lib)
        self.assertEqual(self.server.starts, [])
        (self.runtime / "c.dll").write_bytes(b"\0" * len(MEMBERS["torch/lib/c.dll"]))  # same size, wrong bytes
        (self.runtime / "sub" / "b.bin").unlink()
        self.assertEqual(sorted(e["relpath"] for e in runtime_libs.Missing(self.manifest_path, self.runtime)), ["c.dll", "sub/b.bin"])

    # ------------------------------------------------------------ junctions
    def test_link_creates_and_then_leaves_a_correct_link_alone(self):
        self._Manifest()
        runtime_libs.Ensure(None, self.manifest_path, self.runtime, self.bundle_lib)  # creates the runtime folder content and the link
        runtime_libs.Link(self.runtime, self.bundle_lib)
        self.assertTrue(runtime_libs._SameFolder(self.bundle_lib, self.runtime))

    def test_link_replaces_a_junction_that_points_elsewhere(self):
        other = self.root / "other"
        other.mkdir()
        (other / "keep.txt").write_text("keep")
        self.runtime.mkdir()
        self.bundle_lib.parent.mkdir(parents=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(self.bundle_lib), str(other)], check=True, capture_output=True)
        runtime_libs.Link(self.runtime, self.bundle_lib)
        self.assertTrue(runtime_libs._SameFolder(self.bundle_lib, self.runtime))
        self.assertEqual((other / "keep.txt").read_text(), "keep")  # only the link was removed

    def test_link_replaces_an_empty_real_folder(self):
        self.runtime.mkdir()
        self.bundle_lib.mkdir(parents=True)
        runtime_libs.Link(self.runtime, self.bundle_lib)
        self.assertTrue(runtime_libs._SameFolder(self.bundle_lib, self.runtime))

    def test_link_moves_a_non_empty_real_folder_aside_and_keeps_it(self):
        self.runtime.mkdir()
        self.bundle_lib.mkdir(parents=True)
        (self.bundle_lib / "old.dll").write_bytes(b"old")
        runtime_libs.Link(self.runtime, self.bundle_lib)
        self.assertTrue(runtime_libs._SameFolder(self.bundle_lib, self.runtime))
        asides = list(self.bundle_lib.parent.glob("lib.replaced-*"))
        self.assertEqual(len(asides), 1)
        self.assertEqual((asides[0] / "old.dll").read_bytes(), b"old")

    def test_link_refuses_a_file_in_the_way(self):
        self.runtime.mkdir()
        self.bundle_lib.parent.mkdir(parents=True)
        self.bundle_lib.write_bytes(b"not a folder")
        with self.assertRaisesRegex(runtime_libs.RuntimeLibsError, "is a file"):
            runtime_libs.Link(self.runtime, self.bundle_lib)



class WindowedJobRoutingTest(unittest.TestCase):
    def test_windowed_runtime_job_passes_progress_to_the_progress_slot_not_runtime_dir(self):
        """Regression: job.run(progress) must not land the callback in FetchMember's runtime_dir slot (the packaged app died there)."""
        import app_seedvc
        from unittest import mock
        entry = {"member": "torch/lib/c10.dll"}
        def progress_cb(name: str, done: int, size: int) -> None:
            pass
        calls: list[tuple] = []
        def fake(entry_arg, runtime_dir=None, progress=None):
            calls.append((entry_arg, runtime_dir, progress))
        with mock.patch.object(runtime_libs, "FetchMember", fake):
            app_seedvc._FetchRuntimeEntry(entry, progress=progress_cb)
        self.assertEqual(calls, [(entry, None, progress_cb)])

    def test_windowed_runtime_job_with_none_progress_leaves_runtime_dir_none(self):
        """The non-windowed path job.run(None) must stay unaffected by the callback-slot fix."""
        import app_seedvc
        from unittest import mock
        entry = {"member": "torch/lib/c10.dll"}
        calls: list[tuple] = []
        def fake(entry_arg, runtime_dir=None, progress=None):
            calls.append((entry_arg, runtime_dir, progress))
        with mock.patch.object(runtime_libs, "FetchMember", fake):
            app_seedvc._FetchRuntimeEntry(entry, None)
        self.assertEqual(calls, [(entry, None, None)])


class ChunkDeadlineTest(unittest.TestCase):
    def test_a_trickling_range_read_raises_after_the_deadline(self):
        """Regression: a connection that feeds every socket read inside the timeout but never finishes the chunk must raise (and be
        retried), not hang for hours like the first-run cublas64_12 fetch did."""
        from unittest import mock
        fake_response = io.BytesIO(b"\x00" * 128)
        fake_response.status = 206
        fake_response.headers = {"Content-Range": "bytes 0-127/128"}
        fake_ctx = fake_response  # BytesIO already supports the context-manager protocol
        file_ = runtime_libs.HttpRangeFile("http://localhost/wheel.whl", 128)
        file_.seek(0)
        with mock.patch.object(runtime_libs.urllib.request, "urlopen", lambda request, timeout: fake_ctx), \
             mock.patch.object(runtime_libs.time, "monotonic", side_effect=[1000.0, 1301.0]):
            with self.assertRaisesRegex(OSError, "beyond the 300 s deadline"):
                file_.readinto(bytearray(128))

if __name__ == "__main__":
    unittest.main()
