"""Refusals that must fail closed: hostile pickles, zip slip, zip layout, size caps, index magic and the link guards.

Nothing here reaches the network. Name resolution is mocked, and FetchToFile is mocked wherever a link flow would download.
"""
import http.client
import io
import os
import pathlib
import socket
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import net_guard
from engine import rvc_store
from engine.rvc_checkpoint import LoadCheckpoint, UnsafeCheckpointError
from tests_seedvc.rvc_fixtures import SaveCheckpoint, TinyConfig
from ui import server

PUBLIC_IP = "93.184.216.34"


def _AddressInfo(ip):
    """The answer socket.getaddrinfo gives for one address."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 443))]


def _RedirectHandler(max_redirects):
    """The redirect handler that net_guard puts in front of every download."""
    opener = net_guard._Opener(max_redirects)
    return next(h for h in opener.handlers if isinstance(h, urllib.request.HTTPRedirectHandler))


def _WriteZip(path, members):
    """members is a list of (name, data, external_attr). The name is set after ZipInfo is built, because ZipInfo rewrites backslashes on Windows."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data, attr in members:
            info = zipfile.ZipInfo("member.bin")
            info.filename = name
            info.external_attr = attr
            archive.writestr(info, data)
    return pathlib.Path(path)


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


class HostilePickleTests(_StoreCase):
    def setUp(self):
        super().setUp()
        self.marker = self.root / "marker.txt"
        self.path = self.root / "hostile.pth"
        torch.save({"config": TinyConfig(40000), "weight": _Hostile(self.marker), "version": "v2", "f0": 1, "info": "hostile"}, self.path)

    def test_load_refuses_a_hostile_pickle(self):
        with self.assertRaises(UnsafeCheckpointError):
            LoadCheckpoint(self.path)
        self.assertFalse(self.marker.exists())

    def test_import_refuses_a_hostile_pickle_and_leaves_nothing_behind(self):
        with self.assertRaises(rvc_store.RvcImportError):
            rvc_store.ImportPth(self.path)
        self.assertFalse(self.marker.exists())
        self.assertFalse((rvc_store.StoreDir() / "hostile").exists())
        self.assertEqual(list(rvc_store.IncomingDir().iterdir()), [])

    def test_the_refusal_names_the_users_file_and_the_forbidden_reference(self):
        with self.assertRaises(rvc_store.RvcImportError) as caught:
            rvc_store.ImportPth(self.path)
        text = str(caught.exception)
        self.assertTrue(text.startswith("hostile.pth was refused: it contains a code reference ("), text)
        self.assertIn("system", text)
        self.assertNotIn("weights_only", text)
        self.assertTrue(text.endswith("Nothing in it was run."), text)


class ZipSlipTests(_StoreCase):
    NAMES = ("../evil.pth", "/abs/evil.pth", "C:/evil.pth", "a\\..\\evil.pth", "d/../../evil.pth", "sub\\evil.pth")

    def test_unsafe_entry_names_are_refused_before_anything_is_extracted(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                zip_path = _WriteZip(self.root / "bad.zip", [(name, self.fixture_bytes, 0)])
                with self.assertRaisesRegex(rvc_store.RvcImportError, "unsafe entry name"):
                    rvc_store.ImportZip(zip_path)
                self.assertEqual(list(self.root.rglob("evil.pth")), [])

    def test_a_symbolic_link_member_is_refused(self):
        zip_path = _WriteZip(self.root / "link.zip", [("link.pth", b"model.pth", 0o120777 << 16)])
        with self.assertRaisesRegex(rvc_store.RvcImportError, "symbolic link"):
            rvc_store.ImportZip(zip_path)


class ZipLayoutTests(_StoreCase):
    def test_a_zip_without_a_pth_is_refused(self):
        zip_path = _WriteZip(self.root / "none.zip", [("readme.txt", b"hello", 0)])
        with self.assertRaisesRegex(rvc_store.RvcImportError, "holds no .pth voice model"):
            rvc_store.ImportZip(zip_path)

    def test_a_zip_with_two_pth_files_is_refused(self):
        zip_path = _WriteZip(self.root / "two.zip", [("a.pth", self.fixture_bytes, 0), ("b.pth", self.fixture_bytes, 0)])
        with self.assertRaisesRegex(rvc_store.RvcImportError, "holds 2 .pth"):
            rvc_store.ImportZip(zip_path)

    def test_only_the_voice_members_are_extracted(self):
        zip_path = _WriteZip(self.root / "voice.zip", [("Voice.pth", self.fixture_bytes, 0), ("readme.txt", b"hello", 0)])
        result = rvc_store.ImportZip(zip_path)
        self.assertEqual(result["name"], "Voice")
        self.assertEqual(result["source"], "zip")
        self.assertFalse(result["has_index"])
        self.assertEqual(sorted(p.name for p in (rvc_store.StoreDir() / "Voice").iterdir()), ["meta.json", "model.pth"])
        self.assertEqual(list(rvc_store.IncomingDir().iterdir()), [])


class OversizeTests(_StoreCase):
    def test_a_pth_over_the_cap_is_refused(self):
        self.assertGreater(self.fixture.stat().st_size, 1000)
        with mock.patch.object(rvc_store, "MAX_MODEL_BYTES", 1000):
            with self.assertRaisesRegex(rvc_store.RvcImportError, "larger than"):
                rvc_store.ImportPth(self.fixture)

    def test_a_zip_over_the_cap_is_refused(self):
        zip_path = _WriteZip(self.root / "big.zip", [("Voice.pth", self.fixture_bytes, 0)])
        self.assertGreater(zip_path.stat().st_size, 1000)
        with mock.patch.object(rvc_store, "MAX_ZIP_BYTES", 1000):
            with self.assertRaisesRegex(rvc_store.RvcImportError, "larger than"):
                rvc_store.ImportZip(zip_path)


class IndexMagicTests(_StoreCase):
    def test_an_index_without_the_faiss_magic_is_refused(self):
        index = self.root / "bad.index"
        index.write_bytes(b"XXXX" + bytes(64))
        with self.assertRaisesRegex(rvc_store.RvcImportError, "not a FAISS index"):
            rvc_store.ImportPth(self.fixture, index_path=index)

    def test_an_index_with_the_faiss_magic_is_accepted(self):
        index = self.root / "good.index"
        index.write_bytes(b"IwFl" + bytes(64))
        result = rvc_store.ImportPth(self.fixture, index_path=index, name="indexed")
        self.assertTrue(result["has_index"])


class LinkGuardTests(unittest.TestCase):
    def setUp(self):
        dns = mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo(PUBLIC_IP))
        dns.start()
        self.addCleanup(dns.stop)

    def test_a_public_address_passes(self):
        self.assertTrue(net_guard.IsPublicAddress(PUBLIC_IP))

    def test_private_local_and_reserved_addresses_fail(self):
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fc00::1", "100.64.0.1", "0.0.0.0", "224.0.0.1", "::ffff:127.0.0.1"):
            with self.subTest(ip=ip):
                self.assertFalse(net_guard.IsPublicAddress(ip))

    def test_a_loopback_answer_is_refused(self):
        with mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo("127.0.0.1")):
            with self.assertRaises(net_guard.GuardError) as caught:
                net_guard.ResolvePublic("localhost")
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(str(caught.exception), "the link points at a private or local address")

    def test_a_failed_lookup_is_a_502(self):
        with mock.patch.object(net_guard.socket, "getaddrinfo", side_effect=socket.gaierror("no such host")):
            with self.assertRaises(net_guard.GuardError) as caught:
                net_guard.ResolvePublic("missing.example")
        self.assertEqual(caught.exception.status, 502)
        self.assertTrue(str(caught.exception).startswith("could not resolve missing.example"))

    def test_plain_http_is_refused(self):
        with self.assertRaises(net_guard.GuardError) as caught:
            net_guard.CheckUrl("http://example.com/a.pth")
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(str(caught.exception), "the link must start with https://")

    def test_drive_and_mega_links_are_refused(self):
        for url in ("https://drive.google.com/file/d/x/view", "https://docs.google.com/uc?id=x", "https://mega.nz/file/x", "https://sub.mega.io/x"):
            with self.subTest(url=url):
                with self.assertRaises(net_guard.GuardError) as caught:
                    net_guard.CheckUrl(url)
                self.assertEqual(caught.exception.status, 400)
                self.assertIn("Google Drive and Mega links", str(caught.exception))

    def test_a_literal_private_host_is_refused(self):
        with mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo("127.0.0.1")):
            with self.assertRaises(net_guard.GuardError) as caught:
                net_guard.CheckUrl("https://127.0.0.1/a.pth")
        self.assertEqual(caught.exception.status, 400)

    def test_a_public_https_link_passes(self):
        net_guard.CheckUrl("https://public.example/a.pth")

    def test_a_redirect_target_is_checked_by_the_same_function(self):
        with mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo("10.0.0.1")):
            with self.assertRaises(net_guard.GuardError):
                net_guard.CheckUrl("https://10.0.0.1/x.pth")

    def test_the_redirect_handler_refuses_a_private_hop(self):
        handler = _RedirectHandler(5)
        request = urllib.request.Request("https://public.example/a.pth")
        with mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo("10.0.0.1")):
            with self.assertRaises(net_guard.GuardError) as caught:
                handler.redirect_request(request, None, 302, "Found", {}, "https://10.0.0.1/x.pth")
        self.assertEqual(caught.exception.status, 400)

    def test_the_redirect_handler_stops_after_five_hops(self):
        handler = _RedirectHandler(5)
        request = urllib.request.Request("https://public.example/a.pth")
        for hop in range(5):
            request = handler.redirect_request(request, None, 302, "Found", {}, f"https://public.example/hop{hop}.pth")
        with self.assertRaises(net_guard.GuardError) as caught:
            handler.redirect_request(request, None, 302, "Found", {}, "https://public.example/last.pth")
        self.assertEqual(str(caught.exception), "too many redirects")


class _RawSocket:
    """Just enough socket for http.client.HTTPResponse to parse canned bytes."""

    def __init__(self, raw):
        self.raw = raw

    def makefile(self, mode):
        return io.BytesIO(self.raw)


class _RedirectingConnection:
    """Stands in for the guarded HTTPS connection: every request is answered with a 302 to target."""

    target = ""

    def __init__(self, host, timeout=None, context=None):
        self.sock = None

    def set_debuglevel(self, level):
        pass

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        response = http.client.HTTPResponse(_RawSocket(f"HTTP/1.1 302 Found\r\nLocation: {self.target}\r\nContent-Length: 0\r\n\r\n".encode()))
        response.begin()
        return response

    def close(self):
        pass


class PinnedConnectionTests(unittest.TestCase):
    """The connection dials the address its own lookup proved public, so a later DNS answer cannot send it to a private host."""

    @staticmethod
    def Lookups(public_answers):
        """A getaddrinfo that answers PUBLIC_IP for its first public_answers calls and 127.0.0.1 after that."""
        calls = []

        def lookup(host, port, *args, **kwargs):
            calls.append(host)
            return _AddressInfo(PUBLIC_IP if len(calls) <= public_answers else "127.0.0.1")

        return calls, lookup

    @staticmethod
    def Dials():
        """A socket.create_connection that records the address it is asked for and then refuses, so nothing real is reached."""
        dialled = []

        def dial(address, timeout=None, source_address=None):
            dialled.append(address[0])
            raise ConnectionRefusedError("refused by the test")

        return dialled, dial

    def test_an_answer_that_turns_private_after_the_first_lookup_is_never_dialled(self):
        calls, lookup = self.Lookups(1)
        dialled, dial = self.Dials()
        with mock.patch("socket.getaddrinfo", lookup), mock.patch("socket.create_connection", dial):
            with self.assertRaisesRegex(net_guard.GuardError, "private or local address"):
                net_guard.FetchBytes("https://rebind.example/a.wav", 1024)
        self.assertEqual(dialled, [])
        self.assertEqual(len(calls), 2)  # the early check, then the lookup the connection itself uses

    def test_the_connection_dials_the_validated_address_and_not_the_name(self):
        calls, lookup = self.Lookups(10)
        dialled, dial = self.Dials()
        with mock.patch("socket.getaddrinfo", lookup), mock.patch("socket.create_connection", dial):
            with self.assertRaisesRegex(net_guard.GuardError, "could not download the link"):
                net_guard.FetchBytes("https://rebind.example/a.wav", 1024)
        self.assertEqual(dialled, [PUBLIC_IP])

    def test_a_file_download_has_the_same_guard_and_leaves_no_file(self):
        calls, lookup = self.Lookups(1)
        dialled, dial = self.Dials()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            target = pathlib.Path(folder) / "voice.part"
            with mock.patch("socket.getaddrinfo", lookup), mock.patch("socket.create_connection", dial):
                with self.assertRaisesRegex(net_guard.GuardError, "private or local address"):
                    net_guard.FetchToFile("https://rebind.example/a.pth", target, 1024)
            self.assertEqual((dialled, target.exists()), ([], False))

    def test_the_audio_link_route_is_guarded_the_same_way(self):
        calls, lookup = self.Lookups(1)
        dialled, dial = self.Dials()
        with mock.patch("socket.getaddrinfo", lookup), mock.patch("socket.create_connection", dial):
            with self.assertRaises(server.HTTPException) as caught:
                server._FetchAudio("https://rebind.example/a.wav")
        self.assertEqual((caught.exception.status_code, dialled), (400, []))
        self.assertEqual(caught.exception.detail, "the link points at a private or local address")

    def test_a_redirect_to_a_private_target_is_refused_by_the_real_opener(self):
        def lookup(host, port, *args, **kwargs):
            return _AddressInfo(host if host[0].isdigit() else PUBLIC_IP)

        for target, message in (("http://127.0.0.1/x.wav", "must start with https://"), ("https://10.0.0.5/x.wav", "private or local address")):
            with self.subTest(target=target):
                connection = type("Connection", (_RedirectingConnection,), {"target": target})
                with mock.patch("socket.getaddrinfo", lookup), mock.patch.object(net_guard, "_GuardedHTTPSConnection", connection):
                    with self.assertRaisesRegex(net_guard.GuardError, message):
                        net_guard.FetchBytes("https://public.example/a.wav", 1024)


class ImportLinkTests(_StoreCase):
    def setUp(self):
        super().setUp()
        dns = mock.patch.object(net_guard.socket, "getaddrinfo", return_value=_AddressInfo(PUBLIC_IP))
        dns.start()
        self.addCleanup(dns.stop)

    def test_a_text_link_is_refused_before_any_fetch(self):
        with mock.patch.object(rvc_store, "FetchToFile") as fetch:
            with self.assertRaisesRegex(rvc_store.RvcImportError, "straight at a .pth or .zip file"):
                rvc_store.ImportLink("https://public.example/voice.txt")
        fetch.assert_not_called()

    def test_a_web_page_answer_is_refused(self):
        def ServePage(url, dest, max_bytes, **_):
            pathlib.Path(dest).write_bytes(b"<!DOCTYPE html><html>")
            return {"final_url": url, "content_type": "text/html", "size": 22, "head": b"<!DOCTYPE html><html>"}

        with mock.patch.object(rvc_store, "FetchToFile", side_effect=ServePage):
            with self.assertRaisesRegex(rvc_store.RvcImportError, "web page"):
                rvc_store.ImportLink("https://public.example/voice.pth")

    def test_a_pth_link_imports_with_link_source(self):
        def ServeVoice(url, dest, max_bytes, **_):
            pathlib.Path(dest).write_bytes(self.fixture_bytes)
            return {"final_url": url, "content_type": "application/octet-stream", "size": len(self.fixture_bytes), "head": self.fixture_bytes[:512]}

        with mock.patch.object(rvc_store, "FetchToFile", side_effect=ServeVoice) as fetch:
            result = rvc_store.ImportLink("https://public.example/voices/Mika.pth")
        self.assertEqual(result["name"], "Mika")
        self.assertEqual(result["source"], "link")
        self.assertEqual(fetch.call_args.kwargs["max_bytes"], rvc_store.MAX_MODEL_BYTES)
        self.assertEqual(rvc_store.Get("Mika")["source"], "link")

    def test_a_zip_link_with_an_index_url_is_refused(self):
        with mock.patch.object(rvc_store, "FetchToFile") as fetch:
            with self.assertRaisesRegex(rvc_store.RvcImportError, "carries its own index"):
                rvc_store.ImportLink("https://public.example/voice.zip", index_url="https://public.example/voice.index")
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
