"""Endpoint tests for ui/server.py against ui/fake_pipeline.FakePipeline (a test double)."""
from __future__ import annotations

import io
import os
import sys
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402
from fastapi.routing import APIWebSocketRoute  # noqa: E402
from starlette.routing import Mount  # noqa: E402

import seedvc_params as P  # noqa: E402
from ui.fake_pipeline import FakePipeline  # noqa: E402
import ui.server as server  # noqa: E402
from ui.server import CreateApp  # noqa: E402

LOOPBACK = {"Host": "127.0.0.1"}

CONTRACT_ROUTES = {
    ("/", "GET"),
    ("/api/state", "GET"),
    ("/api/param", "POST"),
    ("/api/preset", "POST"),
    ("/api/manual_slot", "POST"),
    ("/api/devices", "GET"),
    ("/api/start", "POST"),
    ("/api/stop", "POST"),
    ("/api/slot/{idx}", "POST"),
    ("/api/slot/{idx}", "DELETE"),
    ("/api/slot/{idx}/url", "POST"),
    ("/api/open_voice_library", "POST"),
    ("/api/slot/{idx}/record/start", "POST"),
    ("/api/slot/{idx}/record/stop", "POST"),
    ("/api/rvc/models", "GET"),
    ("/api/slot/{idx}/rvc/import", "POST"),
    ("/api/slot/{idx}/rvc/link", "POST"),
    ("/api/slot/{idx}/rvc/use", "POST"),
    ("/api/rvc/models/{name}", "DELETE"),
    ("/ws", "WEBSOCKET"),
    ("/static", "MOUNT"),
}


def WavBytes(seconds: float, sr: int, channels: int = 1) -> bytes:
    t = np.arange(int(round(seconds * sr))) / sr
    tone = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    data = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class UiServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakePipeline()
        self.client = TestClient(CreateApp(self.fake), base_url="http://127.0.0.1")

    def Upload(self, idx: int, data: bytes, name: str = "ref.wav"):
        return self.client.post(f"/api/slot/{idx}", files={"file": (name, data, "audio/wav")})

    def test_state_has_ranges_defaults_slots_and_presets(self):
        body = self.client.get("/api/state").json()
        self.assertEqual(body["ranges"]["block_ms"], {"type": "int", "min": 100, "max": 500, "default": 220})
        self.assertEqual(body["ranges"]["emotion_mode"]["values"], ["auto", "manual"])
        self.assertEqual({k: body["defaults"][k] for k in
                          ("block_ms", "overlap_ms", "diffusion_steps", "cfg_rate", "left_context_s")},
                         {"block_ms": 220, "overlap_ms": 45.0, "diffusion_steps": 10, "cfg_rate": 0.65,
                          "left_context_s": 2.0})
        self.assertEqual(body["slot_names"], list(P.SLOT_NAMES))
        self.assertEqual([(p["name"], p["disabled"]) for p in body["presets"]],
                         [("Natural Conversational", False), ("High-Energy Gaming / Ragebait", False),
                          ("Soft / Whisper", False), ("Ultra-Low- (cut off in source)", True),
                          ("Bright Announcer", False), ("Deep Narrator", False), ("Gruff Villain", False),
                          ("Chipmunk Hype", False), ("Calm Dialog", False), ("Flat Robot", False),
                         ("Ultra-Low-", True)])
        self.assertEqual(body["presets"][10]["label"], "Ultra-Low-... (name cut off in spec)")
        self.assertEqual(body["hotkeys"], "Ctrl+Alt")
        self.assertIs(body["telemetry"]["running"], True)
        self.assertEqual([s["loaded"] for s in body["slots"]], [False] * 10)

    def test_param_post_sets_value(self):
        r = self.client.post("/api/param", json={"name": "diffusion_steps", "value": 12})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "name": "diffusion_steps", "value": 12})
        self.assertEqual(self.fake.params.Get("diffusion_steps"), 12)

    def test_param_post_rejects_bad_values_with_400_and_message(self):
        cases = [
            ({"name": "diffusion_steps", "value": 17}, "diffusion_steps=17 is outside 6..16"),
            ({"name": "cfg_rate", "value": "loud"}, "cfg_rate must be a number"),
            ({"name": "emotion_mode", "value": "loud"}, "emotion_mode must be one of ('auto', 'manual')"),
            ({"name": "monitor_enabled", "value": 1}, "monitor_enabled must be a bool"),
            ({"name": "no_such_param", "value": 1}, "unknown parameter 'no_such_param'"),
        ]
        for body, message in cases:
            r = self.client.post("/api/param", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertEqual(r.json()["detail"], message)
        r = self.client.post("/api/param", content=b'{"name": "cfg_rate", "value": NaN}',
                             headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "cfg_rate=nan is outside 0.0..1.0")
        r = self.client.post("/api/param", content=b"not json", headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.fake.params.Get("diffusion_steps"), 10)
        self.assertEqual(self.fake.params.Get("cfg_rate"), 0.65)

    def test_bool_and_enum_params_are_stored(self):
        r = self.client.post("/api/param", json={"name": "monitor_enabled", "value": True})
        self.assertEqual(r.status_code, 200)
        self.assertIs(self.fake.params.Get("monitor_enabled"), True)

    def test_slot_upload_loads_mono_float32_and_reports_duration(self):
        r = self.Upload(0, WavBytes(1.5, 16000))
        self.assertEqual(r.status_code, 200)
        slot = r.json()["slot"]
        self.assertTrue(slot["loaded"])
        self.assertEqual(slot["seconds"], 1.5)
        self.assertEqual(slot["name"], "slot 1")
        self.assertEqual(self.fake.calls[-1], ("SetSlotFromWav", 0, 24000, "float32", 1, 16000))

    def test_stereo_upload_is_downmixed_to_mono_at_its_own_rate(self):
        r = self.Upload(1, WavBytes(0.5, 48000, channels=2))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slot"]["seconds"], 0.5)
        self.assertEqual(self.fake.calls[-1], ("SetSlotFromWav", 1, 24000, "float32", 1, 48000))

    def test_flac_upload_is_accepted(self):
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(8000, dtype="float32") + 0.1, 16000, format="FLAC")
        r = self.Upload(0, buffer.getvalue(), "x.flac")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["slot"]["seconds"], 0.5)

    def test_import_from_a_link_refuses_unsafe_or_non_audio_targets(self):
        before = len(self.fake.calls)
        for url, status in (("http://example.com/a.wav", 400), ("https://127.0.0.1/a.wav", 400), ("https://localhost/a.wav", 400), ("not a url", 400)):
            r = self.client.post("/api/slot/0/url", json={"url": url})
            self.assertEqual(r.status_code, status, url)
        self.assertEqual(self.client.post("/api/slot/0/url", json={"nope": 1}).status_code, 400)
        self.assertEqual(len(self.fake.calls), before)

    def test_import_from_a_link_loads_audio_and_refuses_a_web_page(self):
        wav = WavBytes(1.0, 16000)
        with mock.patch.object(server, "_FetchAudio", return_value=wav):
            r = self.client.post("/api/slot/2/url", json={"url": "https://example.com/v.wav"})
        self.assertEqual((r.status_code, r.json()["slot"]["seconds"]), (200, 1.0))
        self.assertEqual(self.fake.calls[-1][:2], ("SetSlotFromWav", 2))
        with mock.patch.object(server, "FetchBytes", return_value=(b"<html>not audio</html>", {})):
            r = self.client.post("/api/slot/2/url", json={"url": "https://example.com/page"})
        self.assertEqual(r.status_code, 415)

    def test_voice_library_button_opens_the_fixed_site_only(self):
        with mock.patch("webbrowser.open") as opened:
            r = self.client.post("/api/open_voice_library", json={"url": "https://evil.example/"})
        self.assertEqual(r.status_code, 200)
        opened.assert_called_once_with("https://voice-models.com/")

    def test_non_audio_and_empty_uploads_are_refused(self):
        cases = [
            (b"hello, this is not audio at all", "x.wav"),
            (b"", "x.wav"),
        ]
        before = len(self.fake.calls)
        for data, name in cases:
            r = self.Upload(0, data, name)
            self.assertEqual(r.status_code, 415, name)
        self.assertEqual(len(self.fake.calls), before)

    def test_corrupt_riff_is_400_and_missing_field_is_400(self):
        r = self.Upload(0, b"RIFF" + b"\x00" * 4 + b"WAVE" + b"garbage" * 10)
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/slot/0", data={"x": "1"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], 'send the .wav as multipart field "file"')

    def test_slot_index_bounds(self):
        for path in ("/api/slot/10", "/api/slot/-1"):
            self.assertEqual(self.client.delete(path).status_code, 404, path)
        self.assertEqual(self.Upload(10, WavBytes(0.1, 16000)).status_code, 404)
        self.assertEqual(self.client.post("/api/slot/10/record/start").status_code, 404)
        self.assertEqual(self.client.post("/api/slot/11/record/stop").status_code, 404)
        self.assertEqual(self.client.delete("/api/slot/x").status_code, 422)

    def test_clear_slot_empties_it(self):
        self.Upload(2, WavBytes(0.5, 16000))
        r = self.client.delete("/api/slot/2")
        self.assertEqual(r.status_code, 200)
        self.assertIs(r.json()["slot"]["loaded"], False)
        self.assertEqual(self.fake.calls[-1], ("ClearSlot", 2))

    def test_record_start_and_stop_store_into_the_slot(self):
        r = self.client.post("/api/slot/1/record/start")
        self.assertEqual(r.json(), {"ok": True, "slot": 1, "max_seconds": 10.0})
        self.assertEqual(self.client.get("/api/state").json()["recording"]["slot"], 1)
        self.assertEqual(self.client.post("/api/slot/1/record/start").status_code, 409)
        self.assertEqual(self.client.post("/api/slot/2/record/stop").status_code, 409)
        with self.client.websocket_connect("/ws", headers=LOOPBACK) as ws:
            self.assertEqual(ws.receive_json()["recording"], 1)
        time.sleep(0.2)
        r = self.client.post("/api/slot/1/record/stop")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["slot"]["loaded"])
        self.assertGreater(r.json()["slot"]["seconds"], 0.1)
        self.assertIsNone(self.client.get("/api/state").json()["recording"]["slot"])
        self.assertEqual(self.client.post("/api/slot/1/record/stop").status_code, 409)

    def test_manual_slot_selects_and_sets_manual_mode(self):
        r = self.client.post("/api/manual_slot", json={"idx": 2})
        self.assertEqual(r.json(), {"ok": True, "idx": 2})
        p = self.fake.GetParams()
        self.assertEqual((p["manual_slot"], p["emotion_mode"]), (2, "manual"))
        for bad in ({"idx": 10}, {"idx": True}, {}):
            self.assertEqual(self.client.post("/api/manual_slot", json=bad).status_code, 400, bad)

    def test_preset_changes_exactly_the_listed_controls(self):
        before = self.client.get("/api/state").json()["params"]
        r = self.client.post("/api/preset", json={"name": "Soft / Whisper"})
        self.assertEqual(r.status_code, 200)
        after = self.client.get("/api/state").json()["params"]
        changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        self.assertEqual(changed, {"emotion_mode": ("auto", "manual"), "manual_slot": (0, 2),
                                   "f0_expander": (1.0, 0.8)})

    def test_second_preset_changes_only_f0_expander(self):
        before = self.client.get("/api/state").json()["params"]
        self.assertEqual(self.client.post("/api/preset", json={"name": "High-Energy Gaming / Ragebait"}).status_code, 200)
        after = self.client.get("/api/state").json()["params"]
        self.assertEqual({k: (before[k], after[k]) for k in before if before[k] != after[k]},
                         {"f0_expander": (1.0, 1.3)})

    def test_truncated_fourth_preset_is_refused_and_applies_nothing(self):
        before = self.fake.GetParams()
        r = self.client.post("/api/preset", json={"name": "Ultra-Low-"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.fake.GetParams(), before)
        self.assertFalse(any(c[0] == "ApplyPreset" for c in self.fake.calls))
        self.assertEqual(self.client.post("/api/preset", json={"name": "Nope"}).status_code, 404)

    def test_devices_lists_fake_inputs(self):
        body = self.client.get("/api/devices").json()
        self.assertEqual(body["devices"], [{"id": 0, "name": "FAKE microphone", "hostapi": "FAKE",
                                            "default_samplerate": 48000.0, "max_input_channels": 1}])
        self.assertEqual(body["inputs"], body["devices"])
    def test_websocket_pushes_about_20_hz_with_telemetry_keys(self):
        with self.client.websocket_connect("/ws", headers=LOOPBACK) as ws:
            first = ws.receive_json()
            start = time.monotonic()
            for _ in range(10):
                ws.receive_json()
            elapsed = time.monotonic() - start
        self.assertEqual(sorted(first.keys()), ["levels", "recording", "t", "telemetry"])
        self.assertIs(first["telemetry"]["running"], True)
        self.assertTrue(0.2 < elapsed < 5.0, elapsed)  # 10 frames at 20 Hz take about 0.45 s

    def test_websocket_origin_check_refuses_foreign_and_accepts_loopback(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                    "/ws", headers={"Host": "127.0.0.1", "Origin": "http://evil.example"}) as ws:
                ws.receive_json()
        with self.client.websocket_connect(
                "/ws", headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1:8000"}) as ws:
            self.assertIn("telemetry", ws.receive_json())

    def test_foreign_host_and_origin_are_refused(self):
        self.assertEqual(self.client.get("/api/state", headers={"Host": "evil.example"}).status_code, 403)
        r = self.client.post("/api/param", json={"name": "cfg_rate", "value": 0.5},
                             headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.fake.params.Get("cfg_rate"), 0.65)

    def test_route_table_matches_contract(self):
        app = CreateApp(FakePipeline())
        got = set()
        for route in app.routes:
            if isinstance(route, APIWebSocketRoute):
                got.add((route.path, "WEBSOCKET"))
            elif isinstance(route, Mount):
                got.add((route.path, "MOUNT"))
            else:
                got.update((route.path, method) for method in route.methods if method != "HEAD")
        self.assertEqual(got, CONTRACT_ROUTES)

    def test_page_and_assets_are_local(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertNotIn("http://", r.text)
        self.assertNotIn("https://", r.text)
        self.assertIn('src="/static/app.js"', r.text)
        for asset in ("/static/app.js", "/static/style.css"):
            body = self.client.get(asset)
            self.assertEqual(body.status_code, 200, asset)
            self.assertNotIn("http://", body.text)
            self.assertNotIn("https://", body.text)


if __name__ == "__main__":
    unittest.main()
