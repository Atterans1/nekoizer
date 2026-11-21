import math
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import seedvc_params as P
from audio import input_chain as ic
from audio.dsp import HighPass, SoftCompressor
from audio.vad import SileroVad

FRAME = P.VAD_FRAME_SAMPLES
BLOCK = 3520   # 220 ms at 16 kHz


def Speech16k():
    x, sr = sf.read(os.path.join(ROOT, "testdata", "ref_1.wav"), dtype="float32")
    g = math.gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def Sine(freq, amp, seconds, rate=48000):
    t = np.arange(int(round(seconds * rate))) / rate
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def SplitRuns(x, sizes):
    out, pos, i = [], 0, 0
    while pos < x.size:
        out.append(x[pos:pos + sizes[i % len(sizes)]])
        pos += sizes[i % len(sizes)]
        i += 1
    return out


class FakeFlags:
    def __init__(self, input_overflow=False):
        self.input_overflow = input_overflow

    def __bool__(self):
        return self.input_overflow


class FakeStream:
    """Stands in for sounddevice.InputStream: Feed() calls the PortAudio callback synchronously."""

    def __init__(self, kw):
        self.kw = kw
        self.latency = 0.012
        self.started = self.closed = False
        self.base = 1000.0     # stream clock at the first callback
        self.calls = 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def Feed(self, samples, frames=480, adc_offset=-0.012, overflow=False):
        x = np.asarray(samples, np.float32)
        ch, rate = self.kw["channels"], self.kw["samplerate"]
        if x.ndim == 1:
            x = np.repeat(x[:, None], ch, axis=1)
        for i in range(0, len(x), frames):
            piece = np.ascontiguousarray(x[i:i + frames])
            cur = self.base + self.calls * frames / rate
            self.calls += 1
            info = types.SimpleNamespace(inputBufferAdcTime=cur + adc_offset, currentTime=cur, outputBufferDacTime=0.0)
            self.kw["callback"](piece, len(piece), info, FakeFlags(overflow))


class FakeSd:
    """The slice of sounddevice that InputChain uses, with a MME mic, a CABLE and a mic on the WASAPI host API."""

    class PortAudioError(RuntimeError):
        pass

    def __init__(self, rate=48000.0, channels=2, exclusive_check_ok=True, exclusive_open_ok=True, raw_supported=True,
                 default_is_cable=True, mic_present=True):
        self.PortAudioError = FakeSd.PortAudioError
        self.exclusive_check_ok, self.exclusive_open_ok = exclusive_check_ok, exclusive_open_ok
        self.rate, self.channels = rate, channels
        self._lib = types.SimpleNamespace(eStreamOptionRaw=1) if raw_supported else types.SimpleNamespace()
        self.devices = [
            {"name": "Microphone Array (MME)", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 44100.0},
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0},
            {"name": "Headphones", "hostapi": 1, "max_input_channels": 0, "default_samplerate": 48000.0},
        ]
        if mic_present:
            self.devices.append({"name": "Microphone Array (2- Intel Smart Sound)", "hostapi": 1, "max_input_channels": channels,
                                 "default_samplerate": rate})
        self.default_input = 1 if default_is_cable else 3
        self.streams = []
        self.checks = []

    def query_hostapis(self, index=None):
        apis = [{"name": "MME", "default_input_device": 0}, {"name": "Windows WASAPI", "default_input_device": self.default_input}]
        return apis if index is None else apis[index]

    def query_devices(self, device=None):
        return list(self.devices) if device is None else self.devices[device]

    def WasapiSettings(self, exclusive=False, auto_convert=False, explicit_sample_format=False):
        return types.SimpleNamespace(exclusive=exclusive, _streaminfo=types.SimpleNamespace(streamOption=0))

    def check_input_settings(self, device, channels, dtype, samplerate, extra_settings):
        self.checks.append((device, channels, samplerate, extra_settings.exclusive))
        if extra_settings.exclusive and (not self.exclusive_check_ok or samplerate != self.rate):
            raise FakeSd.PortAudioError("Invalid sample rate [PaErrorCode -9997]")

    def InputStream(self, **kw):
        if kw["extra_settings"].exclusive and not self.exclusive_open_ok:
            raise FakeSd.PortAudioError("Exclusive mode not allowed [Windows WASAPI error -2004287474]")
        stream = FakeStream(kw)
        self.streams.append(stream)
        return stream


class MonitorRingTests(unittest.TestCase):
    def test_write_then_read_returns_the_same_samples(self):
        ring = ic.MonitorRing(48000)
        x = np.arange(480, dtype=np.float32)
        ring.Write(x)
        self.assertTrue(np.array_equal(ring.Read(480), x))
        self.assertEqual((ring.underruns, ring.skips), (0, 0))

    def test_short_ring_is_zero_padded_and_counted(self):
        ring = ic.MonitorRing(48000)
        ring.Write(np.ones(100, np.float32))
        out = ring.Read(480)
        self.assertEqual((float(out[:100].sum()), float(out[100:].sum()), ring.underruns), (100.0, 0.0, 1))

    def test_wraps_around_the_end_of_the_ring(self):
        ring = ic.MonitorRing(48000)
        ring.Write(np.zeros(40000, np.float32))
        ring.Read(40000)
        x = np.arange(20000, dtype=np.float32)
        ring.Write(x)
        self.assertTrue(np.array_equal(ring.Read(20000), x))

    def test_a_reader_far_behind_skips_to_the_newest_20ms(self):
        ring = ic.MonitorRing(48000)
        x = np.arange(24000, dtype=np.float32)
        ring.Write(x)
        out = ring.Read(480)
        self.assertEqual(ring.skips, 1)
        self.assertTrue(np.array_equal(out, x[24000 - 960:24000 - 960 + 480]))

    def test_configure_changes_rate_and_clears(self):
        ring = ic.MonitorRing(48000)
        ring.Write(np.ones(100, np.float32))
        ring.Configure(44100)
        self.assertEqual((ring.sample_rate, float(ring.Read(10).sum())), (44100, 0.0))


class ChainTestBase(unittest.TestCase):
    def NewChain(self, fake=None, device_name="Microphone Array"):
        fake = fake or FakeSd()
        patcher = mock.patch.object(ic, "sd", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        janitor = P.Janitor()
        chain = ic.InputChain(P.ParamStore(), janitor, device_name)
        self.addCleanup(chain.Stop)
        return chain, fake, janitor

    def WaitFor(self, condition, timeout=5.0):
        deadline = time.perf_counter() + timeout
        while not condition() and time.perf_counter() < deadline:
            time.sleep(0.001)
        self.assertTrue(condition())

    def FeedAll(self, chain, stream, x, piece_s=0.5):
        step = int(piece_s * stream.kw["samplerate"])
        for i in range(0, len(x), step):
            stream.Feed(x[i:i + step])
            self.WaitFor(lambda: chain._r == chain._w)
        time.sleep(0.1)   # the worker may still be inside its last chunk

    def ReadAll(self, chain, n=BLOCK):
        blocks = []
        while True:
            block = chain.ReadBlock(n, 0.3)
            if block is None:
                return blocks
            blocks.append(block)


class NegotiationTests(ChainTestBase):
    def test_list_devices_has_only_wasapi_inputs_with_the_contract_keys(self):
        chain, _, _ = self.NewChain()
        devices = chain.ListDevices()
        self.assertEqual([d["name"] for d in devices], ["CABLE Output (VB-Audio Virtual Cable)", "Microphone Array (2- Intel Smart Sound)"])
        self.assertEqual(sorted(devices[0]), ["default_samplerate", "hostapi", "id", "max_input_channels", "name"])
        self.assertEqual({d["hostapi"] for d in devices}, {"Windows WASAPI"})

    def test_no_device_name_skips_a_cable_default(self):
        chain, _, _ = self.NewChain(device_name=None)
        chain.Start()
        self.assertEqual(chain.Telemetry()["device"], "Microphone Array (2- Intel Smart Sound)")

    def test_unknown_name_or_cable_only_system_is_an_error(self):
        chain, _, _ = self.NewChain(device_name="Nope")
        with self.assertRaisesRegex(RuntimeError, "no WASAPI input device matches"):
            chain.Start()
        chain, _, _ = self.NewChain(FakeSd(mic_present=False), device_name=None)
        with self.assertRaisesRegex(RuntimeError, "no usable WASAPI microphone"):
            chain.Start()

    def test_exclusive_is_tried_first_at_the_devices_native_format(self):
        chain, fake, _ = self.NewChain(FakeSd(rate=44100.0, channels=1))
        chain.Start()
        t, d = chain.Telemetry(), chain.Diagnostics()
        self.assertEqual((t["mode"], t["samplerate"], t["enhancements_bypassed"]), ("exclusive", 44100, True))
        self.assertEqual((d["channels"], d["raw_requested"], d["open_errors"]), (1, False, {}))
        self.assertEqual(fake.checks[0], (3, 1, 44100, True))
        self.assertTrue(fake.streams[0].kw["extra_settings"].exclusive)
        self.assertEqual(chain.MonitorRing.sample_rate, 44100)

    def test_refused_exclusive_falls_back_to_shared_lowlatency_with_the_raw_option(self):
        chain, fake, _ = self.NewChain(FakeSd(exclusive_check_ok=False))
        chain.Start()
        t, d = chain.Telemetry(), chain.Diagnostics()
        self.assertEqual((t["mode"], t["enhancements_bypassed"]), ("shared_lowlatency", None))
        self.assertTrue(d["raw_requested"])
        self.assertIn("Invalid sample rate", d["open_errors"]["exclusive"])
        kw = fake.streams[0].kw
        self.assertFalse(kw["extra_settings"].exclusive)
        self.assertEqual(kw["extra_settings"]._streaminfo.streamOption, 1)
        self.assertEqual((kw["latency"], kw["blocksize"], kw["dtype"]), ("low", 0, "float32"))

    def test_exclusive_open_error_text_is_kept_when_the_device_is_busy(self):
        chain, _, _ = self.NewChain(FakeSd(exclusive_open_ok=False))
        chain.Start()
        self.assertEqual(chain.Telemetry()["mode"], "shared_lowlatency")
        self.assertIn("Exclusive mode not allowed", chain.Diagnostics()["open_errors"]["exclusive"])

    def test_shared_without_the_raw_option_reports_it_and_does_not_claim_bypass(self):
        chain, _, _ = self.NewChain(FakeSd(exclusive_check_ok=False, raw_supported=False))
        chain.Start()
        self.assertFalse(chain.Diagnostics()["raw_requested"])
        self.assertIsNone(chain.Telemetry()["enhancements_bypassed"])

    def test_telemetry_has_exactly_the_contract_keys(self):
        chain, _, _ = self.NewChain()
        self.assertIsNone(chain.Telemetry()["mode"])
        chain.Start()
        self.assertEqual(sorted(chain.Telemetry()), sorted(["mode", "device", "samplerate", "buffer_ms", "vad_prob", "vad_active",
                                                            "gain_reduction_db", "makeup_db", "overruns", "enhancements_bypassed"]))


class LifecycleTests(ChainTestBase):
    def test_stop_joins_the_worker_closes_the_stream_and_the_janitor_calls_stop(self):
        before = set(threading.enumerate())
        chain, fake, janitor = self.NewChain()
        chain.Start()
        self.assertIn("InputChainWorker", {t.name for t in threading.enumerate()})
        janitor.Cleanup()
        self.assertTrue(fake.streams[0].closed)
        self.assertIsNone(chain.ReadBlock(BLOCK, 0.01))
        chain.Stop()
        self.assertEqual(set(threading.enumerate()) - before, set())

    def test_read_block_times_out_and_is_released_by_stop(self):
        chain, _, _ = self.NewChain()
        self.assertIsNone(chain.ReadBlock(BLOCK, 0.01))   # not started
        chain.Start()
        with self.assertRaises(ValueError):
            chain.ReadBlock(0, 0.01)
        self.assertIsNone(chain.ReadBlock(BLOCK, 0.05))
        result = []
        reader = threading.Thread(target=lambda: result.append(chain.ReadBlock(BLOCK, 10.0)))
        reader.start()
        time.sleep(0.1)
        chain.Stop()
        reader.join(2.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(result, [None])

    def test_restart_gives_a_fresh_stream(self):
        chain, fake, _ = self.NewChain()
        chain.Start()
        chain.Stop()
        chain.Start()
        self.assertEqual((len(fake.streams), fake.streams[0].closed, fake.streams[1].closed), (2, True, False))

    def test_worker_failure_is_raised_by_read_block(self):
        chain, fake, _ = self.NewChain()
        chain.Start()
        chain._comp.Process = mock.Mock(side_effect=ValueError("boom"))
        fake.streams[0].Feed(Sine(300.0, 0.1, 0.1))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            chain.ReadBlock(BLOCK, 2.0)


class SignalPathTests(ChainTestBase):
    def setUp(self):
        self.x16 = Speech16k()
        self.x48 = resample_poly(self.x16, 3, 1).astype(np.float32)   # 312000 samples = 6.5 s at 48 kHz

    def Offline(self, mono48k):
        comp = SoftCompressor(48000).Process(HighPass(48000).Process(mono48k))
        y16 = resample_poly(comp, 1, 3).astype(np.float32)
        vad = SileroVad()
        return np.concatenate([vad.Gate(y16[i:i + FRAME])[0] for i in range(0, (y16.size // FRAME) * FRAME, FRAME)])

    def test_speech_gives_29_exact_blocks_with_leading_silence_gated_to_exact_zeros(self):
        # 312000 samples -> (312000 - 30) // 3 = 103990 at 16 kHz -> 203 VAD frames -> 203 * 512 // 3520 = 29 blocks
        # the test writes 6.5 s before reading; the product ring (6.0 s) drops the oldest audio by design, so enlarge it here
        self.enterContext(mock.patch.object(ic, "OUTPUT_RING_S", 10.0))
        chain, fake, _ = self.NewChain()
        chain.Start()
        self.FeedAll(chain, fake.streams[0], self.x48)
        blocks = self.ReadAll(chain)
        self.assertEqual(len(blocks), 29)
        for b in blocks:
            self.assertEqual((b.samples16k.shape, b.samples16k.dtype), ((BLOCK,), np.dtype(np.float32)))
        first = blocks[0]
        self.assertTrue(first.active)
        self.assertTrue(np.all(first.samples16k[:2 * FRAME] == 0.0))
        self.assertTrue(np.any(first.samples16k[2 * FRAME:] != 0.0))
        self.assertTrue(-60.0 < first.rms_db < 0.0 and 0.0 < first.peak <= 1.0)

    def test_stream_output_matches_the_offline_hpf_compressor_resample_vad_chain(self):
        self.enterContext(mock.patch.object(ic, "OUTPUT_RING_S", 10.0))
        chain, fake, _ = self.NewChain()
        chain.Start()
        self.FeedAll(chain, fake.streams[0], self.x48)
        got = np.concatenate([b.samples16k for b in self.ReadAll(chain)])
        expected = self.Offline(self.x48)[:got.size]
        self.assertEqual(got.size, 29 * BLOCK)
        self.assertTrue(np.array_equal(got == 0.0, expected == 0.0))
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-5)

    def test_digital_silence_gives_9_inactive_blocks_of_exact_zeros(self):
        # 96000 samples -> 31990 at 16 kHz -> 62 frames -> 62 * 512 // 3520 = 9 blocks
        chain, fake, _ = self.NewChain()
        chain.Start()
        self.FeedAll(chain, fake.streams[0], np.zeros(2 * 48000, np.float32))
        blocks = self.ReadAll(chain)
        self.assertEqual(len(blocks), 9)
        for b in blocks:
            self.assertFalse(b.active)
            self.assertTrue(np.all(b.samples16k == 0.0))
            self.assertEqual((b.rms_db, b.peak), (-180.0, 0.0))

    def test_t_capture_advances_by_the_block_duration_with_or_without_a_usable_adc_time(self):
        for adc_offset, expect_fallback in ((-0.012, False), (0.010, True)):
            with self.subTest(adc_offset=adc_offset):
                chain, fake, _ = self.NewChain()
                chain.Start()
                stream = fake.streams[0]
                for _ in range(8):
                    stream.Feed(Sine(300.0, 0.1, 0.5), adc_offset=adc_offset)
                    self.WaitFor(lambda: chain._r == chain._w)
                time.sleep(0.1)
                times = [b.t_capture for b in self.ReadAll(chain)]
                self.assertGreaterEqual(len(times), 5)
                self.assertLess(float(np.max(np.abs(np.diff(times) - 0.22))), 1e-6)
                self.assertEqual(chain.Diagnostics()["timestamp_fallbacks"] > 0, expect_fallback)

    def test_monitor_ring_carries_the_post_compressor_pre_vad_signal_at_device_rate(self):
        tone = Sine(1000.0, 10.0 ** (-10.0 / 20.0), 0.5)
        chain, fake, _ = self.NewChain()
        chain.Start()
        self.FeedAll(chain, fake.streams[0], tone)
        self.WaitFor(lambda: chain.MonitorRing._w == tone.size)
        expected = SoftCompressor(48000).Process(HighPass(48000).Process(tone))
        self.assertTrue(np.array_equal(chain.MonitorRing.Read(tone.size), expected))
        # 8 ms attack lets the first peaks through and automatic makeup restores about 6 dB, so the peak sits near the input level
        # (0.3162) but well below input + the 9 dB makeup cap (0.89); it is not gated although the VAD sees a tone
        self.assertLess(float(np.max(np.abs(expected))), 0.36)
        self.assertGreater(float(np.max(np.abs(expected))), 0.25)

    def test_native_44k1_mono_is_resampled_to_exact_16k_blocks(self):
        chain, fake, _ = self.NewChain(FakeSd(rate=44100.0, channels=1))
        chain.Start()
        self.FeedAll(chain, fake.streams[0], Sine(300.0, 0.1, 3.0, rate=44100))
        blocks = self.ReadAll(chain)
        self.assertEqual(chain.Telemetry()["samplerate"], 44100)
        self.assertGreaterEqual(len(blocks), 8)
        self.assertTrue(all(b.samples16k.shape == (BLOCK,) for b in blocks))


class RecordingAndLevelTests(ChainTestBase):
    def test_recording_is_hpf_only_float32_mono_at_24khz(self):
        chain, fake, _ = self.NewChain()
        chain.Start()
        chain.StartRecording(10.0)
        self.assertEqual(chain._rec._max, 240000)
        self.FeedAll(chain, fake.streams[0], Sine(1000.0, 0.5, 0.5))
        rec = chain.StopRecording()
        self.assertEqual((rec.size, rec.dtype), (11990, np.dtype(np.float32)))   # 0.5 s at 24 kHz minus the 10-sample look-ahead
        self.assertAlmostEqual(float(np.max(np.abs(rec[-2400:]))), 0.5, delta=0.01)   # settled tail; a compressor would have pulled 0.5 down
        self.assertLess(float(np.max(np.abs(rec))), 0.56)                              # HPF switch-on overshoot is about 8 percent
        self.assertEqual(chain.StopRecording().size, 0)

    def test_recording_stops_growing_at_max_seconds_and_is_clamped_to_the_recorder_limit(self):
        chain, fake, _ = self.NewChain()
        chain.Start()
        chain.StartRecording(1.0)
        self.FeedAll(chain, fake.streams[0], Sine(1000.0, 0.5, 2.0))
        self.assertEqual(chain.StopRecording().size, 24000)
        chain.StartRecording(99.0)
        self.assertEqual(chain._rec._max, int(P.RECORDER_MAX_S * 24000))

    def test_start_recording_requires_a_started_chain(self):
        chain, _, _ = self.NewChain()
        with self.assertRaises(RuntimeError):
            chain.StartRecording()

    def test_level_snapshot_reports_peak_clipping_and_a_100_point_waveform(self):
        chain, fake, _ = self.NewChain()
        self.assertEqual(chain.LevelSnapshot(), {"peak": 0.0, "clipping": False, "waveform": [0.0] * 100})
        chain.Start()
        self.FeedAll(chain, fake.streams[0], np.clip(1.5 * Sine(440.0, 1.0, 0.6), -1.0, 1.0))
        snap = chain.LevelSnapshot()
        self.assertEqual((snap["peak"], snap["clipping"], len(snap["waveform"])), (1.0, True, 100))
        self.FeedAll(chain, fake.streams[0], Sine(440.0, 0.1, 0.6))
        snap = chain.LevelSnapshot()
        self.assertAlmostEqual(snap["peak"], 0.1, delta=0.0005)
        self.assertFalse(snap["clipping"])
        self.assertTrue(all(abs(v) <= 0.1005 for v in snap["waveform"]))


class OverrunTests(ChainTestBase):
    def test_portaudio_overflow_flags_and_a_full_capture_ring_are_counted(self):
        chain, fake, _ = self.NewChain()
        chain.Start()
        stream = fake.streams[0]
        for _ in range(3):
            stream.Feed(np.zeros(480, np.float32), overflow=True)
        self.WaitFor(lambda: chain._r == chain._w)
        self.assertEqual(chain.Diagnostics()["overruns_portaudio"], 3)
        stream.Feed(np.zeros(8 * 48000, np.float32))   # an 8 s burst into the 2 s capture ring
        d = chain.Diagnostics()
        self.assertGreater(d["overruns_capture_ring"], 0)
        self.assertEqual(chain.Telemetry()["overruns"], d["overruns_portaudio"] + d["overruns_capture_ring"] + d["overruns_output_ring"])

    def test_output_ring_drops_the_oldest_audio_when_nobody_reads(self):
        # 8 s -> 127990 samples at 16 kHz -> 249 frames = 127488 samples; the ring holds 6 s = 96000
        chain, fake, _ = self.NewChain()
        chain.Start()
        self.FeedAll(chain, fake.streams[0], np.zeros(8 * 48000, np.float32))
        self.WaitFor(lambda: chain._ow == 127488)
        d = chain.Diagnostics()
        self.assertEqual(d["dropped_16k_samples"], 127488 - 96000)
        self.assertGreaterEqual(d["overruns_output_ring"], 1)
        self.assertIsNotNone(chain.ReadBlock(BLOCK, 0.2))


@unittest.skipUnless(os.environ.get("SEEDVC_HW_MIC") == "1", "set SEEDVC_HW_MIC=1 to open the real microphone")
class HardwareMicTests(unittest.TestCase):
    def test_real_microphone_for_5_seconds_with_a_3_second_recording(self):
        before = sorted(t.name for t in threading.enumerate())
        janitor = P.Janitor()
        chain = ic.InputChain(P.ParamStore(), janitor, os.environ.get("SEEDVC_HW_DEVICE", "Microphone Array"))
        self.addCleanup(janitor.Cleanup)
        print("HW devices:", chain.ListDevices(), flush=True)
        chain.Start()
        print("HW telemetry at start:", chain.Telemetry(), flush=True)
        print("HW diagnostics at start:", chain.Diagnostics(), flush=True)
        t_start = time.perf_counter()
        chain.StartRecording()
        t_rec = time.perf_counter()
        rec, t_rec_stop, blocks = None, None, []
        while time.perf_counter() - t_start < 5.0:
            block = chain.ReadBlock(BLOCK, 1.0)
            self.assertIsNotNone(block)
            now = time.perf_counter()
            self.assertEqual(block.samples16k.shape, (BLOCK,))
            blocks.append((block.active, round(block.rms_db, 1), round(block.peak, 4), round((now - block.t_capture) * 1000, 1)))
            if rec is None and now - t_rec >= 3.0:
                rec, t_rec_stop = chain.StopRecording(), now
        print("HW blocks (active, rms_db, peak, age_ms):", blocks, flush=True)
        tel, diag = chain.Telemetry(), chain.Diagnostics()
        print("HW telemetry at end:", tel, flush=True)
        print("HW diagnostics at end:", diag, flush=True)
        print("HW recording: samples", rec.size, "dtype", rec.dtype, "seconds", round(rec.size / 24000, 3), "peak", round(float(np.abs(rec).max()), 5),
              "wall seconds", round(t_rec_stop - t_rec, 3), flush=True)
        print("HW level snapshot:", {k: v for k, v in chain.LevelSnapshot().items() if k != "waveform"}, flush=True)
        self.assertIsNone(diag["worker_error"])
        self.assertAlmostEqual(rec.size, (t_rec_stop - t_rec) * 24000, delta=2400)
        chain.Stop()
        after = sorted(t.name for t in threading.enumerate())
        print("HW threads before:", before, "after Stop:", after, flush=True)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
