import math
import os
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seedvc_params as P
import audio.output_chain as oc

TI = types.SimpleNamespace(currentTime=5.0, outputBufferDacTime=5.03, inputBufferAdcTime=0.0)  # DAC 30 ms after the callback
CHUNK = 6284  # 220 ms block + 45 ms overlap + 20 ms search at 22050 Hz


def Sine(n, start=0):
    return (0.5 * np.sin(2 * np.pi * 440.0 * (start + np.arange(n)) / 22050.0)).astype(np.float32)


class FakeStream:
    def __init__(self, kw):
        self.kw, self.callback, self.latency, self.events = kw, kw["callback"], 0.012, []

    def start(self):
        self.events.append("start")

    def abort(self):
        self.events.append("abort")

    def close(self):
        self.events.append("close")


class FakeSd:
    """Stands in for sounddevice inside these tests only."""

    def __init__(self, channels=2, fail_second_stream=False):
        self.hostapis = [{"name": "MME", "default_output_device": 0}, {"name": "Windows WASAPI", "default_output_device": 3}]
        self.devices = [
            {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
            {"name": "CABLE In 16ch (VB-Audio Virtual Cable)", "hostapi": 1, "max_output_channels": 16, "default_samplerate": 48000.0},
            {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 1, "max_output_channels": channels, "default_samplerate": 48000.0},
            {"name": "Speaker (2- Realtek(R) Audio)", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000.0},
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "hostapi": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        ]
        self.streams, self.fail_second_stream = [], fail_second_stream

    def query_hostapis(self, index=None):
        return self.hostapis if index is None else self.hostapis[index]

    def query_devices(self):
        return self.devices

    def WasapiSettings(self, **kw):
        return types.SimpleNamespace(**kw)

    def OutputStream(self, **kw):
        if self.fail_second_stream and self.streams:
            raise RuntimeError("busy")
        self.streams.append(FakeStream(kw))
        return self.streams[-1]


class FakeMonitorRing:
    samplerate = 48000

    def __init__(self, short_by=0):
        self.reads, self.short_by = 0, short_by

    def Read(self, n):
        self.reads += 1
        return np.full(n - self.short_by, 0.5, dtype=np.float32)


def Pump(stream, frames, count, ti=TI):
    outs = []
    for _ in range(count):
        out = np.full((frames, stream.kw["channels"]), 7.0, dtype=np.float32)
        stream.callback(out, frames, ti, None)
        outs.append(out)
    return outs


class ChainCase(unittest.TestCase):
    def Make(self, channels=2, fail_second_stream=False, **kw):
        self.fake = FakeSd(channels, fail_second_stream)
        patcher = mock.patch.object(oc, "sd", self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.params, self.janitor = P.ParamStore(), P.Janitor()
        chain = oc.OutputChain(self.params, self.janitor, **kw)
        self.addCleanup(chain.Stop)
        return chain


class DeviceTests(unittest.TestCase):
    def test_wasapi_name_resolution_and_the_16ch_ban(self):
        with mock.patch.object(oc, "sd", FakeSd()):
            cable = oc.ResolveWasapiOutput("CABLE Input", forbid=("16ch",))
            self.assertEqual((cable["index"], cable["channels"], cable["samplerate"]), (2, 2, 48000))  # not the MME duplicate at 0
            with self.assertRaises(RuntimeError):
                oc.ResolveWasapiOutput("CABLE In 16ch", forbid=("16ch",))
            sixteen = oc.ResolveWasapiOutput("CABLE In 16ch")
            self.assertEqual((sixteen["index"], sixteen["channels"]), (1, 16))
            self.assertEqual(oc.ResolveWasapiOutput(None)["name"], "Speaker (2- Realtek(R) Audio)")
            with self.assertRaises(RuntimeError):
                oc.ResolveWasapiOutput("CABLE Output")  # capture-only endpoint

    def test_main_stream_refuses_the_16ch_device_and_opens_nothing(self):
        fake = FakeSd()
        with mock.patch.object(oc, "sd", fake):
            chain = oc.OutputChain(P.ParamStore(), P.Janitor(), device_name="CABLE In 16ch")
            with self.assertRaises(RuntimeError):
                chain.Start()
        self.assertEqual(fake.streams, [])


class FifoTests(unittest.TestCase):
    def test_wrap_around_keeps_frame_order_and_fills_every_channel(self):
        ring = oc._Fifo(8, 2)
        ring.Write(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        out = np.zeros((5, 2), dtype=np.float32)
        ring.Read(out, 5)
        np.testing.assert_array_equal(out, np.repeat([[1], [2], [3], [4], [5]], 2, axis=1))
        ring.Write(np.array([10, 11, 12, 13, 14, 15], dtype=np.float32))  # frames 5..10 wrap past the end
        out = np.zeros((6, 2), dtype=np.float32)
        ring.Read(out, 6)
        np.testing.assert_array_equal(out[:, 0], [10, 11, 12, 13, 14, 15])
        self.assertEqual((ring.w, ring.r), (11, 11))

    def test_pad_writes_exact_zeros_before_the_block(self):
        ring = oc._Fifo(8, 1)
        ring.ring.buf[:] = 9.0
        ring.Write(np.array([1, 2], dtype=np.float32), pad=3)
        out = np.zeros((5, 1), dtype=np.float32)
        ring.Read(out, 5)
        np.testing.assert_array_equal(out[:, 0], [0, 0, 0, 1, 2])


class OutputChainTests(ChainCase):
    def test_start_opens_one_wasapi_stream_on_cable_input_at_the_device_format(self):
        chain = self.Make()
        chain.Start()
        self.assertEqual(len(self.fake.streams), 1)
        kw = self.fake.streams[0].kw
        self.assertEqual((kw["device"], kw["samplerate"], kw["channels"], kw["dtype"], kw["latency"]), (2, 48000, 2, "float32", "low"))
        self.assertEqual(self.fake.streams[0].events, ["start"])
        t = chain.Telemetry()
        self.assertEqual((t["device"], t["samplerate"], t["channels"]), ("CABLE Input (VB-Audio Virtual Cable)", 48000, 2))

    def test_telemetry_has_every_contract_key(self):
        chain = self.Make()
        chain.Start()
        t = chain.Telemetry()
        self.assertLessEqual({"device", "samplerate", "underruns", "buffer_ms", "e2e_ms_last", "e2e_ms_mean", "e2e_ms_max", "blocks_out", "monitor", "balancer"}, set(t))
        self.assertLessEqual({"enabled", "device", "buffer_ms"}, set(t["monitor"]))

    def test_cushion_then_audio_then_underrun_zero_fill(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(t_capture=0.0, block_ms=220))
        self.assertAlmostEqual(chain.Telemetry()["buffer_ms"], 259.5625, places=6)  # 1920 cushion frames + 10539 resampled frames
        outs = Pump(self.fake.streams[0], 480, 12)
        for out in outs[:4]:
            self.assertEqual(np.count_nonzero(out), 0)  # the 40 ms cushion is exact silence
        self.assertEqual(chain.Telemetry()["underruns"], 0)
        later = outs[11]  # 3360 frames after the cushion: past the 45 ms fade-in
        self.assertGreater(np.abs(later).max(), 0.4)
        np.testing.assert_array_equal(later[:, 0], later[:, 1])
        Pump(self.fake.streams[0], 480, 13)  # callbacks 13..25 drain the ring: 12459 = 25 * 480 + 459
        last = Pump(self.fake.streams[0], 480, 1)[0]  # 26th: only 459 frames left
        self.assertEqual(np.count_nonzero(last[459:]), 0)
        self.assertTrue(np.all(last[:459] != 7.0))
        empty = Pump(self.fake.streams[0], 480, 1)[0]  # 27th: ring empty
        self.assertEqual(np.count_nonzero(empty), 0)
        t = chain.Telemetry()
        self.assertEqual((t["underruns"], t["underrun_events"]), (2, 1))
        self.assertAlmostEqual(t["zero_filled_ms"], 10.4375, places=6)  # (21 + 480) frames at 48 kHz

    def test_after_an_underrun_the_next_block_gets_the_cushion_again(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        Pump(self.fake.streams[0], 480, 27)  # drain and starve
        chain.Submit(Sine(CHUNK, start=4851), oc.BlockMeta(0.0, 220))
        self.assertAlmostEqual(chain.Telemetry()["buffer_ms"], 260.0, places=6)  # 1920 + 10560 frames
        for out in Pump(self.fake.streams[0], 480, 4):
            self.assertEqual(np.count_nonzero(out), 0)

    def test_mono_goes_to_every_channel_of_a_three_channel_device(self):
        chain = self.Make(channels=3)
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        out = Pump(self.fake.streams[0], 480, 12)[11]
        self.assertEqual(out.shape, (480, 3))
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        np.testing.assert_array_equal(out[:, 1], out[:, 2])
        self.assertGreater(np.abs(out).max(), 0.4)

    def test_e2e_is_dac_time_of_the_first_sample_minus_t_capture(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(t_capture=99.8, block_ms=220))
        with mock.patch.object(oc, "time", types.SimpleNamespace(perf_counter=lambda: 100.0)):
            Pump(self.fake.streams[0], 700, 3)  # the block starts at frame 1920 = 520 frames into the third callback
        t = chain.Telemetry()
        # (100.0 + 0.030 DAC delta + 520/48000) - 99.8 = 0.240833 s
        for key in ("e2e_ms_last", "e2e_ms_mean", "e2e_ms_max"):
            self.assertAlmostEqual(t[key], 240.8333333, places=4)
        self.assertEqual((t["blocks_out"], t["e2e_blocks"], t["e2e_invalid"]), (1, 1, 0))

    def test_an_implausible_dac_time_is_counted_not_used(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(99.8, 220))
        Pump(self.fake.streams[0], 700, 3, ti=types.SimpleNamespace(currentTime=5.0, outputBufferDacTime=0.0))
        t = chain.Telemetry()
        self.assertEqual((t["e2e_blocks"], t["e2e_invalid"], t["e2e_ms_last"]), (0, 1, 0.0))

    def test_silence_blocks_are_exact_zeros_and_stay_out_of_the_latency_stats(self):
        chain = self.Make()
        chain.Start()
        chain.SubmitSilence(oc.BlockMeta(t_capture=1.0, block_ms=220, active=False))
        for out in Pump(self.fake.streams[0], 480, 25):
            self.assertEqual(np.count_nonzero(out), 0)
        t = chain.Telemetry()
        self.assertEqual((t["blocks_out"], t["e2e_blocks"], t["underruns"]), (1, 0, 0))

    def test_block_ms_change_rebuilds_sola_through_the_param_store(self):
        chain = self.Make()
        chain.Start()
        with self.assertRaises(ValueError):
            chain.Submit(np.zeros(5402, dtype=np.float32), oc.BlockMeta(0.0, 220))  # 180 ms chunk, still configured for 220 ms
        self.params.Set("block_ms", 180)
        chain.Submit(np.zeros(5402, dtype=np.float32), oc.BlockMeta(0.0, 180))  # 3969 + 992 + 441 samples
        self.assertAlmostEqual(chain.Telemetry()["buffer_ms"], 219.5625, places=6)  # 1920 + 8619 frames

    def test_submit_needs_a_started_chain(self):
        chain = self.Make()
        with self.assertRaises(RuntimeError):
            chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        chain.Start()
        chain.Stop()
        with self.assertRaises(RuntimeError):
            chain.SubmitSilence(oc.BlockMeta(0.0, 220, False))

    def test_stop_is_idempotent_creates_no_threads_and_the_janitor_stops_the_chain(self):
        chain = self.Make()
        threads = threading.active_count()
        chain.Start()
        chain.Start()
        self.assertEqual(len(self.fake.streams), 1)
        chain.Stop()
        chain.Stop()
        self.assertEqual(self.fake.streams[0].events, ["start", "abort", "close"])
        self.assertEqual(threading.active_count(), threads)
        chain.Start()
        self.janitor.Cleanup()
        self.assertEqual(self.fake.streams[1].events, ["start", "abort", "close"])

    def test_a_failing_callback_outputs_zeros_and_is_counted(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        bad = np.full((480, 1), 7.0, dtype=np.float32)  # one column where the ring has two
        self.fake.streams[0].callback(bad, 480, TI, None)
        self.assertEqual(np.count_nonzero(bad), 0)
        self.assertEqual(chain.Telemetry()["callback_errors"], 1)

    def test_portaudio_output_underflow_flag_is_counted(self):
        chain = self.Make()
        chain.Start()
        chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        out = np.zeros((480, 2), dtype=np.float32)
        self.fake.streams[0].callback(out, 480, TI, types.SimpleNamespace(output_underflow=True))
        self.fake.streams[0].callback(out, 480, TI, types.SimpleNamespace(output_underflow=False))
        self.assertEqual(chain.Telemetry()["device_underflows"], 1)

    def test_balancer_is_wired_through_the_chain(self):
        chain = self.Make()
        self.assertEqual(chain.EffectiveSteps(10), 10)
        chain.ReportInference(0.22, 0.22)  # ratio 1.0 on the first update
        self.assertEqual(chain.EffectiveSteps(10), 8)
        self.assertTrue(chain.Telemetry()["balancer"]["throttled"])


class MonitorTests(ChainCase):
    def test_monitor_plays_the_ring_times_volume_only_when_enabled_both_read_live(self):
        ring = FakeMonitorRing()
        chain = self.Make(monitor_ring=ring)
        chain.Start()
        self.assertEqual(len(self.fake.streams), 2)
        mon = self.fake.streams[1]
        self.assertEqual((mon.kw["device"], mon.kw["samplerate"], mon.kw["channels"], mon.kw["latency"]), (3, 48000, 2, "low"))
        off = Pump(mon, 480, 1)[0]
        self.assertEqual(np.count_nonzero(off), 0)
        self.assertEqual(ring.reads, 1)  # drained even while disabled
        self.params.Set("monitor_enabled", True)
        np.testing.assert_allclose(Pump(mon, 480, 1)[0], 0.25, atol=1e-7)  # 0.5 ring level * default volume 0.5
        self.params.Set("monitor_volume", 0.1)
        np.testing.assert_allclose(Pump(mon, 480, 1)[0], 0.05, atol=1e-7)
        m = chain.Telemetry()["monitor"]
        self.assertEqual((m["enabled"], m["device"], m["samplerate"], m["channels"]), (True, "Speaker (2- Realtek(R) Audio)", 48000, 2))
        self.assertAlmostEqual(m["buffer_ms"], 10.0, places=6)
        self.assertAlmostEqual(m["delay_ms"], 30.0, places=3)  # measured DAC delta of the monitor callbacks

    def test_monitor_device_by_name_and_every_channel_is_filled(self):
        chain = self.Make(monitor_ring=FakeMonitorRing(), monitor_device_name="CABLE In 16ch")
        self.params.Set("monitor_enabled", True)
        chain.Start()
        mon = self.fake.streams[1]
        self.assertEqual((mon.kw["device"], mon.kw["channels"]), (1, 16))
        out = Pump(mon, 480, 1)[0]
        self.assertEqual(out.shape, (480, 16))
        np.testing.assert_allclose(out, 0.25, atol=1e-7)
        self.assertEqual(chain.Telemetry()["monitor"]["device"], "CABLE In 16ch (VB-Audio Virtual Cable)")

    def test_short_ring_reads_are_zero_padded_and_counted(self):
        chain = self.Make(monitor_ring=FakeMonitorRing(short_by=10))
        self.params.Set("monitor_enabled", True)
        chain.Start()
        out = Pump(self.fake.streams[1], 480, 1)[0]
        self.assertEqual(np.count_nonzero(out[470:]), 0)
        np.testing.assert_allclose(out[:470], 0.25, atol=1e-7)
        self.assertEqual(chain.Telemetry()["monitor"]["short_reads"], 1)

    def test_monitor_failure_is_reported_and_the_main_output_keeps_working(self):
        chain = self.Make(monitor_ring=FakeMonitorRing(), fail_second_stream=True)
        chain.Start()
        m = chain.Telemetry()["monitor"]
        self.assertEqual((m["device"], m["error"]), (None, "RuntimeError: busy"))
        chain.Submit(Sine(CHUNK), oc.BlockMeta(0.0, 220))
        self.assertGreater(chain.Telemetry()["buffer_ms"], 250.0)

    def test_no_monitor_stream_without_a_ring(self):
        chain = self.Make()
        chain.Start()
        self.assertEqual(len(self.fake.streams), 1)
        self.assertIsNone(chain.Telemetry()["monitor"]["device"])


if __name__ == "__main__":
    unittest.main()
