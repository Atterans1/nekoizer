"""RvcEngine: an imported RVC voice (exported .pth) converting the live 16 kHz stream, behind the VoiceEngine interface.

Per block: the HuBERT content features and the RMVPE pitch of a window of audio (left context, the block, a 20 ms lookahead) are
computed, the synthesizer returns the last ret_frames of that window, and the region that SOLA expects is cut out at MODEL_SR.
Pitch shift is native (f0_up_key style); formant shift, F0 expander and energy follow go through the shared WORLD stage.
Not implemented: the .index retrieval (no faiss; the index is stored and reported only).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

import seedvc_params as P
from engine import rvc_checkpoint, rvc_synth, rvc_weights
from engine.voice_engine import KIND_RVC, PROSODY_CONTEXT_S, AllocatedMb, ConvertResult, MakeGrid, SilentResult, VoiceEngine

# ---- PUBLIC
UNUSED_CONTROLS = ("cfg_rate", "diffusion_steps", "timbre_mix", "prosody_follow")
HUBERT_HOP16 = 320  # 20 ms content frames at 16 kHz
FRAME16 = 160  # 10 ms synthesizer and pitch frames at 16 kHz
F0_MIN = 50.0
F0_MAX = 1100.0
RMVPE_THRESHOLD = 0.03


@dataclass(frozen=True)
class RvcPlan:
    win16: int  # window length: whole 20 ms content frames
    p_len: int  # 10 ms frames in the window
    ret_frames: int  # frames the synthesizer returns: the region plus the prosody context and lookahead
    skip: int  # frames of the window the synthesizer skips at its head
    ret_start16: int  # window index where the returned frames start
    region_start22: int  # index in the returned audio (MODEL_SR) where the SOLA region starts
    ctx_start16: int  # window index of the first returned frame (ret_start16)


def PlanRvc(block_ms: float, overlap_ms: float, left_context_s: float) -> RvcPlan:
    grid = MakeGrid(block_ms, overlap_ms)
    left16 = int(round(left_context_s * P.INPUT_SR))
    need = left16 + grid.overlap16 + grid.search16 + grid.block16 + grid.right16
    win16 = -(-need // HUBERT_HOP16) * HUBERT_HOP16
    out16 = grid.block16 + grid.overlap16 + grid.search16
    ctx16 = int(round(PROSODY_CONTEXT_S * P.INPUT_SR))
    ret_frames = -(-(out16 + grid.right16 + ctx16) // FRAME16)
    p_len = win16 // FRAME16
    ret_start16 = win16 - ret_frames * FRAME16
    region_start16 = win16 - grid.right16 - out16
    region_start22 = int(round((region_start16 - ret_start16) * P.MODEL_SR / P.INPUT_SR))
    return RvcPlan(win16, p_len, ret_frames, p_len - ret_frames, ret_start16, region_start22, ret_start16)


def CoarsePitch(f0_hz: np.ndarray) -> np.ndarray:
    """RVC's pitch code: 1..255 on the mel scale between F0_MIN and F0_MAX (0 Hz maps to 1)."""
    mel_min = 1127 * math.log(1 + F0_MIN / 700)
    mel_max = 1127 * math.log(1 + F0_MAX / 700)
    mel = 1127 * np.log(1 + f0_hz / 700)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64)


class RvcEngine(VoiceEngine):
    kind = KIND_RVC
    unused_controls = UNUSED_CONTROLS
    vram_estimate_mb = 700.0  # HuBERT, RMVPE and a v2 40k synthesizer, fp16

    def __init__(self, params, janitor, device: str = "cuda", weights_dir: Path | None = None, backbone=None, half: bool | None = None):
        """backbone: an object with Features(wave (1,N) tensor, version) -> (1,T,D) and Pitch(np wave16) -> Hz per 10 ms frame.
        None loads the real HuBERT and RMVPE (or the FCPE fallback) from weights_dir when Load() is called."""
        self._params = params
        self._device = torch.device(device)
        self._half = (self._device.type == "cuda") if half is None else bool(half)
        self._weights_dir = weights_dir
        self._backbone = backbone
        self._owns_backbone = backbone is None
        self._net = None
        self._voice: dict | None = None
        self._ring: np.ndarray | None = None
        self._prosody = None
        self._telemetry: dict = {}
        self._voice_load_s = 0.0
        janitor.Add(self.Release)

    # ---- PUBLIC
    def Prepare(self) -> dict:
        """The pinned HuBERT and RMVPE files, fetched once in the background. An injected backbone needs none."""
        if not self._owns_backbone:
            return {"state": "ready"}
        return {**rvc_weights.Ensure(weights_dir=self._weights_dir), "label": "RVC base models"}

    def Load(self) -> dict:
        started = time.perf_counter()
        before = AllocatedMb(self._device)
        if self._backbone is None:
            from engine.rvc_backbone import LoadBackbone
            self._backbone = LoadBackbone(self._weights_dir, self._device, self._half)
        return {"backbone": type(self._backbone).__name__, "vram_mb": AllocatedMb(self._device) - before,
                "load_seconds": time.perf_counter() - started}

    def LoadVoice(self, voice: dict) -> dict:
        """Load the synthesizer of a stored voice (its model_path comes from engine.rvc_store.Get). Replaces the previous voice."""
        started = time.perf_counter()
        if self._backbone is None:
            self.Load()
        self._net = None
        checkpoint = rvc_checkpoint.LoadCheckpoint(voice["model_path"], max_bytes=voice.get("max_bytes"))
        self._net = rvc_synth.BuildSynthesizer(checkpoint, self._device, self._half)
        self._voice = {"name": voice["name"], "version": checkpoint.version, "sr": checkpoint.sr, "f0": checkpoint.f0,
                       "has_index": bool(voice.get("has_index", False))}
        self._ring = None
        self._voice_load_s = time.perf_counter() - started
        return dict(self._voice)

    def Voice(self) -> dict | None:
        return None if self._voice is None else dict(self._voice)

    def UnloadVoice(self) -> None:
        self._net = None
        self._voice = None
        self._ring = None

    def Release(self) -> None:
        self._net = None
        self._voice = None
        self._ring = None
        self._prosody = None
        if self._owns_backbone:
            self._backbone = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def Loaded(self) -> bool:
        return self._backbone is not None and self._net is not None

    def ResetStream(self) -> None:
        self._ring = None
        if self._prosody is not None:
            self._prosody.Reset()

    def Telemetry(self) -> dict:
        return {**self._telemetry, "voice": self.Voice(), "half": self._half, "voice_load_s": self._voice_load_s,
                "pitch": getattr(self._backbone, "pitch_name", None)}

    def ConvertBlock(self, block16k: np.ndarray, active: bool, steps: int | None = None) -> ConvertResult:
        grid = MakeGrid(self._params.Get("block_ms"), self._params.Get("overlap_ms"))
        block = np.ascontiguousarray(block16k, dtype=np.float32)
        if block.shape != (grid.block16,):
            raise ValueError(f"block16k must hold {grid.block16} samples, got {block.shape}")
        if self._net is None or self._backbone is None:
            raise RuntimeError("no RVC voice is loaded")
        p = self._params.Snapshot()
        plan = PlanRvc(p["block_ms"], p["overlap_ms"], p["left_context_s"])
        if self._ring is None or len(self._ring) != plan.win16:
            self._ring = np.zeros(plan.win16, np.float32)
        self._ring[:-grid.block16] = self._ring[grid.block16:]
        self._ring[-grid.block16:] = block
        if not active:
            return SilentResult(grid.out22)
        started = time.perf_counter()
        wave = torch.from_numpy(self._ring).to(self._device).unsqueeze(0)
        with torch.no_grad():
            feats = self._backbone.Features(wave, self._voice["version"]).float()
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            feats = torch.nn.functional.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if feats.shape[1] < plan.p_len:
                feats = torch.cat((feats, feats[:, -1:, :].expand(1, plan.p_len - feats.shape[1], -1)), 1)
            feats = feats[:, :plan.p_len]
            t_feat = time.perf_counter()
            phone = feats.half() if self._half else feats
            lengths = torch.LongTensor([plan.p_len]).to(self._device)
            sid = torch.LongTensor([0]).to(self._device)
            if self._voice["f0"]:
                f0 = self._backbone.Pitch(self._ring)[: plan.p_len].astype(np.float64)
                f0_voiced_median, f0_std = self._PitchStats(f0)
                f0 = self._Shift(f0, p["pitch_shift_st"])
                coarse = torch.from_numpy(CoarsePitch(f0.copy())).unsqueeze(0).to(self._device)
                pitchf = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0).to(self._device)
                out = self._net.infer(phone, lengths, coarse, pitchf, sid, plan.skip, plan.ret_frames, plan.ret_frames)[0]
            else:
                f0_voiced_median, f0_std = 0.0, 0.0
                out = self._net.infer(phone, lengths, sid, plan.skip, plan.ret_frames, plan.ret_frames)[0]
            y_tgt = out.reshape(-1).float().cpu().numpy()
            t_synth = time.perf_counter()
        g = math.gcd(P.MODEL_SR, self._voice["sr"])
        y22 = resample_poly(y_tgt, P.MODEL_SR // g, self._voice["sr"] // g).astype(np.float32)
        seg_start = plan.region_start22
        seg_end = seg_start + grid.out22
        if seg_end > len(y22):
            raise RuntimeError(f"the synthesizer returned {len(y22)} samples, the region ends at {seg_end}")
        region = y22[seg_start:seg_end]
        stage_applied = False
        if p["formant_shift"] != 0.0 or p["f0_expander"] != 1.0 or p["energy_follow"] > 0.0:
            region, stage_applied = self._Stage(y22, plan, grid, p, region)
        out22 = np.clip(region, -1.0, 1.0).astype(np.float32)
        done = time.perf_counter()
        self._telemetry = {"features_ms": (t_feat - started) * 1000.0, "synth_ms": (t_synth - t_feat) * 1000.0,
                           "post_ms": (done - t_synth) * 1000.0, "inference_s_last": done - started, "win16": plan.win16,
                           "ret_frames": plan.ret_frames}
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) + 1e-9)
        return ConvertResult(out22, done - started, 0, [0.0] * 4, f0_voiced_median, f0_std, rms_db, stage_applied)

    # ---- PRIVATE
    @staticmethod
    def _Shift(f0: np.ndarray, semitones: float) -> np.ndarray:
        voiced = f0 > 0.0
        if not voiced.any():
            return f0
        out = f0.copy()
        out[voiced] = f0[voiced] * 2.0 ** (semitones / 12.0)
        return out

    @staticmethod
    def _PitchStats(f0: np.ndarray) -> tuple[float, float]:
        voiced = f0[f0 > 0.0]
        if voiced.size == 0:
            return 0.0, 0.0
        log_st = 12.0 * np.log2(voiced)
        return float(np.median(voiced)), float(np.std(log_st))

    def _Stage(self, y22, plan, grid, p, region):
        from engine.prosody_stage import FcpeF0, ProsodyStage, SourceAudio
        if self._prosody is None:
            self._prosody = ProsodyStage(FcpeF0(), P.MODEL_SR)
        ctx = grid.prosody_ctx22
        begin = plan.region_start22 - ctx
        if begin < 0:
            return region, False
        segment = y22[begin: plan.region_start22 + grid.out22]
        source = None
        if p["energy_follow"] > 0.0:
            t0 = plan.ctx_start16 + begin * P.INPUT_SR / P.MODEL_SR
            source = SourceAudio(self._ring, float(t0))
        out, applied = self._prosody.Process(segment, grid.out22, grid.block22, 0.0, p["formant_shift"], p["f0_expander"],
                                             0.0, p["energy_follow"], source)
        return out, applied

