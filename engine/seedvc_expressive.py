"""SeedVCExpressive: streaming Seed-VC (seed-uvit-tat-xlsr-tiny) with emotion conditioning, timbre mix and a WORLD prosody stage.

Adapted from third_party/seed-vc/real-time-gui.py: load_models (local assets, no network), custom_infer (prompt condition, XLSR,
length regulator, CFM, HiFT), the input-buffer arithmetic of start_vc and the output trim of audio_callback, generalised from
20 ms units to samples. Interfaces follow docs/seedvc/CONTRACT.md; deviations are listed in docs/seedvc/ENGINE_NOTES.md.
"""
from __future__ import annotations

import contextlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

from engine import compat
from engine.dit_graph import GraphedDiT
from engine.emotion import EmotionSelector, PitchSpreadTracker
from engine.prosody_stage import WORLD_LEAD_S, FcpeF0, ProsodyStage, SourceAudio
from engine.reference_bank import ReferenceBank
from engine.voice_engine import KIND_CLIP, PROSODY_CONTEXT_S, AllocatedMb, ConvertResult, VoiceEngine
from seedvc_params import (
    EXTRA_CONTEXT_RIGHT_S,
    INPUT_SR,
    MODEL_SR,
    SLOT_NAMES,
    SOLA_SEARCH_MS,
    Janitor,
    ParamStore,
    Validate,
)

# ---------------------------------------------------------------- PUBLIC
SEED_VC_DIR = Path(__file__).resolve().parents[1] / "third_party" / "seed-vc"
# Seed-VC real-time-gui.py GUIConfig.extra_time and its documented config: the DiT sees 0.5 s of left context, while the content
# encoder sees all of left_context_s (the GUI's extra_time_ce).
DIT_CONTEXT_S = 0.5
FCPE_CONTEXT_S = 0.2  # choice: extra input audio FCPE sees before the new block, so the block's first frames are not edge frames
LIVE_STYLE_REFRESH_BLOCKS = 5  # R16: the live speaker's CAMPPlus embedding is refreshed every few blocks (choice: 5)
LIVE_STYLE_MIN_S = 1.0  # choice: active speech needed in the left-context window before the live embedding is used
XLSR_FRAME = 320  # samples per content-encoder frame at 16 kHz
FCPE_HOP = 160  # samples per FCPE frame at 16 kHz (10 ms)


@dataclass(frozen=True)
class BlockPlan:
    """Sample arithmetic for one block (all derived from block_ms, overlap_ms, left_context_s and the constants)."""

    block16: int  # new input samples per block at 16 kHz
    ring16: int  # left-context ring length at 16 kHz: left + overlap + search + block + right
    ce_cut_frames: int  # content frames (20 ms) dropped before the length regulator: left_context_s minus the DiT context
    block22: int
    overlap22: int
    search22: int
    right22: int  # right-context (lookahead) samples at 22050 Hz
    out22: int  # block + overlap + search at 22050 Hz = length of ConvertResult.wav22k
    prosody_ctx22: int
    mel_frames: int  # mel frames the DiT generates after the prompt (DiT context + out + right)
    dit_context_s: float


def PlanBlock(block_ms: float, overlap_ms: float, left_context_s: float, hop: int = 256) -> BlockPlan:
    cut = int(math.floor((left_context_s - DIT_CONTEXT_S) * 50 + 1e-9))
    dit_context_s = left_context_s - cut / 50.0
    block_s, overlap_s, search_s = block_ms / 1000.0, overlap_ms / 1000.0, SOLA_SEARCH_MS / 1000.0
    block22 = int(round(block_s * MODEL_SR))
    overlap22 = int(round(overlap_s * MODEL_SR))
    search22 = int(round(search_s * MODEL_SR))
    right22 = int(round(EXTRA_CONTEXT_RIGHT_S * MODEL_SR))
    out22 = block22 + overlap22 + search22
    block16 = int(round(block_s * INPUT_SR))
    ring16 = (
        int(round(left_context_s * INPUT_SR))
        + int(round(overlap_s * INPUT_SR))
        + int(round(search_s * INPUT_SR))
        + block16
        + int(round(EXTRA_CONTEXT_RIGHT_S * INPUT_SR))
    )
    return BlockPlan(
        block16=block16,
        ring16=ring16,
        ce_cut_frames=cut,
        block22=block22,
        overlap22=overlap22,
        search22=search22,
        right22=right22,
        out22=out22,
        prosody_ctx22=int(round(PROSODY_CONTEXT_S * MODEL_SR)),
        mel_frames=(int(round(dit_context_s * MODEL_SR)) + out22 + right22) // hop,
        dit_context_s=dit_context_s,
    )


@dataclass
class Components:
    """The Seed-VC modules. LoadComponents builds the real ones; tests inject small stand-ins."""

    model: Any  # Munch(cfm, length_regulator)
    semantic_fn: Callable  # float32 numpy 16 kHz mono -> (1, frames, 1024) float32 tensor on device
    vocoder: Any  # HiFTGenerator
    campplus: Any
    to_mel: Callable  # (1, n) tensor at 22050 Hz -> (1, 80, frames)
    device: torch.device
    hop: int
    estimator_graph: GraphedDiT | None = None  # the CUDA-graph wrapper around the DiT when LoadComponents installed one


def LoadComponents(assets_dir, device: torch.device, cuda_graph: bool = True) -> Components:
    """Seed-VC load_models with local files only: DiT xlsr-tiny, CAMPPlus, HiFT, XLSR content encoder, mel function."""
    assets = Path(assets_dir)
    compat.PrepareSeedVc(SEED_VC_DIR)
    from modules.audio import mel_spectrogram
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor
    from modules.hifigan.generator import HiFTGenerator

    with open(assets / "config_dit_mel_seed_uvit_xlsr_tiny.yml", "r") as handle:
        config = yaml.safe_load(handle)
    model_params = recursive_munch(config["model_params"])
    if model_params.vocoder.type != "hifigan" or model_params.speech_tokenizer.type != "xlsr":
        raise ValueError("this engine loads the xlsr-tiny model: expected a hifigan vocoder and an xlsr speech tokenizer")
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(
        model, None, str(assets / "DiT_uvit_tat_xlsr_ema.pth"), load_only_params=True, ignore_modules=[], is_distributed=False
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    estimator_graph = None
    if cuda_graph and device.type == "cuda":
        estimator_graph = model.cfm.estimator = GraphedDiT(model.cfm.estimator, compat.HostKnownSequenceMask)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(assets / "campplus_cn_common.bin", map_location="cpu"))
    campplus.eval()
    campplus.to(device)

    with open(SEED_VC_DIR / "configs" / "hifigan.yml", "r") as handle:
        hift_config = yaml.safe_load(handle)
    hift = HiFTGenerator(**hift_config["hift"], f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]))
    hift.load_state_dict(torch.load(assets / "hift.pt", map_location="cpu"))
    hift.eval()
    hift.to(device)

    spect = config["preprocess_params"]["spect_params"]
    mel_args = {
        "n_fft": spect["n_fft"],
        "win_size": spect["win_length"],
        "hop_size": spect["hop_length"],
        "num_mels": spect["n_mels"],
        "sampling_rate": config["preprocess_params"]["sr"],
        "fmin": spect.get("fmin", 0),
        "fmax": None if spect.get("fmax", "None") == "None" else 8000,
        "center": False,
    }

    def ToMel(wave):
        return mel_spectrogram(wave, **mel_args)

    semantic = _MakeSemantic(assets / "wav2vec2-xls-r-300m", model_params.speech_tokenizer.output_layer, device)
    return Components(model, semantic, hift, campplus, ToMel, device, int(spect["hop_length"]), estimator_graph)


class SeedVCExpressive(VoiceEngine):
    kind = KIND_CLIP
    vram_estimate_mb = 1500.0  # the Seed-VC resident set; the EngineHost replaces it with the measured footprint after the first load

    def __init__(self, params: ParamStore, janitor: Janitor, assets_dir, device: str = "cuda", fp16: bool = True, cuda_graph: bool = True):
        self._params = params
        self._cuda_graph = cuda_graph
        self._assets = Path(assets_dir)
        self._device = torch.device(device)
        self._fp16 = bool(fp16) and self._device.type == "cuda"
        self._c: Components | None = None
        self._fcpe: FcpeF0 | None = None
        self._fcpe_source: FcpeF0 | None = None
        self._prosody: ProsodyStage | None = None
        self._bank = ReferenceBank(self._EncodePrompt)
        self._emotion = EmotionSelector()
        self._pitch = PitchSpreadTracker()
        self._stream_lock = threading.RLock()
        self._ring: np.ndarray | None = None
        self._live_style = None
        self._live_age = 0
        self._telemetry: dict = {}
        janitor.Add(self._Cleanup)

    def Load(self) -> dict:
        started = time.perf_counter()
        before = AllocatedMb(self._device)
        self.InstallComponents(LoadComponents(self._assets, self._device, self._cuda_graph))
        return {
            "models": ["DiT", "length_regulator", "CAMPPlus", "HiFT", "XLSR", "FCPE"],
            "vram_mb": AllocatedMb(self._device) - before,
            "load_seconds": time.perf_counter() - started,
            "backend": self._device.type,
            "fp16": self._fp16,
            "fcpe_device": self._fcpe.device,
        }

    def InstallComponents(self, components: Components, fcpe: FcpeF0 | None = None) -> None:
        with self._stream_lock:
            self._c = components
            self._fcpe = fcpe if fcpe is not None else FcpeF0()
            self._fcpe_source = self._fcpe if fcpe is not None else self._fcpe.WithThreshold(FcpeF0.SOURCE_THRESHOLD)
            self._prosody = ProsodyStage(self._fcpe, MODEL_SR)
            self.ResetStream()

    def SetSlot(self, idx: int, wav: np.ndarray, sr: int) -> dict:
        """Precompute and cache the CAMPPlus embedding, prompt condition and prompt mel of at most PROMPT_MAX_S seconds."""
        return self._bank.SetSlot(idx, wav, sr)

    def ClearSlot(self, idx: int) -> None:
        self._bank.ClearSlot(idx)

    def SlotInfo(self) -> list:
        return self._bank.SlotInfo()

    def SetManualSlot(self, idx: int) -> None:
        """Hotkey path: select slot idx and switch emotion_mode to manual so the choice takes effect."""
        self._params.Set("manual_slot", Validate("manual_slot", idx))
        self._params.Set("emotion_mode", "manual")

    def ResetStream(self) -> None:
        with self._stream_lock:
            self._ring = None
            self._live_style, self._live_age = None, 0
            self._emotion.Reset()
            self._pitch.Reset()
            if self._prosody is not None:
                self._prosody.Reset()

    def Loaded(self) -> bool:
        return self._c is not None

    def Release(self) -> None:
        self._Cleanup()

    def Telemetry(self) -> dict:
        graph = self._c.estimator_graph.Stats() if self._c is not None and self._c.estimator_graph is not None else None
        return {**self._telemetry, "fp16": self._fp16, "backend": self._device.type, "estimator_graph": graph}

    def ConvertBlock(self, block16k: np.ndarray, active: bool, steps: int | None = None) -> ConvertResult:
        """Convert one block_ms block of 16 kHz audio. An inactive block only advances the left-context ring (silence in,
        exact zeros out), so the ring stays aligned with real time."""
        c = self._Components()
        with self._stream_lock:
            clock = _StageClock(self._device.type == "cuda")
            p = self._ReadParams()
            plan = PlanBlock(p["block_ms"], p["overlap_ms"], p["left_context_s"], c.hop)
            block = np.ascontiguousarray(block16k, dtype=np.float32)
            if block.shape != (plan.block16,):
                raise ValueError(f"block16k must hold {plan.block16} samples ({p['block_ms']} ms at {INPUT_SR} Hz), got {block.shape}")
            self._PushRing(block, plan)
            if not active:
                return ConvertResult(np.zeros(plan.out22, np.float32), 0.0, 0, self._emotion.Weights(), self._prosody.MedianF0Hz(), 0.0, -120.0, False)
            slots = self._bank.Snapshot()
            loaded = [s is not None for s in slots]
            if not any(loaded):
                raise RuntimeError("no reference slot is loaded: call SetSlot first")
            rms_db, has_speech = self._BlockLevel(block)
            clock.Mark("other")
            f0_frames, f0_origin = self._InputF0(plan, p["prosody_follow"] > 0.0)
            self._pitch.Add(f0_frames[-(plan.block16 // FCPE_HOP) :])
            clock.Mark("fcpe")
            spread = self._pitch.StdSemitones()
            weights = self._emotion.Update(rms_db, spread, loaded, p["emotion_mode"], p["manual_slot"], active=has_speech)
            prompt = slots[int(np.argmax(weights))]
            style = None
            for slot, weight in zip(slots, weights):
                if slot is not None and weight > 0.0:
                    style = slot.style * weight if style is None else style + slot.style * weight
            if p["timbre_mix"] < 1.0:
                live = self._LiveStyle(c)
                if live is not None:
                    style = p["timbre_mix"] * style + (1.0 - p["timbre_mix"]) * live
            clock.Mark("other")
            n_steps = int(steps) if steps is not None else int(p["diffusion_steps"])
            with torch.no_grad():
                s_alt = c.semantic_fn(self._ring)[:, plan.ce_cut_frames :]
                clock.Mark("xlsr")
                cond = c.model.length_regulator(
                    s_alt, ylens=torch.LongTensor([plan.mel_frames]).to(c.device), n_quantizers=3, f0=None
                )[0]
                cat_condition = torch.cat([prompt.prompt_condition, cond], dim=1)
                clock.Mark("length_regulator")
                with torch.autocast(device_type="cuda", dtype=torch.float16) if self._fp16 else contextlib.nullcontext():
                    target = c.model.cfm.inference(
                        cat_condition,
                        # on the CPU on purpose: the DiT builds its attention mask with sequence_mask(x_lens), whose length.max() and
                        # arange(max_length) force a GPU to CPU sync on every estimator call when x_lens lives on the GPU
                        torch.LongTensor([cat_condition.size(1)]),
                        prompt.mel,
                        style,
                        None,
                        n_timesteps=n_steps,
                        inference_cfg_rate=p["cfg_rate"],
                    )
                    target = target[:, :, prompt.mel.size(-1) :]
                    clock.Mark("cfm")
                    wave = c.vocoder(target).squeeze()
                clock.Mark("vocoder")
            end = wave.numel() - plan.right22
            start = end - plan.out22
            if start < 0:
                raise RuntimeError(f"vocoder returned {wave.numel()} samples, need {plan.out22 + plan.right22}")
            context = min(plan.prosody_ctx22, start)
            segment = wave[start - context : end].float().cpu().numpy()
            clock.Mark("other")
            source = None
            if p["prosody_follow"] > 0.0 or p["energy_follow"] > 0.0:
                # The vocoder output ends where the ring ends (the DiT is conditioned on the ring's last mel_frames of content), so
                # the segment starts that many source samples before the ring end. Measured on the real models, the converted
                # loudness envelope sits 6 ms later than this (both test sources); that is under one F0 frame and is not corrected.
                first = len(self._ring) - (wave.numel() - (start - context)) * (INPUT_SR / MODEL_SR)
                source = SourceAudio(self._ring, first, f0_frames if p["prosody_follow"] > 0.0 else None, f0_origin)
            out, applied = self._prosody.Process(
                segment, plan.out22, plan.block22, p["pitch_shift_st"], p["formant_shift"], p["f0_expander"],
                p["prosody_follow"], p["energy_follow"], source,
            )
            spent = clock.Mark("other")
            if applied:
                stage = self._prosody.Telemetry()
                clock.ms["fcpe"] += stage["fcpe_ms"]
                clock.ms["world"] = stage["world_ms"]
                clock.ms["other"] -= min(spent, stage["fcpe_ms"] + stage["world_ms"])
                np.clip(out, -1.0, 1.0, out=out)
            inference_s = clock.TotalS()
            self._telemetry = {
                **{f"{name}_ms": clock.ms.get(name, 0.0) for name in ("xlsr", "length_regulator", "cfm", "vocoder", "fcpe", "world", "other")},
                **{f"{name}_cpu_ms": clock.cpu_ms.get(name, 0.0) for name in ("xlsr", "length_regulator", "cfm", "vocoder", "other")},
                "inference_s_last": inference_s,
                "steps_used": n_steps,
                "slot_weights": list(weights),
                "selected_slot": int(np.argmax(weights)),
                "fcpe_device": self._fcpe.device,
            }
            return ConvertResult(
                np.ascontiguousarray(out, dtype=np.float32),
                inference_s,
                n_steps,
                list(weights),
                self._prosody.MedianF0Hz(),
                0.0 if spread is None else float(spread),
                rms_db,
                bool(applied),
            )

    # ---------------------------------------------------------------- PRIVATE
    def _Components(self) -> Components:
        c = self._c
        if c is None:
            raise RuntimeError("models are not loaded: call Load() first")
        return c

    def _ReadParams(self) -> dict:
        snapshot = self._params.Snapshot()
        return {
            name: snapshot[name]
            for name in (
                "block_ms",
                "overlap_ms",
                "diffusion_steps",
                "cfg_rate",
                "left_context_s",
                "pitch_shift_st",
                "formant_shift",
                "f0_expander",
                "prosody_follow",
                "energy_follow",
                "timbre_mix",
                "emotion_mode",
                "manual_slot",
            )
        }

    def _PushRing(self, block: np.ndarray, plan: BlockPlan) -> None:
        old = self._ring
        if old is None or len(old) != plan.ring16:
            ring = np.zeros(plan.ring16, np.float32)
            if old is not None:
                keep = min(len(old), len(ring))
                ring[-keep:] = old[-keep:]
            self._ring = old = ring
        old[: -plan.block16] = old[plan.block16 :]
        old[-plan.block16 :] = block

    @staticmethod
    def _BlockLevel(block: np.ndarray) -> tuple[float, bool]:
        """RMS of the active (non-zero) samples in dBFS; exact zeros are gated silence. False when under 10 ms of speech."""
        speech = block[block != 0.0]
        if speech.size < FCPE_HOP:
            return -120.0, False
        return 20.0 * math.log10(float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))) + 1e-9), True

    def _InputF0(self, plan: BlockPlan, transfer: bool) -> tuple[np.ndarray, int]:
        """FCPE frames (10 ms) of the ring's tail and the ring index of the first one. The pitch tracker uses the last block's
        frames. The prosody transfer also needs the source F0 under the returned region, so it widens the window and reads the source
        with the stricter SOURCE_THRESHOLD tracker (the pitch tracker then sees those frames too: one FCPE call, not two)."""
        size = self._TransferWindow(plan) if transfer else plan.block16 + int(round(FCPE_CONTEXT_S * INPUT_SR))
        origin = max(len(self._ring) - size, 0)
        tracker = self._fcpe_source if transfer else self._fcpe
        return tracker.Infer(self._ring[origin:], INPUT_SR), origin

    @staticmethod
    def _TransferWindow(plan: BlockPlan) -> int:
        """Ring samples FCPE must see for the transfer, a whole number of FCPE hops: the returned region and the lookahead after it,
        the WORLD stage's lead before it, and FCPE's own left context."""
        span = (plan.out22 + plan.right22) * (INPUT_SR / MODEL_SR) + (WORLD_LEAD_S + FCPE_CONTEXT_S) * INPUT_SR
        return int(math.ceil(span / FCPE_HOP)) * FCPE_HOP

    def _LiveStyle(self, c: Components):
        """R16: CAMPPlus embedding of the speaker's own active speech in the left-context window, refreshed every few blocks."""
        self._live_age += 1
        if self._live_style is not None and self._live_age < LIVE_STYLE_REFRESH_BLOCKS:
            return self._live_style
        frames = self._ring[: len(self._ring) // XLSR_FRAME * XLSR_FRAME].reshape(-1, XLSR_FRAME)
        speech = np.ascontiguousarray(frames[np.any(frames != 0.0, axis=1)].reshape(-1))
        if speech.size < LIVE_STYLE_MIN_S * INPUT_SR:
            return self._live_style
        with torch.no_grad():
            wave = torch.from_numpy(speech).to(c.device)
            feat = kaldi.fbank(wave.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=INPUT_SR)
            feat = feat - feat.mean(dim=0, keepdim=True)
            self._live_style = c.campplus(feat.unsqueeze(0))
        self._live_age = 0
        return self._live_style

    def _EncodePrompt(self, wav22k: np.ndarray) -> tuple:
        """Prompt features exactly as real-time-gui.py custom_infer builds them: style, prompt condition, prompt mel."""
        c = self._Components()
        with torch.no_grad():
            wave = torch.from_numpy(wav22k).to(c.device)
            wave16 = torchaudio.functional.resample(wave, MODEL_SR, INPUT_SR)
            s_ori = c.semantic_fn(wave16.cpu().numpy())
            feat = kaldi.fbank(wave16.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=INPUT_SR)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = c.campplus(feat.unsqueeze(0))
            mel = c.to_mel(wave.unsqueeze(0))
            lengths = torch.LongTensor([mel.size(2)]).to(c.device)
            prompt_condition = c.model.length_regulator(s_ori, ylens=lengths, n_quantizers=3, f0=None)[0]
        return style, prompt_condition, mel

    def _Cleanup(self) -> None:
        with self._stream_lock:
            self._c, self._prosody, self._fcpe, self._fcpe_source, self._ring, self._live_style = None, None, None, None, None, None
            self._bank.Clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _MakeSemantic(xlsr_dir: Path, output_layer: int, device: torch.device) -> Callable:
    """XLSR content encoder as in load_models, from the local folder: layers cut to output_layer, fp16 on CUDA."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(xlsr_dir))
    xlsr = Wav2Vec2Model.from_pretrained(str(xlsr_dir))
    xlsr.encoder.layers = xlsr.encoder.layers[:output_layer]
    xlsr = xlsr.to(device).eval()
    half = device.type == "cuda"
    if half:
        xlsr = xlsr.half()

    def Semantic(wave16k: np.ndarray):
        inputs = extractor(
            [wave16k], return_tensors="pt", return_attention_mask=True, padding=True, sampling_rate=INPUT_SR
        ).to(device)
        with torch.no_grad():
            out = xlsr(inputs.input_values.half() if half else inputs.input_values)
        return out.last_hidden_state.float()

    return Semantic


class _StageClock:
    """Per-stage wall time and the calling thread's CPU time. On CUDA every mark synchronises, so asynchronous kernels are charged to
    the stage that launched them. A stage whose CPU time is far below its wall time was waiting (GPU, or the GIL held by another thread);
    one whose CPU time is close to its wall time was computing on a slow or busy core."""

    def __init__(self, sync: bool):
        self._sync = sync
        self.ms: dict[str, float] = {}
        self.cpu_ms: dict[str, float] = {}
        self._start = self._last = time.perf_counter()
        self._cpu_last = time.thread_time()

    def Mark(self, stage: str) -> float:
        if self._sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = (now - self._last) * 1000.0
        self.ms[stage] = self.ms.get(stage, 0.0) + elapsed
        self._last = now
        cpu = time.thread_time()
        self.cpu_ms[stage] = self.cpu_ms.get(stage, 0.0) + (cpu - self._cpu_last) * 1000.0
        self._cpu_last = cpu
        return elapsed

    def TotalS(self) -> float:
        return time.perf_counter() - self._start
