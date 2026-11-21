"""Chatterbox S3Gen voice-conversion engine.

Path: 16 kHz audio -> S3Tokenizer (25 Hz tokens) -> CFM decoder (target CAMPPlus embedding + cached
reference prompt) -> HiFT vocoder. The TTS generate() path is never used and the reference is encoded
exactly once, in LoadReference().
"""
from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

import pitch

LOG = logging.getLogger("vc")

INPUT_SR = 16000
OUTPUT_SR = 24000
TOKEN_HZ = 25
MEL_PER_TOKEN = 2
HOP_24K = 480
MIN_REF_S = 6.0
MAX_REF_S = 15.0
EMBED_REF_S = 10.0
MIN_REF_SR = 8000
F0_MIN_HZ = 60.0
F0_MAX_HZ = 500.0
F0_FRAME_16K = 640
F0_HOP_16K = 320
F0_HOP_S = F0_HOP_16K / 16000
PITCH_CONTEXT_S = 0.1
VOICED_THRESHOLD = 0.35
REPO_ID = "ResembleAI/chatterbox"
WEIGHT_FILES = ("s3gen.safetensors",)

InputSampleRate = INPUT_SR
OutputSampleRate = OUTPUT_SR


# ---------------------------------------------------------------- PUBLIC
def SyncFreeChunkMask(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                      static_chunk_size, num_decoding_left_chunks, enable_full_context=True):
    """add_optional_chunk_mask without the host sync: rows that are all False are set True, as upstream does,
    but via an elementwise OR instead of a .item() check, so the decoder can be captured in a CUDA graph."""
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask

    if use_dynamic_chunk:
        raise NotImplementedError("dynamic chunk masks are training-only")
    chunk_masks = masks
    if static_chunk_size > 0:
        chunk_masks = masks & subsequent_chunk_mask(xs.size(1), static_chunk_size, num_decoding_left_chunks, xs.device).unsqueeze(0)
    return chunk_masks | (chunk_masks.sum(dim=-1, keepdim=True) == 0)


def DownloadWeights() -> Path:
    """Fetch s3gen weights from Hugging Face. Raises on any failure; never substitutes a model."""
    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(repo_id=REPO_ID, filename=name) for name in WEIGHT_FILES]
    return Path(paths[0]).parent


def ReadReferenceWav(source) -> tuple[np.ndarray, int]:
    """Validate and decode a reference clip. Accepts a path or WAV bytes. Returns (float32 mono, sr)."""
    if isinstance(source, (str, Path)):
        if Path(source).suffix.lower() != ".wav":
            raise ValueError("Reference must be a .wav file")
        handle = str(source)
    elif isinstance(source, (bytes, bytearray)):
        handle = io.BytesIO(source)
    else:
        raise ValueError("Reference must be a path or WAV bytes")
    try:
        data, sr = sf.read(handle, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Reference is not a readable WAV file: {exc}") from exc
    if data.shape[1] > 2:
        raise ValueError(f"Reference has {data.shape[1]} channels; use mono or stereo")
    if sr < MIN_REF_SR:
        raise ValueError(f"Reference sample rate {sr} Hz is below the {MIN_REF_SR} Hz minimum")
    wav = data.mean(axis=1)
    duration = len(wav) / sr
    if not MIN_REF_S <= duration <= MAX_REF_S:
        raise ValueError(
            f"Reference is {duration:.1f} s long; it must be between {MIN_REF_S:.0f} and {MAX_REF_S:.0f} s"
        )
    if float(np.sqrt(np.mean(wav**2))) < 1e-3:
        raise ValueError("Reference is silent")
    return wav, sr


def Resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)
    g = np.gcd(src_sr, dst_sr)
    return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)


class VCEngine:
    """Thread-safe VC engine. One Convert() at a time (serialised by an internal lock)."""

    def __init__(self, device: str = "cuda", precision: str = "auto", prompt_s: float = 3.0):
        self._device = torch.device(device)
        self._requested_precision = precision
        self._prompt_s = prompt_s
        self._lock = threading.Lock()
        self._s3gen = None
        self._ref = None
        self._target_embedding = None
        self._target_f0_median = None
        self._source_embedding = None
        self._source_embedding_age = 0
        self._source_f0_hist: list[float] = []
        self._estimator_calls = 0
        self._telemetry = {
            "backend": "uninitialised",
            "precision": "n/a",
            "device": str(device),
            "compiled": False,
            "fallback_reason": None,
            "last_inference_ms": 0.0,
        }
        self._nvml = None
        self._vocoder = None

    # ---- lifecycle
    def Load(self) -> None:
        from chatterbox.models.s3gen import S3Gen
        from safetensors.torch import load_file

        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False; refusing to fall back to CPU")
        ckpt = DownloadWeights() / "s3gen.safetensors"
        model = S3Gen()
        missing = model.load_state_dict(load_file(ckpt), strict=False)
        LOG.info("s3gen loaded; missing=%d unexpected=%d", len(missing.missing_keys), len(missing.unexpected_keys))
        model.to(self._device).eval()
        self._s3gen = model
        self._SelectPrecision()
        self._GraphEstimator()
        self._BuildVocoder()
        self._HookEstimator()
        self._TryCompile()
        self._InitNvml()

    def LoadReference(self, source) -> dict:
        self._RequireLoaded()
        wav, sr = ReadReferenceWav(source)
        wav16 = Resample(wav, sr, INPUT_SR)
        wav24 = Resample(wav, sr, OUTPUT_SR)
        with self._lock, torch.inference_mode():
            ref = self._s3gen.embed_ref(torch.from_numpy(wav24[: int(EMBED_REF_S * OUTPUT_SR)]), OUTPUT_SR, device=self._device)
            n_prompt = int(self._prompt_s * TOKEN_HZ)
            n_prompt = min(n_prompt, ref["prompt_token"].shape[1])
            ref["prompt_token"] = ref["prompt_token"][:, :n_prompt]
            ref["prompt_token_len"] = torch.tensor([n_prompt], device=self._device)
            ref["prompt_feat"] = ref["prompt_feat"][:, : n_prompt * MEL_PER_TOKEN]
            self._ref = ref
            self._target_embedding = ref["embedding"].detach().clone()
            self._target_f0_median = self._MedianF0(torch.from_numpy(wav16).to(self._device))
            self._source_embedding = None
            self._source_f0_hist.clear()
        return {
            "duration_s": round(len(wav) / sr, 2),
            "sample_rate": sr,
            "embedding_dim": int(self._target_embedding.shape[-1]),
            "target_f0_median_hz": round(self._target_f0_median, 1),
            "prompt_tokens": n_prompt,
        }

    def ResetStream(self) -> None:
        with self._lock:
            self._source_f0_hist.clear()
            self._source_embedding = None

    def ConvertWindow(self, audio16k: np.ndarray, n_out: int, params: dict) -> np.ndarray:
        """Convert a window (left context + new audio + lookahead) and return its last n_out 24 kHz samples."""
        self._RequireLoaded()
        if self._ref is None:
            raise RuntimeError("No target voice loaded; call LoadReference() first")
        steps = int(params["cfm_steps"])
        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            wav16 = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32)).to(self._device)[None]
            tokens, _ = self._s3gen.tokenizer(wav16)
            embedding = self._BlendedEmbedding(wav16, float(params["timbre_blend"]))
            mels = self._Flow(tokens, embedding, steps)
            wav = self._Vocode(mels.float())
            keep = n_out + int(PITCH_CONTEXT_S * OUTPUT_SR)
            out = wav[0, -keep:] if wav.shape[1] >= keep else F.pad(wav[0], (keep - wav.shape[1], 0))
            out = out.detach().float().cpu().numpy()
        out = self._ApplyPitch(out, wav16, int(params["pitch_semitones"]), bool(params["f0_preserve"]))[-n_out:]
        self._telemetry["last_inference_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    def Telemetry(self) -> dict:
        t = dict(self._telemetry)
        t["estimator_calls_total"] = self._estimator_calls
        if t.get("cuda_graph") and t.get("vocoder_graph"):
            t["backend"] = "pytorch-cuda+cudagraph"
        t.update(self._GpuStats())
        return t

    @property
    def EstimatorCalls(self) -> int:
        return self._estimator_calls

    # ---------------------------------------------------------------- PRIVATE
    def _RequireLoaded(self) -> None:
        if self._s3gen is None:
            raise RuntimeError("VCEngine.Load() has not been called")

    def _SelectPrecision(self) -> None:
        want = self._requested_precision
        if self._device.type != "cuda":
            self._telemetry.update(backend="pytorch-cpu", precision="fp32", fallback_reason="device is not CUDA")
            return
        order = ["fp16", "bf16"] if want == "auto" else [want]
        for name in order:
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
            try:
                self._s3gen.flow.to(dtype)
                self._telemetry.update(backend="pytorch-cuda", precision=name, fallback_reason=None)
                return
            except Exception as exc:
                LOG.warning("precision %s failed: %s", name, exc)
                self._s3gen.flow.float()
                self._telemetry["fallback_reason"] = f"{name} failed: {exc}"
        self._telemetry.update(backend="pytorch-cuda", precision="fp32")

    def _GraphEstimator(self) -> None:
        """Replay the CFM estimator as a CUDA graph per static input shape (removes per-kernel launch overhead).

        Eager fallback (reported in telemetry) when the device is not CUDA or capture fails for a shape.
        """
        if self._device.type != "cuda":
            return
        import chatterbox.models.s3gen.decoder as decoder_module

        decoder_module.add_optional_chunk_mask = SyncFreeChunkMask
        estimator = self._s3gen.flow.decoder.estimator
        eager = estimator.forward
        graphs: dict = {}
        telemetry = self._telemetry

        def graphed(x, mask, mu, t, spks, cond, r=None):
            key = (tuple(x.shape), x.dtype)
            entry = graphs.get(key)
            if entry is None:
                try:
                    static = {k: v.clone() for k, v in (("x", x), ("mask", mask), ("mu", mu), ("t", t), ("spks", spks), ("cond", cond))}
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            eager(**static)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = eager(**static)
                    entry = (graph, static, out)
                    telemetry["cuda_graph"] = True
                except Exception as exc:
                    LOG.warning("cuda graph capture failed for %s: %s", key, exc)
                    telemetry["cuda_graph"] = False
                    telemetry["fallback_reason"] = f"cuda graph capture failed: {exc}"
                    entry = False
                graphs[key] = entry
            if entry is False or r is not None:
                return eager(x, mask, mu, t, spks, cond, r)
            graph, static, out = entry
            static["x"].copy_(x)
            static["mask"].copy_(mask)
            static["mu"].copy_(mu)
            static["t"].copy_(t)
            static["spks"].copy_(spks)
            static["cond"].copy_(cond)
            graph.replay()
            return out.clone()

        estimator.forward = graphed

    def _HookEstimator(self) -> None:
        estimator = self._s3gen.flow.decoder.estimator
        original = estimator.forward

        def counted(*args, **kwargs):
            self._estimator_calls += 1
            return original(*args, **kwargs)

        estimator.forward = counted
        self._estimator_original = original

    def _TryCompile(self) -> None:
        if self._device.type != "cuda":
            return
        try:
            import triton  # noqa: F401
        except Exception as exc:
            self._telemetry["fallback_reason"] = ((self._telemetry["fallback_reason"] or "") + f" torch.compile unavailable: {exc}").strip()
            return
        try:
            est = self._s3gen.flow.decoder.estimator
            est.forward = torch.compile(est.forward, dynamic=True)
            self._telemetry["compiled"] = "pending-first-call"
        except Exception as exc:
            self._telemetry["fallback_reason"] = ((self._telemetry["fallback_reason"] or "") + f" torch.compile failed: {exc}").strip()

    def _InitNvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            self._nvml = None

    def _GpuStats(self) -> dict:
        stats = {"gpu_util_pct": None, "gpu_mem_mb": None}
        if self._device.type == "cuda":
            stats["gpu_mem_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
        if self._nvml:
            nv, handle = self._nvml
            stats["gpu_util_pct"] = nv.nvmlDeviceGetUtilizationRates(handle).gpu
        return stats

    def _Flow(self, tokens: torch.Tensor, embedding: torch.Tensor, steps: int) -> torch.Tensor:
        """S3 tokens -> mel through the CFM decoder.

        Calls flow.inference directly: S3Token2Mel.forward casts every ref tensor (token ids included) to the
        model dtype, which corrupts integer token ids above 2048 in fp16.
        """
        ref = self._ref
        dtype = self._s3gen.dtype
        mels, _ = self._s3gen.flow.inference(
            token=tokens,
            token_len=torch.tensor([tokens.shape[1]], device=self._device),
            prompt_token=ref["prompt_token"],
            prompt_token_len=torch.tensor([ref["prompt_token"].shape[1]], device=self._device),
            prompt_feat=ref["prompt_feat"].to(dtype),
            prompt_feat_len=None,
            embedding=embedding.to(dtype),
            finalize=True,
            n_timesteps=steps,
            noised_mels=None,
            meanflow=False,
        )
        return mels

    def _BlendedEmbedding(self, wav16: torch.Tensor, blend: float) -> torch.Tensor:
        """blend=1.0: pure target speaker; blend=0.0: the live source speaker's own CAMPPlus embedding."""
        if blend >= 0.999:
            return self._target_embedding
        self._source_embedding_age += 1
        if self._source_embedding is None or self._source_embedding_age >= 8:
            emb = self._s3gen.speaker_encoder.inference(wav16.to(self._s3gen.dtype))
            self._source_embedding = emb.detach()
            self._source_embedding_age = 0
        tgt = F.normalize(self._target_embedding.float(), dim=-1)
        src = F.normalize(self._source_embedding.float(), dim=-1)
        mixed = F.normalize(blend * tgt + (1.0 - blend) * src, dim=-1)
        return mixed.to(self._target_embedding.dtype)

    def _TrackF0(self, wav16: torch.Tensor) -> torch.Tensor:
        """FFT autocorrelation F0 tracker. Returns Hz per 20 ms frame, 0 where unvoiced."""
        x = wav16.float()
        pad = F0_FRAME_16K // 2
        x = F.pad(x[None, None], (pad, pad), mode="reflect")[0, 0]
        frames = x.unfold(0, F0_FRAME_16K, F0_HOP_16K)
        frames = frames - frames.mean(dim=1, keepdim=True)
        frames = frames * torch.hann_window(F0_FRAME_16K, device=x.device)
        n = 2 * F0_FRAME_16K
        spec = torch.fft.rfft(frames, n=n)
        ac = torch.fft.irfft(spec.abs() ** 2, n=n)[:, :F0_FRAME_16K]
        ac = ac / ac[:, :1].clamp_min(1e-9)
        lo, hi = int(INPUT_SR / F0_MAX_HZ), int(INPUT_SR / F0_MIN_HZ)
        seg = ac[:, lo:hi]
        peak, idx = seg.max(dim=1)
        f0 = INPUT_SR / (idx + lo).float()
        return torch.where(peak > VOICED_THRESHOLD, f0, torch.zeros_like(f0))

    def _MedianF0(self, wav16: torch.Tensor) -> float:
        f0 = self._TrackF0(wav16)
        voiced = f0[f0 > 0]
        if voiced.numel() < 10:
            raise ValueError("Reference has too little voiced speech to estimate pitch")
        return float(voiced.median())

    def _Vocode(self, mels: torch.Tensor) -> torch.Tensor:
        return self._vocoder(mels)

    def _VocoderCore(self, mels):
        hift = self._s3gen.mel2wav
        f0 = hift.f0_predictor(mels)
        s = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = hift.m_source(s)
        return hift.decode(x=mels, s=s.transpose(1, 2))

    def _ApplyPitch(self, wav24: np.ndarray, wav16: torch.Tensor, semitones: int, f0_preserve: bool) -> np.ndarray:
        """Pitch shift / F0 transplant on the converted waveform (see pitch.py). No-op when both are off."""
        if semitones == 0 and not f0_preserve:
            return wav24
        tail = wav24
        source_f0 = None
        kwargs = {}
        if f0_preserve:
            track = self._TrackF0(wav16[0] if wav16.dim() > 1 else wav16).cpu().numpy()
            voiced = track[track > 0]
            if voiced.size:
                self._source_f0_hist.extend(voiced.tolist())
                del self._source_f0_hist[:-400]
            if self._source_f0_hist:
                span = len(tail) / OUTPUT_SR
                total = len(wav16.reshape(-1)) / INPUT_SR
                first = max(int((total - span) / F0_HOP_S) - 1, 0)
                source_f0 = track[first:]
                kwargs = {
                    "source_start_s": first * F0_HOP_S - (total - span),
                    "register_ratio": self._target_f0_median / float(np.median(self._source_f0_hist)),
                }
        t0 = time.perf_counter()
        shifted = pitch.Process(tail, OUTPUT_SR, semitones, source_f0=source_f0, source_hop_s=F0_HOP_S, **kwargs)
        self._telemetry["pitch_ms"] = (time.perf_counter() - t0) * 1000.0
        return shifted

    def _MakeVocoderDeviceNative(self) -> None:
        """Remove the host round trips inside HiFT: the CPU-resident STFT window is moved to the device once, and
        SineGen's CPU-side Uniform phase sampling becomes a device-side uniform draw (same distribution)."""
        hift = self._s3gen.mel2wav
        gen = hift.m_source.l_sin_gen
        hift.stft_window = hift.stft_window.to(self._device)

        @torch.no_grad()
        def sine_forward(f0):
            harmonics = torch.arange(1, gen.harmonic_num + 2, device=f0.device, dtype=f0.dtype).view(1, -1, 1)
            theta = 2 * np.pi * (torch.cumsum(f0 * harmonics / gen.sampling_rate, dim=-1) % 1)
            phase = (torch.rand(f0.size(0), gen.harmonic_num + 1, 1, device=f0.device) * 2 - 1) * np.pi
            phase[:, 0, :] = 0
            sine = gen.sine_amp * torch.sin(theta + phase)
            uv = (f0 > gen.voiced_threshold).type(torch.float32)
            noise = (uv * gen.noise_std + (1 - uv) * gen.sine_amp / 3) * torch.randn_like(sine)
            return sine * uv + noise, uv, noise

        gen.forward = sine_forward

        n_fft, hop = hift.istft_params["n_fft"], hift.istft_params["hop_len"]
        window = hift.stft_window

        def istft(magnitude, phase):
            """Overlap-add inverse STFT without torch.istft's host-side envelope check (identical maths, center=True)."""
            magnitude = torch.clip(magnitude, max=1e2)
            spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
            frames = torch.fft.irfft(spec, n=n_fft, dim=1) * window.view(1, -1, 1)
            length = (frames.shape[-1] - 1) * hop + n_fft
            fold = torch.nn.Fold(output_size=(1, length), kernel_size=(1, n_fft), stride=(1, hop))
            out = fold(frames)[:, 0, 0]
            env = fold((window ** 2).view(1, -1, 1).expand(1, n_fft, frames.shape[-1]).contiguous())[:, 0, 0]
            out = out / env.clamp_min(1e-11)
            return out[:, n_fft // 2: -(n_fft // 2)]

        hift._istft = istft
        self._istft_reference = None

    def _BuildVocoder(self) -> None:
        self._MakeVocoderDeviceNative()
        core = self._VocoderCore
        self._vocoder = core
        if self._device.type != "cuda":
            return
        graphs: dict = {}
        telemetry = self._telemetry

        def graphed(mels):
            key = tuple(mels.shape)
            entry = graphs.get(key)
            if entry is None:
                try:
                    static = [mels.clone()]
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(2):
                            core(*static)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = core(*static)
                    entry = (graph, static, out)
                    telemetry["vocoder_graph"] = True
                except Exception as exc:
                    LOG.warning("vocoder graph capture failed for %s: %s", key, exc)
                    telemetry["vocoder_graph"] = False
                    entry = False
                graphs[key] = entry
            if entry is False:
                return core(mels)
            graph, static, out = entry
            static[0].copy_(mels)
            graph.replay()
            return out.clone()

        self._vocoder = graphed
