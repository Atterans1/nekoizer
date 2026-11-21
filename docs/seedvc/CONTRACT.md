# Interface contract (all agents build against this; change it only by messaging the owner of pipeline.py)

Python 3.12, venv C:/Users/jeffr/s2s-voicechanger/.venv-seedvc/Scripts/python.exe (hardlink overlay of the working venv; never modify C:/Users/jeffr/s2s-voicechanger/.venv). Project root R = C:/Users/jeffr/s2s-voicechanger.
New packages under R: audio/, engine/, ui/, tests_seedvc/. Do not edit legacy flat files (audio_engine.py, vc_inference.py, server.py, ...). Shared: seedvc_params.py (read it first; it holds every parameter name, range, default, constant, ParamStore, Janitor).

Code rules (owner's guidelines): PascalCase for classes, functions and methods (as in audio_engine.py); parameter and dict-key names are the snake_case names in seedvc_params.ParamSpec exactly; each module has "# ---- PUBLIC" and "# ---- PRIVATE" comment sections; anything that holds a stream, thread, process or file registers its cleanup on a Janitor; keep audio callbacks free of locks and allocation where practical; responsiveness first, no busy loops without sleep.
Tests: plain unittest under R/tests_seedvc/ (run with the venv python -m unittest), literal expected values, deterministic. Mocks are allowed only inside tests. No stubs, hardcoded outputs or renamed parameters in product code.
Long commands (>25 s) must run detached: write a .ps1, Start-Process powershell -WindowStyle Hidden, log file + exit-code marker file, poll the marker with short cells. Never print secrets. Do not grep recursively over R (the venv has 36k files): grep named files or subfolders only.

## Sample rates and data flow
mic (device rate, e.g. 48 kHz) -> InputChain: HPF 80 Hz -> compressor 2.5:1 -> Silero VAD (gates to exact zeros) -> resampled to 16 kHz float32 -> Pipeline worker -> SeedVCExpressive.ConvertBlock -> 22050 Hz wav -> OutputChain: Sola -> resample to device rate -> CABLE Input. Sidetone monitor: InputChain publishes the post-compressor, pre-VAD signal to a ring at device rate; OutputChain's monitor stream plays it at monitor_volume when monitor_enabled.

## audio/input_chain.py (owner: InputAgent; also audio/dsp.py, audio/vad.py, audio/recorder.py)
Pure classes (unit tested, state kept across blocks): HighPass(sr), SoftCompressor(sr) with automatic makeup (GetState() -> gain_reduction_db, makeup_db), SileroVad(onnx) wrapping silero_vad's ONNX model on 512-sample 16 kHz frames with threshold VAD_THRESHOLD and VAD_HANG_MS hang (output probability + active flag).
@dataclass InputBlock: samples16k (np.float32, exactly n samples, exact zeros when inactive), active (bool: any speech in the block or hang tail), t_capture (time.perf_counter() of the first sample of the block as captured, derived from the PortAudio ADC timestamp), rms_db (float, of the block before gating), peak (float, 0..1 before gating).
class InputChain(params: ParamStore, janitor: Janitor, device_name: str | None = None):
  ListDevices() -> list[dict(id, name, hostapi, default_samplerate, max_input_channels)]
  Start() / Stop()  (opens WASAPI Exclusive first, falls back to Low-Latency Shared; no other host API)
  ReadBlock(n_samples16k: int, timeout_s: float) -> InputBlock | None  (blocks until n post-chain 16 kHz samples are available)
  MonitorRing: object with Read(n) -> np.ndarray (float32 mono at device rate), used by OutputChain
  Telemetry() -> dict(mode 'exclusive'|'shared_lowlatency', device, samplerate, buffer_ms, vad_prob, vad_active, gain_reduction_db, makeup_db, overruns, enhancements_bypassed: bool|None)
  StartRecording(max_seconds=RECORDER_MAX_S) / StopRecording() -> np.ndarray float32 mono at 24000 Hz of the raw (HPF only, no compressor, no gate) mic; LevelSnapshot() -> dict(peak, clipping: bool, waveform: list[float] downsampled last ~0.5 s) for the UI's live meter. Clipping = any |sample| >= 0.999.

## engine/seedvc_expressive.py (owner: EngineAgent; also engine/emotion.py, engine/prosody_stage.py, engine/reference_bank.py)
Assets in R/assets/seedvc/ (see manifest.json): DiT_uvit_tat_xlsr_ema.pth, config_dit_mel_seed_uvit_xlsr_tiny.yml, campplus_cn_common.bin, hift.pt, wav2vec2-xls-r-300m/ (config.json, preprocessor_config.json, pytorch_model.bin), seed-vc-<sha>.zip (unzip to R/third_party/seed-vc/; import its modules via sys.path; load everything from these local paths, no network at runtime). Download state: assets/seedvc/dl_progress.txt and dl_done.txt (may still be running when you start; the DiT checkpoint and config are already there).
@dataclass ConvertResult: wav22k (np.float32, length = block_samples + overlap_samples + search_samples at 22050 Hz, as Seed-VC's real-time-gui.py produces before SOLA), inference_s, steps_used, slot_weights (list[float] of 4), f0_median_hz, pitch_std_st, rms_db, stage_applied (bool: WORLD stage ran).
class SeedVCExpressive(params: ParamStore, janitor: Janitor, assets_dir, device='cuda', fp16=True):
  Load() -> dict (models loaded, vram_mb, load_seconds)
  SetSlot(idx: 0..3, wav: np.ndarray, sr: int) -> dict(idx, name, seconds_used, embedding_dim, prompt_tokens, cached: bool) (precompute CAMPPlus embedding + acoustic latents/prompt condition once, cache by content hash; at most PROMPT_MAX_S used)
  ClearSlot(idx) ; SlotInfo() -> list[dict]
  ResetStream()  (clears the left-context ring buffer and emotion state)
  ConvertBlock(block16k: np.ndarray, active: bool, steps: int | None = None) -> ConvertResult  (block16k is exactly block_ms of 16 kHz audio; keeps a rolling left-context ring buffer of left_context_s seconds; uses cfg_rate and diffusion_steps (or steps when given), timbre_mix, pitch_shift_st, formant_shift, f0_expander, emotion_mode, manual_slot)
  SetManualSlot(idx) (hotkey path) ; Telemetry() -> dict
Prosody: FCPE (torchfcpe) F0 on every block; WORLD stage (pyworld) applies pitch_shift_st, formant_shift (spectral envelope axis scaled by 2^(x/12)) and f0_expander (scale log-F0 deviation from the running median over F0_MEDIAN_WINDOW_S) to the converted block with 0.5 s of context; the stage is skipped (bit-exact) when pitch 0, formant 0 and expander 1.0. See SPEC_RESOLUTIONS.md R1, R11, R12, R14-R16. Do not substitute Seed-VC, CAMPPlus, FCPE or the Silero VAD; if a component cannot be loaded or run, stop and report.

## audio/output_chain.py (owner: OutputAgent; also audio/sola.py, audio/balancer.py)
class Sola(sr: int, block_ms: float, overlap_ms: float, search_ms: float): Process(wav: np.ndarray) -> np.ndarray of exactly block_ms of samples; pure and unit tested (correlation search over search_ms, sin^2/cos^2 crossfade of overlap_ms, port the logic of Seed-VC real-time-gui.py sola_buffer / sola_search). Rebuilt when block_ms or overlap_ms changes. Input wav length = block + overlap + search samples.
class AutoBalancer(params): Update(inference_s, block_s, now=None); EffectiveSteps(requested: int, now=None) -> int (R17: EMA >= BALANCER_THRESHOLD -> requested - BALANCER_STEP_DROP for BALANCER_HOLD_S, never below ParamSpec min 6, then restore; honours the auto_balancer param); State() -> dict.
@dataclass BlockMeta: t_capture (perf_counter at capture of the block's first sample), block_ms, active, inference_s.
class OutputChain(params: ParamStore, janitor: Janitor, device_name=CABLE_OUTPUT_NAME, monitor_device_name=None, monitor_ring=None):
  Start() / Stop()  (WASAPI to 'CABLE Input (VB-Audio Virtual Cable)', not 'CABLE In 16ch')
  Submit(wav22k: np.ndarray, meta: BlockMeta) -> None   (Sola, resample to device rate, enqueue to the ring)
  SubmitSilence(meta: BlockMeta) -> None                  (exact zeros for block_ms; keeps timing and the SOLA state sane)
  Balancer: AutoBalancer ; EffectiveSteps(requested) -> int ; ReportInference(inference_s, block_s)
  Telemetry() -> dict(device, samplerate, underruns, buffer_ms, e2e_ms_last, e2e_ms_mean, e2e_ms_max, blocks_out, monitor: dict(enabled, device, buffer_ms), balancer: dict)
  End-to-end latency per block = playback start time of the block's first sample (PortAudio DAC time converted to perf_counter) minus BlockMeta.t_capture. Underrun = the output callback needed samples and the ring was empty (counted, filled with zeros).
  Sidetone monitor: its own lowest-latency stream; plays MonitorRing * monitor_volume only when monitor_enabled; reports its buffer size and measured delay honestly.

## ui/ (owner: UiAgent; also ui/hotkeys.py)
FastAPI server (ui/server.py) + static dark UI (ui/static/index.html, app.js, style.css) shown by pywebview, reading and writing a Pipeline (see below). REST: GET /api/state, POST /api/param {name,value}, POST /api/preset {name}, POST /api/slot/{i} (multipart wav), DELETE /api/slot/{i}, POST /api/slot/{i}/record/start|stop, POST /api/manual_slot {idx}, GET /api/devices; WebSocket /ws pushes telemetry + live recorder waveform + clipping meter at about 20 Hz. Controls: tone rack (pitch_shift_st, formant_shift, f0_expander, cfg_rate, diffusion_steps, left_context_s, block_ms; also overlap_ms, timbre_mix, monitor toggle + volume, emotion mode auto/manual), four drag-and-drop .wav slots named by SLOT_NAMES with a per-slot record button (10 s, live waveform, clipping meter) and a manual-select button, three working presets and a fourth disabled button labelled "Ultra-Low-... (name cut off in spec)" that applies nothing (R20). Global hotkeys Ctrl+Alt+1..4 via ctypes RegisterHotKey in ui/hotkeys.py (calls Pipeline.SetManualSlot). Slider ranges and defaults are read from seedvc_params.ParamSpec at runtime (never hardcoded in JS).

## pipeline.py + app_seedvc.py (owner: lead)
class Pipeline(assets_dir, params: ParamStore | None = None): Start() (engine.Load, input/output start, worker thread: block = input.ReadBlock(block_samples16k); if block.active -> steps = output.EffectiveSteps(diffusion_steps); res = engine.ConvertBlock(...); output.ReportInference(...); output.Submit(...) else output.SubmitSilence(...)), Stop(), SetParam/GetParams/ApplyPreset, SetSlotFromWav(idx, wav, sr), ClearSlot, SetManualSlot, StartRecording(idx)/StopRecording(idx) (stores into the slot), Telemetry() (all modules merged), ProcessOffline(audio16k) -> (wav22k, stats) running the same engine + SOLA without devices. app_seedvc.py: python app_seedvc.py [--headless-test]; opens the UI window.
