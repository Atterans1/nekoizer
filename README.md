# S2S Voice Changer (Chatterbox S3Gen VC)

Local real-time speech-to-speech voice changer. Backbone: **Chatterbox S3Gen VC only** (S3Tokenizer 25 Hz tokens,
CAMPPlus 192-d speaker embedding, conditional flow matching decoder, HiFT vocoder). No ASR, LLM, TTS, Seed-VC, RVC or
HuBERT in the conversion path. The reference is encoded once in `VCEngine.LoadReference` and cached.

Project path: `C:\Users\jeffr\s2s-voicechanger`

## Executable
`dist\S2SVoiceChanger\S2SVoiceChanger.exe` (PyInstaller onedir, ~4.4 GB because it bundles torch + CUDA 12.6 runtime; not a single-file exe). Build: `.venv\Scripts\python.exe build_exe.py` (add `--console` for a debug build). Mutable data (presets, references, created voices, app.log) lives in `%LOCALAPPDATA%\S2SVoiceChanger` (override with `S2S_DATA_DIR`). Model weights are downloaded to the Hugging Face cache on first launch. Verified on this PC (RTX 3050, Windows 11 build 26300, 2026-11-16): the exe opened its native window, served the API, loaded the model, and converted real speech from the virtual cable (21 model calls, 47 silent chunks gated, backend pytorch-cuda+cudagraph fp16, no dropped chunks, e2e latency 565-722 ms). Not verified: running the exe on a machine without this Python environment, CUDA driver compatibility on other GPUs.

## Run
`run.bat` (one command): creates `.venv` (Python 3.12 via uv), installs `requirements.txt` (`uv pip install --no-deps`),
downloads weights from Hugging Face on first launch (`download_weights.py`, fails loudly, never substitutes a model),
then opens the desktop GUI (`desktop.py`, pywebview over the FastAPI/WebSocket server on 127.0.0.1:8765).
Server only: `.venv\Scripts\python.exe server.py`.

## Modules
| file | role |
|---|---|
| vc_inference.py | S3Gen VC: reference validation (.wav, 6-15 s, resampled), cached prompt/embedding, CFM steps -> real sampler schedule, fp16, CUDA-graph replay of the CFM estimator and vocoder, timbre blend, telemetry |
| pitch.py | pitch shift / F0 transplant on the converted waveform with WORLD (formant- and duration-preserving) |
| vad_gate.py | Silero VAD gate (silence never reaches the model) |
| audio_engine.py | WASAPI input/output streams, SPSC rings, SOLA at every chunk boundary, ADC->DAC latency measurement |
| server.py + frontend/ | FastAPI + WebSocket, W-Okada-style GUI, presets, voice creation |
| voice_creator.py, web_audio.py | offline voice creation (Qwen2.5-0.5B on CPU + public-domain web audio) |
| params.py | single source of truth for ranges/validation |

## Controls (all validated server-side; each reaches the engine)
- Pitch -12..+12 semitones: WORLD analysis/resynthesis of the converted chunk. Only F0 changes; spectral envelope (formants) and duration are kept. Measured: F0 x0.50 at -12, x2.00 at +12, duration unchanged. (The vocoder's own F0 input cannot do this: HiFT follows the mel's harmonics and ignored a rescaled excitation.)
- F0 preservation (default off): ON replaces the converted voice's F0 contour with the live speaker's contour (moved into the target register), OFF keeps the model's native contour.
- Timbre blend 0..1: interpolation between the target speaker embedding (1.0) and the live speaker's own CAMPPlus embedding (0.0).
- Chunk 80-320 ms, crossfade 5-chunk/2 ms, CFM steps 4-15, VAD threshold 0.05-0.95, output gain 0-2.

## Measured results (RTX 3050 Laptop 4 GB, driver 616.56, torch 2.14.0+cu126, fp16)
Backend reported in telemetry: `pytorch-cuda+cudagraph`, precision fp16. `torch.compile` is NOT used (Triton is not installed on this Windows setup); telemetry reports that fallback reason. No ONNX Runtime / DirectML path is used.
- Conversion is real: CAMPPlus cosine to target 0.436 (source) -> 0.887 (converted); independent encoder gains in the evaluation below.
- Controls (evidence/run_evidence.py): CFM steps 4/8/15 -> exactly 4/8/15 estimator calls per chunk; timbre blend 1.0/0.5/0.0 -> cos to target 0.89/0.83/0.65; pitch -12/0/+12 -> median F0 99.7/197.8/399.0 Hz; F0 toggle contour corr 0.953 on vs 0.797 off; VAD: silence and faint noise -> 0 model calls, quiet speech gated at threshold 0.95 (0 calls) but passed at 0.05 (30 calls); SOLA: constant hop and no boundary discontinuities.
- Live path with real speech (virtual mic via the CABLE pair -> engine -> Speaker, output tap): chunk 320 ms, 4 CFM steps, inference ~195 ms (< 320 ms chunk), end-to-end latency (mic ADC timestamp -> output DAC timestamp) 549-690 ms, median 681 ms, 0 dropped chunks, 12 underruns, converted similarity to target 0.876 vs 0.436 for the source.
- Production wiring (physical mic -> CABLE Input) was exercised with silence only: stream opened, VAD skipped all chunks, 0 model calls, e2e ~350 ms at a 200 ms chunk (pure buffering).

### Added latency target (<150 ms): NOT MET
Isolated model time is ~135-158 ms per window at 4 steps even with CUDA graphs; the chunk itself is >= 80 ms, plus 25 ms lookahead, 40 ms output prebuffer and WASAPI buffering. The only configuration that ran without dropped chunks here is chunk 320 ms / 4 steps, at a measured end-to-end latency of 549-690 ms (median 681 ms). That is the delivered figure; it is 4-5x the requested limit. Chunk 240 ms with 4 steps ran at 190-200 ms inference and its latency drifted upward (no headroom). Lowering chunk size or CFM steps further breaks real time or quality; I did not do it to make a number pass. A smaller/faster backbone or a stronger GPU would be needed.

## Evaluation on varied speech (evaluation/)
Criteria fixed before running (preregistered_thresholds.json, hash-locked): independent (resemblyzer) similarity gain >= 0.10, CAMPPlus cos >= 0.60 and gain >= 0.15, log-F0 contour correlation >= 0.55, energy-envelope correlation >= 0.60, duration ratio 0.95-1.05; set passes at >= 75%.
- Main run: 4 targets x 4 sources, 5+ recordings per role, both gender ranges: **10/16 passed (62.5%) -> the set FAILS the pre-registered rule.** All 16 met the independent similarity gain; 15/16 met all similarity criteria; 6 failed the F0 contour criterion (mostly low-pitched male targets, incl. cross-gender pairs); one also failed on a very deep source voice (74 Hz).
- Held-out experiment (fresh recordings, criteria fixed before running, all female-range voices): default 5/10 pass, F0-preserve 6/10 pass. F0-preserve fixes the contour (10/10 vs 7/10 at >=0.55) but loses voice similarity (6/10 vs 9/10). The pre-registered decision rule (adopt only if +0.20 pass rate) was not met, so the default is unchanged.
- Judge/critic: an LLM judge scored each conversion from the measured metrics only (no audio-capable judge is available here; it cannot hear). It agreed with the thresholds on 15/16 rows; row 1 (borderline contour 0.55) it rated more conservatively. See evaluation/judge_results.json. The thresholds, not the judge, decide pass/fail, and **the judge did not pass this build as a whole**: the registered set criterion failed.

## Voice creation (offline, never in the real-time path)
Prompt -> Qwen2.5-0.5B-Instruct (CPU) writes search queries and ranks candidates -> public-domain/CC audio (Internet Archive, Wikimedia Commons, direct audio links) -> VAD-selected 6.5-14 s clip -> mechanical voice-fit check (median F0 vs requested gender/pitch) -> existing `LoadReference` validation -> audition in the GUI -> Accept (saves preset, download) or "Not quite" + feedback (re-runs excluding rejected sources). Prompts naming a real person are refused. Refuses to start while the live engine runs (CPU contention).
Verified through the HTTP API on the real engine: (1) "deep slow calm male storyteller" -> LLM queries, 30 candidates, a female-range clip (181 Hz) rejected by the fit check, a timeout skipped, a Moby Dick clip (median F0 109 Hz, male/low) loaded via `LoadReference`; (2) accept saved the preset and the download was byte-identical to the clip; (3) "Not quite" + feedback re-ran and returned a different source (F0 128 Hz). Known weaknesses: the 0.5B model only gets a weak say in ranking, the fit check is a pitch heuristic (not age/accent), a rejected source excludes only that file (the same reader can come back), and the feedback word "deeper" does not map to the "low pitch" rule. Quality of the voice is for you to judge by listening; I could not.

## Cache-empty weights download
Verified part: `download_weights.py` was run with `HF_HOME` pointed at a new empty directory (C:\Users\jeffr\hf_clean_test, logs were written to clean.out.log, which I later deleted by mistake with a wildcard cleanup; the downloaded file in that directory still exists, the log does not). It fetched `s3gen.safetensors` (1,056,484,620 bytes) from Hugging Face and then the Qwen2.5-0.5B-Instruct files, and printed "Chatterbox weights ready". Not verified: a full `run.bat` on a machine with no venv (that would also reinstall every package, including 2.7 GB of torch, which I did not repeat). The first attempt of this download was interrupted when my tooling restarted and was re-run; the Hugging Face xet backend stalled at 0 bytes on this network, so `download_weights.py` sets `HF_HUB_DISABLE_XET=1`.

## Latency floor (measured 2026-11-17, RTX 3050 4 GB, 4 CFM steps, one run per cell, offline ProcessOffline on a 6 s clip)
| prompt_s (prompt tokens) | chunk 160 ms | chunk 320 ms |
|---|---|---|
| 3.0 (75, shipped default) | 181 ms | 196 ms |
| 1.5 (37) | 158 ms | 201 ms |
| 0.75 (18) | 154 ms | 160 ms |
Shortening the reference prompt saves at most about 30 ms per window and lowered the output RMS (0.069 to 0.056 at chunk 160), so it would need the pre-registered evaluation again before use. It is not adopted. The earlier evidence/timing.json sweep read 252-311 ms for the same settings, so inference time varies with GPU load between runs. Estimate (a model, not a measurement): end-to-end latency is at least chunk + lookahead + inference + buffering, which is over 300 ms even at chunk 160 and inference 154 ms. The 150 ms target is therefore not reachable with this decoder on this GPU; it needs a different decoder (fewer or cheaper flow steps), not a parameter change. Measured end-to-end latency, chunk 320 ms: evidence/live_test_320.log (live cable run) min 548.7, mean 671.5, median 681.2, max 689.7 ms with 12 underruns; a separate offline-to-cable hop test on 2026-11-17 read 676 ms with 22 underruns. The 565-722 ms range quoted earlier in this README is not traced to a raw log.

## Known limits
- Added latency far above 150 ms (see above). Real-time only at chunk >= ~300 ms with 4 steps on this GPU.
- pitch/F0-preserve use CPU WORLD processing (+40-130 ms per chunk) so they reduce real-time headroom.
- requirements.txt: exact versions from the working venv. torch 2.14.0+cu126 came from an existing identical install on this machine (re-downloading 2.7 GB was impractical on this link); a clean-room reinstall of the torch wheel was not re-run.
- First model run per chunk size triggers CUDA-graph capture (warmed before streaming starts; changing chunk size mid-stream pauses ~1-2 s).
- GUI verified structurally (DOM, controls, WebSocket round trips); I could not view screenshots in this environment.

## Rejected experiment: classifier-free guidance off
evidence/cfg_experiment.py (6 pairs, same engine, 2026-11-16): dropping the unconditional branch did not speed inference (mean 254.4 ms vs 256.5 ms with it) and lowered similarity to the target (CAMPPlus 0.767 vs 0.808, Resemblyzer 0.856 vs 0.875). Not adopted. The remaining latency gap (about 570-720 ms end to end vs 150 ms) is chunking (320 ms) plus roughly 250 ms inference plus buffering, so it needs a smaller chunk and a faster decoder, not this change.
