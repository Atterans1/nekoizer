# Specification (verbatim from the owner, 2026-11-17)

Priority: output quality over minimum latency. Keep the defaults stated in the spec (220 ms block, 45 ms SOLA overlap, 10 diffusion steps, CFG 0.65, 2.0 s left context). Where these conflict with the 'ultra-low-latency' goal, keep the defaults and report the measured latency honestly.

Anti-workaround constraints:
- Do not satisfy a requirement with a stub, a hardcoded value, or a renamed parameter.
- Do not replace listed components (Seed-VC, Silero VAD, FCPE or RMVPE, CAMPPlus, SOLA) with substitutes. If one cannot be obtained or run, stop and report it.
- Do not claim the app works without running it.

Definition of done:
- The app launches and routes live microphone input through the pipeline to CABLE Input.
- Each listed control is read by the inference or output path (search confirms this for every parameter name).
- The report includes measured end-to-end latency and underrun count from an actual run, plus the conflict resolutions.

1. Input chain (audio/input_chain.py)
- Open the microphone in raw WASAPI Exclusive or Low-Latency Shared mode, bypassing Windows Audio Enhancements and OS noise suppression.
- Apply an 80 Hz high-pass filter.
- Apply a soft 2.5:1 compressor with automatic makeup gain.
- Run Silero VAD (silero_vad ONNX) with a 120 ms hang-time tail. Output true digital silence when inactive.

2. Prosody and inference (engine/seedvc_expressive.py)
- Use Seed-VC (Plachtaa/seed-vc, seed-uvit-tat-xlsr-tiny, optional v2 checkpoints) with --f0-condition True.
- Extract F0 with FCPE or RMVPE.
- Add an F0 Dynamic Range Expander slider, 0.8x to 1.4x around the speaker's median pitch.
- Multi-emotion reference bank: up to 4 clips per character (Neutral, Hyped/Laughing, Whisper/Soft, Angry/Intense). Pre-compute and cache CAMPPlus embeddings and acoustic latents on load. Blend or switch conditioning based on input RMS energy and pitch variance, or on manual hotkeys.
- Rolling left-context ring buffer of 2.0 s (adjustable 1.5 to 2.5 s).
- Diffusion steps default 10 (range 6 to 16).
- Inference CFG rate default 0.65 (range 0.0 to 1.0).
- Controls: semitone pitch shift (-12 to +12), formant shift (-2.0 to +2.0), dry/wet timbre mix.

3. Output chain (audio/output_chain.py)
- SOLA: 220 ms block size, 45 ms crossfade overlap (0.04 to 0.05 s).
- Safety auto-balancer: track inference time against block time. When inference approaches 85% of block time, drop diffusion steps by 2 for 3 seconds, then restore.
- Route output to CABLE Input (VB-Audio Virtual Cable). Add a zero-latency sidetone monitor toggle with adjustable monitor volume.

4. UI (ui/)
- Minimal dark interface.
- Character and multi-emotion reference loader: drag-and-drop .wav slots, plus a built-in 10 s recorder with live waveform and clipping meter.
- Tone rack: pitch shift, formant shift, F0 expressiveness multiplier, CFG rate (0.65), diffusion steps (10), left context (2.0 s), block size (220 ms).
- One-click presets: Natural Conversational, High-Energy Gaming / Ragebait, Soft / Whisper, and a fourth preset that is cut off in the source ("Ultra-Low-").
