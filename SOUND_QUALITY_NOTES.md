# Making the Nekoizer voice sound like a real human

Engine: Seed-VC (app_seedvc.py, .venv-seedvc). Status: MEASURED = observed on this machine, INFERENCE = not tested.

## What is already true (MEASURED)
- Seed-VC xlsr-tiny checkpoint has zero F0 weights. F0 conditioning is impossible, so pitch and tone cannot be steered by the model. Flat tonality is fixed only after the model, with WORLD contour transfer (pyworld).
- At spec defaults (220 ms block, 45 ms overlap, 10 steps, CFG 0.65, 2.0 s context) inference is 394 ms offline and 491 ms live per block. That is about 2x too slow: 741 underruns and 4.5 s mean latency in the live run. A model that cannot keep up cannot sound human, because glitches and dropouts dominate.
- Step cost offline: 6 steps 264 ms, 10 steps 394 ms, 16 steps 594 ms.

## What the program needs, in priority order
1. Real-time first. Every quality gain is lost if blocks underrun. Try block 320 ms with 6 steps (INFERENCE: should fit the budget, untested). Keep the output ring from growing past a few hundred ms.
2. Reference voice. Use 10 to 30 s of clean, dry, single-speaker target speech with natural intonation and no music, reverb or noise. A flat or whispered reference gives a flat voice. Pick the reference for expressiveness. (INFERENCE from how zero-shot VC works.)
3. Clean input. Mic through WASAPI Exclusive, enhancements off (MEASURED working). Gate or denoise before Seed-VC, not after. Noise in becomes artifacts out.
4. Pitch and tone post-stage. WORLD F0 contour transfer so speech is not monotone, with a limit on the shift so formants do not sound chipmunk or robotic. Keep the transfer subtle and test on varied speech, not one sentence.
5. Context window. Longer context (2.0 s default) helps coherence but costs time. Reduce only if latency forces it.
6. Seam handling. Crossfade overlap between blocks (45 ms default). Audible clicks at block joins are the usual cause of a robotic sound.
7. Output loudness. Normalise level and add a light de-esser or soft limiter. Harsh sibilants read as synthetic.
8. Hardware headroom. RTX 3050 4 GB is the limit. fp16, CUDA graphs and fewer steps are the levers. A bigger GPU is the cleanest real fix. torch.compile is unavailable (no triton on Windows).

## How to judge it (do not tune after seeing scores)
- Pre-register the test set and pass rule first. The earlier rule live_run/tone/PICK_RULE.md is hash-locked: do not edit it after scores are seen.
- Test varied speech: questions, laughs, whispers, fast and slow, numbers. One clean sentence proves nothing.
- Report latency, underruns, and a human listening score. Report failures plainly.

## Known gaps
- No claim yet that it sounds human. Nobody has listened to the latest build against a real recording.
- Discord Rich Presence and the live critic are not in the current build.

## Added: upstream guidance (SOURCE: Seed-VC README, https://github.com/Plachtaa/seed-vc, search snippet only, not the full page)
- Real-time diffusion steps: upstream says 4 to 10 is the usual range for fastest inference. Our 6-step trial sits inside it.
- Inference CFG rate: upstream says it has a subtle effect on output, default 0.7, and setting it to 0.0 gives about 1.5x speed-up. Our default is 0.65. Try CFG 0.0 as the cheapest speed lever (MEASURED effect on our hardware: not yet; the 1.5x is upstream's figure). Judge by ear that quality holds.
- The upstream real-time GUI (real-time-gui.py) splits audio into blocks with a context window and crossfade, the same design as ours. Its parameters are block time, extra context, crossfade length and diffusion steps. Tune these four, not the model.
- Paper: Seed-VC uses a diffusion transformer with external timbre shifter training to cut timbre leakage (https://arxiv.org/abs/2411.09943). Leakage means the output keeps traces of the source speaker, so choose a reference whose timbre is clearly distinct and consistent.

## Added: next experiments, in order (each is a plan, nothing here is tested)
1. Speed pass: CFG 0.0, 6 steps, block 320 ms. Log underruns and mean latency. Pass = zero underruns over 5 minutes.
2. Quality pass at the speed that passes: compare 6, 8 and 10 steps on the same pre-registered clips.
3. Reference pass: try 3 different reference clips of the same target voice. Keep the one with the most natural prosody.
4. Pitch pass: WORLD contour transfer on and off, same clips, blind listening.
5. Seam pass: vary crossfade length. Listen for clicks at joins.
Record every run's settings and result in one table, including failures.
