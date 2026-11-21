# Art provenance and licences

| Asset | Source | Licence status |
|---|---|---|
| `ui/static/art/*.png`, `site/art/*.png` (Mimi sprites, icons, hero scene, background tile) | Generated 2026-11-19 with FLUX.1 schnell (`black-forest-labs-flux-1-schnell`) on RunPod's pay-per-request public endpoint, then reduced to the fixed palette in `art/palette.json` by `tools/art_postprocess.py` (box downscale, palette snap, background keyed out) | FLUX.1 schnell is published by Black Forest Labs under Apache-2.0 as far as I know. **Unverified here:** I did not re-read the model card or RunPod's terms this session, and generated output may not be copyrightable in every jurisdiction. Read both before selling anything. |
| `site/art/hero_scene.png`, `site/art/og_banner.png` | Composed in code by `tools/compose_hero.py` from the square sprites above (whole-number scales only, so nothing is stretched) | Same as the sprites it uses |
| `site/shots/*` | Screenshots of the app running on the author's PC (19 Nov 2026), a spectrogram drawn by `tools/make_demo_plot.py` from a real conversion, and Fretwork editor renders produced by that plugin's own test tool | The author's own work. The spectrogram is a picture of audio, not audio. |
| `installer/mimi.ico` | Derived from `mimi_idle.png` by `installer/make_icon.py` | Same as above |
| Fonts | None bundled: the UI and site use the system monospace stack (Cascadia Mono, Consolas) | n/a |

Cost: 14 images at 0.003 USD each, 0.042 USD in total as the endpoint reported it (one probe image plus 13 assets). Prompts and seeds: the shared style prefix and per-asset prompts are in `art/PROVENANCE.json`.
The raw JPEG outputs are in `art/raw/`. Nobody has looked at the final sprites at pixel level: the palette, transparency and size checks are numeric.
