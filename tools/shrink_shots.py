"""Shrink the website screenshots: halve the 2x captures, then quantise to 256 colours (the UI is flat colours, so this is lossless to the eye)."""
from pathlib import Path

from PIL import Image

SHOTS = Path(__file__).resolve().parent.parent / "site" / "shots"
for name, width in (("nekoizer-before", 1280), ("nekoizer-after", 1280), ("nekoizer-phone", 390), ("nekoizer-spectrogram", 1000)):
    path = SHOTS / f"{name}.png"
    image = Image.open(path).convert("RGB")
    if image.width != width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image.quantize(256, method=Image.MEDIANCUT, dither=Image.NONE).save(path, optimize=True)
    print(name, image.size, path.stat().st_size // 1024, "KB")
