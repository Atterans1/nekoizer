"""Draw an input-versus-output spectrogram picture for the website from two real audio files (nothing synthetic).
  python tools/make_demo_plot.py INPUT.wav OUTPUT.wav OUT.png [seconds]"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy.signal import resample_poly, stft

PLUM, CREAM = (74, 43, 90), (255, 244, 248)
RAMP = [(255, 244, 248), (232, 222, 255), (201, 179, 240), (255, 143, 199), (74, 43, 90)]  # quiet to loud
W, H, TOP, FMAX = 1000, 220, 34, 8000


def Mono(path, seconds):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr) if sr % 16000 == 0 or 16000 % sr == 0 else resample_poly(audio, 16000, sr)
    return audio[: int(16000 * seconds)]


def Spec(audio):
    _, _, z = stft(audio, fs=16000, nperseg=512, noverlap=384)
    db = 20 * np.log10(np.abs(z) + 1e-6)
    db = np.clip(db, db.max() - 70, db.max())
    return (db - db.min()) / (db.max() - db.min())


def Colour(v):
    pos = v * (len(RAMP) - 1)
    i = min(int(pos), len(RAMP) - 2)
    f = pos - i
    return tuple(int(RAMP[i][k] * (1 - f) + RAMP[i + 1][k] * f) for k in range(3))


def Panel(draw, canvas, audio, y0, label):
    spec = Spec(audio)
    spec = spec[: int(spec.shape[0] * FMAX / 8000)]
    img = Image.new("RGB", (spec.shape[1], spec.shape[0]))
    img.putdata([Colour(spec[spec.shape[0] - 1 - r, c]) for r in range(spec.shape[0]) for c in range(spec.shape[1])])
    canvas.paste(img.resize((W - 20, H - TOP - 8), Image.NEAREST), (10, y0 + TOP))
    draw.rectangle([10, y0 + TOP, W - 10, y0 + H - 8], outline=PLUM, width=3)
    draw.text((12, y0 + 8), label, fill=PLUM)


def Main(argv):
    src, out_wav, png = argv[:3]
    seconds = float(argv[3]) if len(argv) > 3 else 6.0
    a, b = Mono(src, seconds), Mono(out_wav, seconds)
    canvas = Image.new("RGB", (W, H * 2 + 6), CREAM)
    draw = ImageDraw.Draw(canvas)
    Panel(draw, canvas, a, 0, "BEFORE: the microphone input (spectrogram, 0 to 8 kHz)")
    Panel(draw, canvas, b, H + 6, "AFTER: the converted voice, same words (spectrogram, 0 to 8 kHz)")
    canvas.save(png, optimize=True)
    print(png, canvas.size)


if __name__ == "__main__":
    Main(sys.argv[1:])
