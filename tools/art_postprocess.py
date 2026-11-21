"""Quantize raw art to the fixed palette: nearest-neighbour downscale to a pixel grid, key out the background, save transparent PNG."""
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = {"hero_scene": (128, 64), "og_banner": (128, 64), "bg_tile": (32, 32)}
SCENES = ("hero_scene", "og_banner", "bg_tile")  # full-frame art: no background keyed out
DEFAULT_GRID = (48, 48)
BG = np.array([0xFF, 0xF4, 0xF8])
BG_TOL = 60


def Palette() -> np.ndarray:
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in json.loads((ROOT / "art/palette.json").read_text())])


def FloodKey(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    near = np.abs(rgb.astype(int) - BG).sum(axis=2) < BG_TOL
    keep = np.ones((h, w), bool)
    seen = np.zeros((h, w), bool)
    stack = [(y, x) for y in (0, h - 1) for x in range(w)] + [(y, x) for x in (0, w - 1) for y in range(h)]
    while stack:
        y, x = stack.pop()
        if seen[y, x] or not near[y, x]:
            continue
        seen[y, x] = True
        keep[y, x] = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                stack.append((ny, nx))
    return keep


def Process(name: str, scale: int) -> None:
    img = Image.open(ROOT / "art/raw" / f"{name}.jpg").convert("RGB")
    gw, gh = GRID.get(name, DEFAULT_GRID)
    small = np.array(img.resize((gw, gh), Image.BOX))
    pal = Palette()
    idx = np.abs(small[:, :, None, :].astype(int) - pal[None, None]).sum(axis=3).argmin(axis=2)
    out = pal[idx].astype(np.uint8)
    alpha = np.full((gh, gw), 255, np.uint8)
    if name not in SCENES:
        alpha = (FloodKey(small) * 255).astype(np.uint8)
    rgba = np.dstack([out, alpha])
    Image.fromarray(rgba, "RGBA").resize((gw * scale, gh * scale), Image.NEAREST).save(ROOT / "ui/static/art" / f"{name}.png", optimize=True)


if __name__ == "__main__":
    (ROOT / "ui/static/art").mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "art/raw").glob("*.jpg")):
        Process(p.stem, 4)
