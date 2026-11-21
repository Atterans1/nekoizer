"""Compose the square (1:1) hero picture from the generated square sprites, on a 128 x 128 pixel grid shown at 4x.
Nothing is resized except by whole numbers (2x down, 4x up, nearest neighbour), so no sprite is stretched.  python tools/compose_hero.py"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "ui" / "static" / "art"
GRID, SCALE = 128, 4
PALETTE = {name: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for name, h in {
    "plum": "#4A2B5A", "cream": "#FFF4F8", "lilac": "#E8DEFF", "white": "#FFFFFF", "pink": "#FF8FC7", "mint": "#7FE5C4", "sky": "#8EC9FF",
    "butter": "#FFE59A", "blush": "#FFC9E3", "lilac2": "#D8C8F5", "lilac3": "#C9B3F0"}.items()}
SKY_BANDS = ("lilac", "lilac", "lilac2", "lilac", "cream", "cream", "blush", "cream")


def Sprite(name: str, shrink: int = 1) -> Image.Image:
    """A 192 x 192 sprite is a 48 x 48 pixel-art grid at 4x: undo the 4x exactly, then optionally halve it again."""
    image = Image.open(ART / f"{name}.png").convert("RGBA")
    assert image.width == image.height == 192, f"{name} is {image.size}, expected a square 192 x 192 sprite"
    grid = image.resize((48, 48), Image.NEAREST)
    return grid if shrink == 1 else grid.resize((48 // shrink, 48 // shrink), Image.NEAREST)


def Main() -> None:
    canvas = Image.new("RGBA", (GRID, GRID), PALETTE["cream"] + (255,))
    draw = ImageDraw.Draw(canvas)
    band = 76 // len(SKY_BANDS)
    for i, colour in enumerate(SKY_BANDS):
        draw.rectangle([0, i * band, GRID, (i + 1) * band], fill=PALETTE[colour])
    for x, y, c in ((12, 10, "butter"), (104, 14, "pink"), (30, 40, "sky"), (112, 52, "mint"), (60, 8, "white"), (92, 36, "butter"), (8, 62, "pink")):
        draw.rectangle([x, y - 1, x, y + 1], fill=PALETTE[c]); draw.rectangle([x - 1, y, x + 1, y], fill=PALETTE[c])
    floor = 100
    draw.rectangle([0, floor, GRID, GRID], fill=PALETTE["pink"])
    draw.rectangle([0, floor, GRID, floor + 1], fill=PALETTE["plum"])
    for x in range(0, GRID, 8):
        draw.rectangle([x, floor + 8, x + 3, floor + 9], fill=PALETTE["blush"])
    mimi = Sprite("mimi_talking")
    canvas.alpha_composite(mimi.resize((96, 96), Image.NEAREST), ((GRID - 96) // 2, floor - 96 + 2))  # 2x up: whole-number scale, still square
    for name, x, y in (("icon_headphones", 4, 52), ("icon_waveform", 100, 56), ("icon_wand", 6, 20), ("icon_cable", 98, 22)):
        canvas.alpha_composite(Sprite(name, 2), (x, y))
    out = canvas.resize((GRID * SCALE, GRID * SCALE), Image.NEAREST)
    assert out.width == out.height
    for path in (ART / "hero_scene.png", ROOT / "site" / "art" / "hero_scene.png"):
        out.save(path, optimize=True)
    og = Image.new("RGBA", (1200, 630), PALETTE["lilac"] + (255,))  # social preview: the square hero centred on a lilac field, no resizing
    og.alpha_composite(out, ((1200 - out.width) // 2, (630 - out.height) // 2))
    og.convert("RGB").save(ROOT / "site" / "art" / "og_banner.png", optimize=True)
    print(json.dumps({"size": out.size, "ratio": out.width / out.height, "og": og.size}))


if __name__ == "__main__":
    Main()
