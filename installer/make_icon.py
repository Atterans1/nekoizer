"""Write installer/mimi.ico from the mascot sprite (Windows icon, 16 to 256 px, nearest-neighbour so the pixels stay crisp)."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ICO = ROOT / "installer" / "mimi.ico"
sprite = Image.open(ROOT / "ui" / "static" / "art" / "mimi_idle.png").convert("RGBA")
side = max(sprite.size)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(sprite, ((side - sprite.width) // 2, (side - sprite.height) // 2))
canvas.resize((256, 256), Image.NEAREST).save(ICO, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
