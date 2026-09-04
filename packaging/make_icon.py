"""Generate the Morphify application icon.

Two overlapping face profiles inside a camera aperture ring — the "morph"
idea, plus what the app actually is. Written as code rather than checked in
as a binary so it stays tied to the palette in modules/ui_theme.py.

    python packaging/make_icon.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from modules.ui_theme import PALETTE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Morphify.ico")
SIZES = [16, 24, 32, 48, 64, 128, 256]

# Render large and downsample so the curves stay clean at small sizes.
CANVAS = 1024


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def head_silhouette(draw, cx: float, cy: float, scale: float, colour: tuple) -> None:
    """A simple head-and-shoulders mark centred on (cx, cy)."""
    head_r = CANVAS * 0.105 * scale
    head_cy = cy - CANVAS * 0.035 * scale
    draw.ellipse(
        [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
        fill=colour,
    )
    shoulder_w = CANVAS * 0.27 * scale
    shoulder_top = head_cy + head_r * 0.82
    draw.pieslice(
        [cx - shoulder_w / 2, shoulder_top,
         cx + shoulder_w / 2, shoulder_top + shoulder_w * 0.95],
        start=180, end=360, fill=colour,
    )


def draw_icon() -> Image.Image:
    bg_top = hex_to_rgb(PALETTE["surface_hi"])
    bg_bottom = hex_to_rgb(PALETTE["sidebar"])
    accent = hex_to_rgb(PALETTE["accent"])
    accent_hi = hex_to_rgb(PALETTE["accent_hi"])
    accent2 = hex_to_rgb(PALETTE["accent2"])
    text = hex_to_rgb(PALETTE["text"])

    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))

    # Vertical gradient backdrop, clipped to a rounded square.
    gradient = Image.new("RGBA", (1, CANVAS))
    for y in range(CANVAS):
        t = y / (CANVAS - 1)
        gradient.putpixel((0, y), tuple(
            int(bg_top[i] + (bg_bottom[i] - bg_top[i]) * t) for i in range(3)
        ) + (255,))
    gradient = gradient.resize((CANVAS, CANVAS))

    mask = Image.new("L", (CANVAS, CANVAS), 0)
    pad = CANVAS // 16
    ImageDraw.Draw(mask).rounded_rectangle(
        [pad, pad, CANVAS - pad, CANVAS - pad], radius=CANVAS // 5, fill=255)
    image.paste(gradient, (0, 0), mask)

    draw = ImageDraw.Draw(image)
    centre = CANVAS / 2

    # The two faces being morphed: a soft ghost behind, a solid one in front.
    ghost = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    head_silhouette(ImageDraw.Draw(ghost), centre - CANVAS * 0.058,
                    centre + CANVAS * 0.012, 0.95, accent2 + (170,))
    ghost = ghost.filter(ImageFilter.GaussianBlur(CANVAS // 110))
    image.alpha_composite(ghost)

    head_silhouette(draw, centre + CANVAS * 0.052, centre + CANVAS * 0.012,
                    0.95, text + (255,))

    # Aperture ring with blade ticks.
    ring_pad = CANVAS // 5
    draw.ellipse(
        [ring_pad, ring_pad, CANVAS - ring_pad, CANVAS - ring_pad],
        outline=accent + (255,), width=CANVAS // 24,
    )
    r_outer = (CANVAS - 2 * ring_pad) / 2
    for index in range(6):
        angle = math.radians(index * 60 + 15)
        x0 = centre + math.cos(angle) * r_outer * 0.995
        y0 = centre + math.sin(angle) * r_outer * 0.995
        x1 = centre + math.cos(angle) * r_outer * 1.16
        y1 = centre + math.sin(angle) * r_outer * 1.16
        draw.line([x0, y0, x1, y1], fill=accent_hi + (255,), width=CANVAS // 42)

    return image


def main() -> int:
    icon = draw_icon()
    icon.save(OUT, format="ICO", sizes=[(size, size) for size in SIZES])
    png = OUT.replace(".ico", ".png")
    icon.resize((256, 256), Image.LANCZOS).save(png)

    # Remove the icon from the previous name so stale files do not get
    # picked up by the packaging scripts.
    for stale in ("DeepLiveCam.ico", "DeepLiveCam.png"):
        path = os.path.join(HERE, stale)
        if os.path.exists(path):
            os.remove(path)

    print(f"wrote {OUT}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
