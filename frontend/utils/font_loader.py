"""Shared font loading for Pillow-based rendering."""
from __future__ import annotations

from PIL import ImageFont

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
    "segoeui.ttf",
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font with Cyrillic support, falling back to default."""
    for font_name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
