"""Overlay renderer for optional anatomy keypoints in education mode."""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from frontend.utils.font_loader import load_font as _load_font


def render_keypoint_overlay(
    image: Image.Image,
    keypoints: Sequence[tuple[float, float]],
    *,
    labels: Sequence[str] | None = None,
) -> Image.Image:
    """Draw compact circles and labels over the provided image."""
    canvas = image.convert("RGBA")
    if not keypoints:
        return canvas.convert("RGB")

    draw = ImageDraw.Draw(canvas, "RGBA")
    base_size = max(canvas.width, canvas.height)
    radius = max(5, int(base_size * 0.009))
    halo_radius = radius + max(3, int(base_size * 0.004))
    label_font = _load_font(max(12, int(base_size * 0.018)))
    label_padding_x = max(6, int(base_size * 0.007))
    label_padding_y = max(3, int(base_size * 0.003))
    offset_x = max(10, int(base_size * 0.012))
    offset_y = max(12, int(base_size * 0.014))

    for index, (x, y) in enumerate(keypoints):
        x_pos = float(x)
        y_pos = float(y)
        draw.ellipse(
            (x_pos - halo_radius, y_pos - halo_radius, x_pos + halo_radius, y_pos + halo_radius),
            fill=(0, 230, 255, 52),
        )
        draw.ellipse(
            (x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius),
            fill=(0, 230, 255, 240),
            outline=(255, 255, 255, 220),
            width=max(1, radius // 3),
        )

        if labels is None or index >= len(labels):
            continue

        label = str(labels[index])
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = int(text_box[2] - text_box[0])
        text_height = int(text_box[3] - text_box[1])
        left = min(
            max(0, int(x_pos + offset_x)),
            max(canvas.width - (text_width + label_padding_x * 2), 0),
        )
        top = min(
            max(0, int(y_pos - offset_y - label_padding_y)),
            max(canvas.height - (text_height + label_padding_y * 2), 0),
        )
        right = left + text_width + label_padding_x * 2
        bottom = top + text_height + label_padding_y * 2
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=max(6, int(base_size * 0.01)),
            fill=(9, 14, 20, 214),
            outline=(0, 230, 255, 180),
            width=1,
        )
        draw.text(
            (left + label_padding_x, top + label_padding_y - text_box[1]),
            label,
            font=label_font,
            fill=(248, 250, 252, 255),
        )

    return canvas.convert("RGB")
