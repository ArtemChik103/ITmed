"""Overlay renderer for optional anatomy keypoints in education mode."""
from __future__ import annotations

import os
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    """Load a true TTF font with full Cyrillic support, preventing tiny bitmap fallback rectangles."""
    # 1. Matplotlib bundled DejaVuSans (guaranteed across Windows, Linux and Streamlit Cloud)
    try:
        import matplotlib

        font_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        for fname in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
            fpath = os.path.join(font_dir, fname)
            if os.path.exists(fpath):
                return ImageFont.truetype(fpath, size=size)
    except Exception:
        pass

    # 2. Linux system paths
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass

    # 3. Windows system fonts
    for fname in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(fname, size=size)
        except OSError:
            pass

    return ImageFont.load_default()


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
    label_font = _load_font(max(16, int(base_size * 0.022)))
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


def render_anatomical_landmarks_overlay(
    image: Image.Image,
    result: dict | None = None,
    keypoints: Sequence[tuple[float, float]] | None = None,
) -> Image.Image:
    """Render clinical pediatric orthopedic landmarks, lines, and quadrant grid."""
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = canvas.size
    base_size = max(w, h)

    font_main = _load_font(max(16, int(base_size * 0.022)))

    metrics = (result.get("metrics") or {}) if result else {}
    is_pathology = bool(result.get("disease_detected")) if result else False
    tonnis_g = int(round(float(metrics.get("tonnis_grade", 2 if is_pathology else 0))))
    reimers_val = float(metrics.get("reimers_index_pct", 32.6 if is_pathology else 13.0))
    alpha_deg = float(metrics.get("acetabular_angle_deg", 29.5 if is_pathology else 21.0))

    mid_x = w // 2
    y_hilg = int(h * 0.532)

    # Y-cartilages (triradiate cartilages / дно вертлужной впадины)
    x_yr = int(mid_x - w * 0.155)
    x_yl = int(mid_x + w * 0.155)

    # Perkins lines (passing through outer acetabular rim / наружный край крыши)
    x_pr = int(mid_x - w * 0.242)
    x_pl = int(mid_x + w * 0.242)

    # Roof rim Y coordinates derived from acetabular index alpha
    import math
    dy_r = int((x_yr - x_pr) * math.tan(math.radians(alpha_deg)))
    y_rr = max(int(h * 0.15), y_hilg - dy_r)
    dy_l = int((x_pl - x_yl) * math.tan(math.radians(alpha_deg)))
    y_rl = max(int(h * 0.15), y_hilg - dy_l)

    # Femoral head / proximal metaphysis centers
    if tonnis_g >= 2 or (is_pathology and reimers_val >= 25.0):
        # Displaced laterally into infero-lateral Ombrédanne quadrant (subluxation)
        x_hr = x_pr - int(w * 0.025)
        y_hr = y_hilg + int(h * 0.075)
        x_hl = x_pl + int(w * 0.025)
        y_hl = y_hilg + int(h * 0.075)
    else:
        # Safe anatomical position inside infero-medial quadrant (normal / Grade 1)
        x_hr = x_pr + int(w * 0.038)
        y_hr = y_hilg + int(h * 0.080)
        x_hl = x_pl - int(w * 0.038)
        y_hl = y_hilg + int(h * 0.080)

    # 1. Tint target Ombrédanne quadrant
    if tonnis_g >= 2:
        draw.rectangle([int(x_pr - w * 0.12), y_hilg, x_pr, int(y_hilg + h * 0.18)], fill=(239, 68, 68, 38))
        draw.rectangle([x_pl, y_hilg, int(x_pl + w * 0.12), int(y_hilg + h * 0.18)], fill=(239, 68, 68, 38))
    else:
        draw.rectangle([x_pr, y_hilg, x_yr, int(y_hilg + h * 0.18)], fill=(16, 185, 129, 30))
        draw.rectangle([x_yl, y_hilg, x_pl, int(y_hilg + h * 0.18)], fill=(16, 185, 129, 30))

    # 2. Hilgenreiner line (H-line)
    draw.line([(int(w * 0.06), y_hilg), (int(w * 0.94), y_hilg)], fill=(0, 220, 255, 230), width=2)

    # 3. Perkins lines (P-lines)
    y_p_top = int(y_hilg - h * 0.22)
    y_p_bot = int(y_hilg + h * 0.24)
    draw.line([(x_pr, y_p_top), (x_pr, y_p_bot)], fill=(245, 158, 11, 230), width=2)
    draw.line([(x_pl, y_p_top), (x_pl, y_p_bot)], fill=(245, 158, 11, 230), width=2)

    # 4. Pelvic Midline (dashed)
    for y_step in range(int(h * 0.15), int(h * 0.85), 14):
        draw.line([(mid_x, y_step), (mid_x, y_step + 8)], fill=(148, 163, 184, 180), width=1)

    # 5. Acetabular roof incline lines
    roof_color = (239, 68, 68, 240) if alpha_deg > 25.0 else (16, 185, 129, 240)
    draw.line([(x_yr, y_hilg), (x_pr, y_rr)], fill=roof_color, width=3)
    draw.line([(x_yl, y_hilg), (x_pl, y_rl)], fill=roof_color, width=3)

    # Helper badge drawer with boundary safeguards
    def _draw_badge(x: int, y: int, text: str, col: tuple[int, ...], align_left: bool = True):
        bbox = draw.textbbox((0, 0), text, font=font_main)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = max(6, int(base_size * 0.006))
        if align_left:
            bx = max(pad, min(x, w - tw - pad * 2 - 4))
        else:
            bx = max(pad, min(x - tw - pad * 2, w - tw - pad * 2 - 4))
        by = max(pad, min(y, h - th - pad * 2 - 4))
        draw.rounded_rectangle(
            [bx, by, bx + tw + pad * 2, by + th + pad + 2],
            radius=4,
            fill=(15, 23, 42, 215),
            outline=col,
            width=1,
        )
        draw.text((bx + pad, by + pad // 2), text, font=font_main, fill=(248, 250, 252, 255))

    # Points: Y-cartilage
    r_pt = max(4, int(base_size * 0.007))
    for px, py in [(x_yr, y_hilg), (x_yl, y_hilg)]:
        draw.ellipse([px - r_pt, py - r_pt, px + r_pt, py + r_pt], fill=(0, 220, 255, 255), outline=(255, 255, 255, 255), width=2)
    _draw_badge(x_yr - 15, y_hilg + 8, "Y-хрящ (D)", (0, 220, 255, 200), align_left=False)
    _draw_badge(x_yl + 15, y_hilg + 8, "Y-хрящ (S)", (0, 220, 255, 200), align_left=True)

    # Points: Acetabular roof rims & angle
    for px, py in [(x_pr, y_rr), (x_pl, y_rl)]:
        draw.ellipse([px - r_pt, py - r_pt, px + r_pt, py + r_pt], fill=roof_color, outline=(255, 255, 255, 255), width=2)
    _draw_badge(x_pr - 10, y_rr - 22, f"Угол α = {alpha_deg:.1f}°", roof_color, align_left=False)
    _draw_badge(x_pl + 10, y_rl - 22, f"Угол α = {alpha_deg:.1f}°", roof_color, align_left=True)

    # Points: Femoral heads (centered below markers)
    head_color = (239, 68, 68, 255) if tonnis_g >= 2 else (16, 185, 129, 255)
    r_head = max(6, int(base_size * 0.012))
    for px, py, s_name in [(x_hr, y_hr, "Головка D"), (x_hl, y_hl, "Головка S")]:
        draw.ellipse([px - r_head - 4, py - r_head - 4, px + r_head + 4, py + r_head + 4], fill=(*head_color[:3], 50))
        draw.ellipse([px - r_head, py - r_head, px + r_head, py + r_head], fill=head_color, outline=(255, 255, 255, 255), width=2)
        h_label = f"{s_name}: Подвывих (Степень II)" if tonnis_g >= 2 else f"{s_name}: Норма"
        bbox = draw.textbbox((0, 0), h_label, font=font_main)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = max(6, int(base_size * 0.006))
        bx = max(pad, min(int(px - tw // 2), w - tw - pad * 2 - 4))
        by = int(py + r_head + 6)
        draw.rounded_rectangle(
            [bx, by, bx + tw + pad * 2, by + th + pad + 2],
            radius=4,
            fill=(15, 23, 42, 215),
            outline=head_color,
            width=1,
        )
        draw.text((bx + pad, by + pad // 2), h_label, font=font_main, fill=(248, 250, 252, 255))

    # Top-RIGHT anatomical legend (positioned cleanly in top-right, never colliding with top-left badge)
    legend_right_x = w - int(base_size * 0.025)
    _draw_badge(legend_right_x, int(base_size * 0.025), "— Линия Хильгенрейнера (H-Line)", (0, 220, 255, 220), align_left=False)
    _draw_badge(legend_right_x, int(base_size * 0.065), "— Линия Перкинса (P-Line)", (245, 158, 11, 220), align_left=False)
    _draw_badge(legend_right_x, int(base_size * 0.105), f"— Наклон крыши α = {alpha_deg:.1f}° (Тённис {tonnis_g})", roof_color, align_left=False)

    out = Image.alpha_composite(canvas, overlay)
    return out.convert("RGB")
