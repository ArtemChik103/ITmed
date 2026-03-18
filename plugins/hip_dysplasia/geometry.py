"""Low-level geometry helpers for optional landmark post-processing."""
from __future__ import annotations

import math
from typing import NamedTuple


class LineEquation(NamedTuple):
    """Normalized line representation ax + by + c = 0."""

    a: float
    b: float
    c: float


def point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Return Euclidean distance between two image-space points."""
    return math.dist(point_a, point_b)


def midpoint(point_a: tuple[float, float], point_b: tuple[float, float]) -> tuple[float, float]:
    """Return the midpoint between two points."""
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def line_from_points(point_a: tuple[float, float], point_b: tuple[float, float]) -> LineEquation:
    """Build a normalized line equation from two points."""
    if point_a == point_b:
        raise ValueError("Cannot build a line from two identical points.")

    x1, y1 = point_a
    x2, y2 = point_b
    a = y1 - y2
    b = x2 - x1
    c = (x1 * y2) - (x2 * y1)
    norm = math.hypot(a, b)
    if norm == 0.0:
        raise ValueError("Cannot normalize a degenerate line.")
    return LineEquation(a / norm, b / norm, c / norm)


def angle_between_lines_deg(line_a: LineEquation, line_b: LineEquation) -> float:
    """Return the acute angle between two lines in degrees."""
    dot_product = max(min((line_a.a * line_b.a) + (line_a.b * line_b.b), 1.0), -1.0)
    angle = math.degrees(math.acos(abs(dot_product)))
    return float(angle)


def perpendicular_projection(
    point: tuple[float, float],
    line: LineEquation,
) -> tuple[float, float]:
    """Project a point onto a line."""
    x0, y0 = point
    distance = (line.a * x0) + (line.b * y0) + line.c
    projected_x = x0 - (line.a * distance)
    projected_y = y0 - (line.b * distance)
    return (float(projected_x), float(projected_y))


def scaled_distance_mm(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    pixel_spacing_mm: list[float] | tuple[float, float] | None,
) -> float | None:
    """Convert pixel distance into millimeters using anisotropic row/column spacing."""
    if not pixel_spacing_mm or len(pixel_spacing_mm) < 2:
        return None

    row_spacing = float(pixel_spacing_mm[0])
    col_spacing = float(pixel_spacing_mm[1])
    dx_mm = (point_a[0] - point_b[0]) * col_spacing
    dy_mm = (point_a[1] - point_b[1]) * row_spacing
    return float(math.hypot(dx_mm, dy_mm))
