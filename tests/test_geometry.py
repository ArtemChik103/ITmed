from __future__ import annotations

import math

import pytest

from plugins.hip_dysplasia.geometry import (
    angle_between_lines_deg,
    line_from_points,
    midpoint,
    perpendicular_projection,
    point_distance,
    scaled_distance_mm,
)


def test_geometry_distance_midpoint_and_projection():
    assert point_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert midpoint((0.0, 0.0), (4.0, 6.0)) == (2.0, 3.0)

    horizontal = line_from_points((0.0, 1.0), (4.0, 1.0))
    assert perpendicular_projection((2.0, 5.0), horizontal) == pytest.approx((2.0, 1.0))


def test_geometry_angle_and_scaled_distance():
    horizontal = line_from_points((0.0, 0.0), (4.0, 0.0))
    vertical = line_from_points((2.0, -1.0), (2.0, 5.0))

    assert angle_between_lines_deg(horizontal, vertical) == pytest.approx(90.0)
    assert scaled_distance_mm((0.0, 0.0), (3.0, 4.0), [0.5, 0.5]) == pytest.approx(2.5)
    assert scaled_distance_mm((0.0, 0.0), (3.0, 4.0), None) is None


def test_geometry_line_from_identical_points_raises():
    with pytest.raises(ValueError):
        line_from_points((1.0, 1.0), (1.0, 1.0))
