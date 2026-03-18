"""Stable frontend labels for the optional anatomy overlay."""
from __future__ import annotations

from collections.abc import Sequence

RAW_KEYPOINT_LABELS = ("re", "ry", "rc", "rh", "le", "ly", "lc", "lh")
NEUTRAL_KEYPOINT_LABELS = tuple(f"P{index}" for index in range(1, len(RAW_KEYPOINT_LABELS) + 1))


def overlay_keypoint_labels(*, use_neutral_labels: bool = True) -> list[str]:
    """Return the compact label set used by the frontend overlay."""
    labels: Sequence[str] = NEUTRAL_KEYPOINT_LABELS if use_neutral_labels else RAW_KEYPOINT_LABELS
    return list(labels)


def raw_keypoint_order_text() -> str:
    """Return the raw training order for docs and education-mode explanations."""
    return ", ".join(RAW_KEYPOINT_LABELS)
