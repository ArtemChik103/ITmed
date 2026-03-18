"""Shared checkpoint loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    """Load a PyTorch checkpoint with backward-compatible fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
