"""Base helpers for plugin implementations."""
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

from core.plugin_manager import IPlugin


class BasePlugin(IPlugin, ABC):
    """Thin base class for concrete plugins."""

    def preprocess(self, image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        return image
