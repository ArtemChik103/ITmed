"""Compatibility wrapper exposing the plugin registry API explicitly."""
from __future__ import annotations

from core.plugin_manager import PluginManager


class PluginRegistry(PluginManager):
    """Thin alias around PluginManager for clearer imports."""

