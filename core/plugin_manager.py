"""
core/plugin_manager.py — Plugin architecture stub for Phase 1.
Full implementation in Phase 2.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class PluginInterface(Protocol):
    """Интерфейс, который должны реализовывать все ML-плагины."""

    name: str
    version: str

    def analyze(
        self, image: Any, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Запускает анализ снимка и возвращает результат."""
        ...


class PluginManager:
    """
    Менеджер плагинов. Управляет регистрацией и вызовом ML-плагинов.
    Заглушка для Phase 1 — реальные плагины подключаются в Phase 2.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, PluginInterface] = {}
        logger.info("PluginManager инициализирован.")

    def register(self, plugin: PluginInterface) -> None:
        """Регистрирует плагин по имени."""
        if not isinstance(plugin, PluginInterface):
            raise TypeError(f"Объект {plugin!r} не реализует PluginInterface.")
        self._plugins[plugin.name] = plugin
        logger.info("Плагин '%s' v%s зарегистрирован.", plugin.name, plugin.version)

    def unregister(self, name: str) -> None:
        """Удаляет плагин из реестра."""
        if name in self._plugins:
            del self._plugins[name]
            logger.info("Плагин '%s' удалён.", name)

    def list_plugins(self) -> list[str]:
        """Возвращает список имён зарегистрированных плагинов."""
        return list(self._plugins.keys())

    def get_plugin(self, name: str) -> PluginInterface:
        """Возвращает плагин по имени или raises KeyError."""
        if name not in self._plugins:
            raise KeyError(f"Плагин '{name}' не найден. Доступные: {self.list_plugins()}")
        return self._plugins[name]

    def run(self, plugin_name: str, image: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Запускает плагин с указанными данными."""
        plugin = self.get_plugin(plugin_name)
        logger.info("Запуск плагина '%s'...", plugin_name)
        return plugin.analyze(image, metadata)
