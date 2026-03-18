"""Status widgets for API health and runtime transparency."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.utils.report_formatting import runtime_model_loaded, runtime_status_text


def render_api_status(api_status: dict[str, Any], plugins_payload: dict[str, Any]) -> None:
    if api_status.get("ok"):
        version = api_status.get("version") or "unknown"
        st.success(f"API доступен · version {version}")
    else:
        st.error(f"API недоступен: {api_status.get('message', 'unknown error')}")

    plugin_names = [plugin.get("name", "?") for plugin in plugins_payload.get("plugins", [])]
    if plugin_names:
        st.caption("Плагины: " + ", ".join(plugin_names))
    elif not plugins_payload.get("ok"):
        st.caption("Список плагинов недоступен.")


def render_runtime_status(result: dict[str, Any] | None) -> None:
    if result is None:
        st.info("Результат анализа еще не получен.")
        return

    if runtime_model_loaded(result):
        st.success(runtime_status_text(result))
    else:
        st.warning(runtime_status_text(result))
