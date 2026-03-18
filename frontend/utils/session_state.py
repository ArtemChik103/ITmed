"""Local Streamlit session-state helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

HISTORY_KEY = "analysis_history"
LAST_RESULT_KEY = "last_result"
LAST_FILE_KEY = "last_file_signature"


def initialize_session_state() -> None:
    st.session_state.setdefault(HISTORY_KEY, [])
    st.session_state.setdefault(LAST_RESULT_KEY, None)
    st.session_state.setdefault(LAST_FILE_KEY, None)


def set_last_result(result: dict[str, Any] | None) -> None:
    st.session_state[LAST_RESULT_KEY] = result


def get_last_result() -> dict[str, Any] | None:
    return st.session_state.get(LAST_RESULT_KEY)


def update_file_signature(signature: str | None) -> None:
    st.session_state[LAST_FILE_KEY] = signature


def get_file_signature() -> str | None:
    return st.session_state.get(LAST_FILE_KEY)


def add_history_entry(entry: dict[str, Any], *, limit: int = 10) -> None:
    history = list(st.session_state.get(HISTORY_KEY, []))
    history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **entry,
        },
    )
    st.session_state[HISTORY_KEY] = history[:limit]


def get_history() -> list[dict[str, Any]]:
    return list(st.session_state.get(HISTORY_KEY, []))


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []
