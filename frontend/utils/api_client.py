"""Thin HTTP client used by the Streamlit frontend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when the frontend cannot complete an API request."""


@dataclass(slots=True)
class ApiClient:
    """Simple wrapper around the project FastAPI endpoints."""

    base_url: str
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "status": payload.get("status"), "version": payload.get("version")}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": str(exc)}

    def list_plugins(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/api/v1/plugins", timeout=5.0)
            response.raise_for_status()
            return {"ok": True, "plugins": response.json().get("plugins", [])}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "plugins": [], "message": str(exc)}

    def analyze(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        plugin_type: str,
        mode: str,
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}/api/v1/analyze",
                params={"plugin_type": plugin_type, "mode": mode},
                files={"file": (filename, file_bytes, "application/dicom")},
                timeout=self.timeout_seconds,
            )
        except httpx.ConnectError as exc:
            raise ApiClientError(
                "Не удалось подключиться к API. Убедитесь, что backend запущен."
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiClientError(f"Ошибка сетевого запроса: {exc}") from exc

        if response.is_error:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise ApiClientError(f"Ошибка API [{response.status_code}]: {detail}")

        return response.json()
