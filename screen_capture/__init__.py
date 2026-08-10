"""
Componente Streamlit: captura un frame de una ventana/pestaña del sistema
(vía Screen Capture API del navegador) y lo devuelve como data-URL PNG/JPEG.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")

_screen_capture = components.declare_component(
    "screen_capture",
    path=_FRONTEND,
)


def screen_capture_button(
    label: str = "📸 Capturar pantalla del juego",
    key: str | None = None,
) -> dict[str, Any] | None:
    """
    Muestra un botón. Al usarlo, el navegador pide elegir pantalla/ventana/pestaña.
    Devuelve dict {"data_url": "data:image/jpeg;base64,...", "ts": int} o None.
    """
    value = _screen_capture(label=label, key=key, default=None)
    if isinstance(value, dict) and value.get("data_url"):
        return value
    return None


def data_url_to_bytes(data_url: str) -> bytes:
    """Convierte un data-URL a bytes de imagen."""
    if "," not in data_url:
        raise ValueError("data_url inválido")
    header, b64 = data_url.split(",", 1)
    return base64.standard_b64decode(b64)
