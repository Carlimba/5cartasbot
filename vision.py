"""
Reconocimiento de cartas en capturas de Video Póker vía OpenAI Vision (gpt-4o).

Lee la API key desde (en este orden):
  1. Argumento `api_key`
  2. Variable de entorno OPENAI_API_KEY / archivo .env
  3. Streamlit secrets (`st.secrets["OPENAI_API_KEY"]`)
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Sequence

from card_evaluator import CardError, normalize_card, parse_cards

# Carga .env si existe (local). En Streamlit Cloud se usan secrets.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SYSTEM_PROMPT = (
    "Eres un experto reconociendo cartas de póker en capturas de pantalla de casinos. "
    "Analiza la imagen y devuelve ÚNICAMENTE un array en formato JSON con las 5 cartas "
    "que ves de izquierda a derecha. Usa el formato: Valor (2-10, J, Q, K, A) y Palo "
    "(H, D, C, S para Corazones, Diamantes, Tréboles, Picas). "
    "Ejemplo: [\"AH\", \"10C\", \"2S\", \"JD\", \"QC\"]."
)

DEFAULT_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")


def get_openai_api_key(explicit: str | None = None) -> str | None:
    """Resuelve la API key: UI → entorno/.env → st.secrets."""
    if explicit and explicit.strip():
        return explicit.strip()

    env_val = os.getenv("OPENAI_API_KEY")
    if env_val:
        return env_val.strip()

    try:
        import streamlit as st

        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return str(st.secrets["openai"]["api_key"]).strip()
        if "api_keys" in st.secrets and "OPENAI_API_KEY" in st.secrets["api_keys"]:
            return str(st.secrets["api_keys"]["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return None


def _guess_mime(filename: str | None) -> str:
    if not filename:
        return "image/png"
    ext = Path(filename).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def image_to_data_url(image_bytes: bytes, filename: str | None = None) -> str:
    """Convierte bytes de imagen a data URL base64 para la API de OpenAI."""
    mime = _guess_mime(filename)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_cards_payload(text: str) -> list:
    """Extrae un array JSON de cartas desde la respuesta del modelo."""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Array directo
    if text.startswith("["):
        return json.loads(text)

    # Objeto {"cards": [...]}
    if text.startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, dict) and "cards" in obj:
            return obj["cards"]
        raise CardError("JSON recibido sin clave 'cards'.")

    # Buscar primer array en el texto
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])

    raise CardError(f"No se pudo parsear la respuesta de visión:\n{text[:400]}")


def detect_cards_from_image(
    image_bytes: bytes,
    filename: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Envía la captura a OpenAI Vision y devuelve exactamente 5 cartas normalizadas.

    Raises:
        CardError: si falta la key, la API falla o las cartas son inválidas/duplicadas.
    """
    key = get_openai_api_key(api_key)
    if not key:
        raise CardError(
            "Falta OPENAI_API_KEY. Configúrala en `.env`, en Streamlit Secrets "
            "o pégala en la barra lateral."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CardError("Instala el paquete openai: pip install openai") from exc

    if not image_bytes:
        raise CardError("La imagen está vacía.")

    data_url = image_to_data_url(image_bytes, filename)
    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Estas son las 5 cartas de la mano de Video Póker "
                            "(izquierda a derecha). Devuelve solo el JSON array."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise CardError("OpenAI devolvió una respuesta vacía.")

    try:
        cards_raw = _extract_cards_payload(raw)
    except json.JSONDecodeError as exc:
        raise CardError(f"JSON inválido de OpenAI: {raw[:400]}") from exc

    if not isinstance(cards_raw, list) or len(cards_raw) != 5:
        raise CardError(
            f"Se esperaban 5 cartas; OpenAI devolvió: {cards_raw!r}"
        )

    return parse_cards([normalize_card(str(c)) for c in cards_raw])


def validate_hand(cards: Sequence[str]) -> list[str]:
    """Revalida una mano de 5 cartas (tras corrección manual en la UI)."""
    return parse_cards(list(cards))


# Alias usados por la UI
analyze_screenshot = detect_cards_from_image
