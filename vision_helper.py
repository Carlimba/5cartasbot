"""
Módulo de visión / extracción de cartas desde capturas de pantalla.

Soporta tres modos:
  1. Visión OpenAI (GPT-4o / GPT-4.1) vía API
  2. Visión Anthropic Claude (claude-sonnet / opus) vía API
  3. Extracción heurística por texto / JSON pegado (fallback sin API)

Configura las claves con variables de entorno:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Sequence

from card_evaluator import CardError, normalize_card, parse_cards

VISION_SYSTEM_PROMPT = """Eres un extractor de cartas de Video Póker.
Analiza la imagen y localiza exactamente las 5 cartas de la mano del jugador
(de izquierda a derecha).

Devuelve ÚNICAMENTE un JSON válido con esta forma:
{"cards": ["AH", "KH", "10C", "2S", "5D"], "confidence": 0.0}

Reglas de notación:
- Rango: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2
- Palo: H=corazones/hearts, D=diamantes/diamonds, C=tréboles/clubs, S=picas/spades
- Orden: izquierda → derecha tal como aparecen en pantalla
- Si no puedes ver 5 cartas con claridad, usa confidence baja y tu mejor estimación
- No inventes texto fuera del JSON
"""


def _encode_image_bytes(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


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
    }.get(ext, "image/png")


def _extract_json_object(text: str) -> dict:
    """Extrae el primer objeto JSON de una respuesta de modelo."""
    text = text.strip()
    # Bloque ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))
    # Objeto crudo
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise CardError(f"No se pudo parsear JSON de la respuesta del modelo:\n{text[:500]}")


def cards_from_model_payload(payload: dict) -> list[str]:
    """Valida y normaliza el JSON devuelto por el modelo de visión."""
    if "cards" not in payload:
        raise CardError("La respuesta del modelo no incluye la clave 'cards'.")
    cards = payload["cards"]
    if not isinstance(cards, list):
        raise CardError("'cards' debe ser una lista.")
    return parse_cards([normalize_card(str(c)) for c in cards])


def parse_cards_from_text(text: str) -> list[str]:
    """
    Parsea cartas desde texto libre o JSON.

    Ejemplos aceptados:
      AH KH 10C 2S 5D
      ["AH","KH","10C","2S","5D"]
      A♥ K♥ 10♣ 2♠ 5♦
    """
    text = text.strip()
    if not text:
        raise CardError("Texto vacío: no hay cartas que parsear.")

    # Intentar JSON
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return cards_from_model_payload(data)
            if isinstance(data, list):
                return parse_cards([str(c) for c in data])
        except json.JSONDecodeError:
            pass

    # Tokens separados
    tokens = re.findall(
        r"(?:10|[2-9TJQKA])\s*[HDCS♥♦♣♠]|10[HDCS]|[2-9TJQKA][HDCS]",
        text.upper().replace("♥", "H").replace("♦", "D").replace("♣", "C").replace("♠", "S"),
    )
    # Limpiar espacios internos tipo "10 H"
    cleaned = [t.replace(" ", "") for t in tokens]
    if len(cleaned) >= 5:
        return parse_cards(cleaned[:5])

    # Fallback: split simple
    parts = re.split(r"[\s,;|/]+", text.upper())
    parts = [p for p in parts if p]
    if len(parts) == 5:
        return parse_cards(parts)

    raise CardError(
        "No se pudieron detectar 5 cartas en el texto. "
        "Usa notación como: AH KH 10C 2S 5D"
    )


def _resolve_api_key(explicit: str | None, env_name: str, secret_name: str) -> str | None:
    """Prioridad: argumento UI → variable de entorno → st.secrets (Streamlit Cloud)."""
    if explicit:
        return explicit
    env_val = os.getenv(env_name)
    if env_val:
        return env_val
    try:
        import streamlit as st

        secrets = st.secrets
        if secret_name in secrets:
            return str(secrets[secret_name])
        # También aceptar bloque [api_keys]
        if "api_keys" in secrets and secret_name in secrets["api_keys"]:
            return str(secrets["api_keys"][secret_name])
    except Exception:
        pass
    return None


def analyze_image_openai(
    image_bytes: bytes,
    filename: str | None = None,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> dict:
    """
    Llama a la API de OpenAI Vision y devuelve:
      {"cards": [...], "confidence": float, "raw": str, "provider": "openai"}
    """
    key = _resolve_api_key(api_key, "OPENAI_API_KEY", "OPENAI_API_KEY")
    if not key:
        raise CardError(
            "Falta OPENAI_API_KEY. Configúrala en Secrets de Streamlit Cloud, "
            "en el entorno, o pégala en la UI."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise CardError(
            "El paquete 'openai' no está instalado. Ejecuta: pip install openai"
        ) from exc

    mime = _guess_mime(filename)
    data_url = _encode_image_bytes(image_bytes, mime)
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extrae las 5 cartas de esta captura de Video Póker.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    raw = response.choices[0].message.content or ""
    payload = _extract_json_object(raw)
    cards = cards_from_model_payload(payload)
    return {
        "cards": cards,
        "confidence": float(payload.get("confidence", 0.5)),
        "raw": raw,
        "provider": "openai",
        "model": model,
    }


def analyze_image_claude(
    image_bytes: bytes,
    filename: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> dict:
    """
    Llama a la API de Anthropic Claude Vision y devuelve el mismo esquema que OpenAI.
    """
    key = _resolve_api_key(api_key, "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    if not key:
        raise CardError(
            "Falta ANTHROPIC_API_KEY. Configúrala en Secrets de Streamlit Cloud, "
            "en el entorno, o pégala en la UI."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise CardError(
            "El paquete 'anthropic' no está instalado. Ejecuta: pip install anthropic"
        ) from exc

    mime = _guess_mime(filename)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        system=VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extrae las 5 cartas de esta captura de Video Póker.",
                    },
                ],
            }
        ],
    )
    raw = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    payload = _extract_json_object(raw)
    cards = cards_from_model_payload(payload)
    return {
        "cards": cards,
        "confidence": float(payload.get("confidence", 0.5)),
        "raw": raw,
        "provider": "anthropic",
        "model": model,
    }


def analyze_image(
    image_bytes: bytes,
    filename: str | None = None,
    provider: str = "openai",
    api_key: str | None = None,
) -> dict:
    """
    Punto de entrada unificado para visión.

    provider: 'openai' | 'claude' | 'anthropic'
    """
    provider = provider.lower().strip()
    if provider in ("openai", "gpt"):
        return analyze_image_openai(image_bytes, filename=filename, api_key=api_key)
    if provider in ("claude", "anthropic"):
        return analyze_image_claude(image_bytes, filename=filename, api_key=api_key)
    raise CardError(f"Proveedor de visión desconocido: {provider!r}")


def validate_five_cards(cards: Sequence[str]) -> list[str]:
    """Re-valida una lista de 5 cartas (útil tras edición manual post-visión)."""
    return parse_cards(list(cards))
