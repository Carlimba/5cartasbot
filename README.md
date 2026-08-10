# Jacks or Better 9/6 — Asistente de EV óptimo

Screenshot → **GPT-4o Vision** → corrección manual → **HOLD** de máximo EV (exacto).

## App online

**https://5cartasbot.streamlit.app/**

## Estructura

| Archivo | Rol |
|---------|-----|
| `app.py` | UI Streamlit (captura, corrección, resultados) |
| `vision.py` | OpenAI Vision → 5 cartas JSON |
| `poker_math.py` | API del motor de EV (32 holds) |
| `card_evaluator.py` | Clasificación de manos + tabla 9/6 |
| `math_engine.py` | Enumeración exacta de draws |

## Secrets / API Key

**Streamlit Cloud → Settings → Secrets:**

```toml
OPENAI_API_KEY = "sk-..."
```

**Local** — copia `.env.example` a `.env`:

```bash
OPENAI_API_KEY=sk-...
```

## Local

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Flujo de uso

1. Pausa el vídeo de Video Póker en la mano de 5 cartas.
2. Sube o pega el screenshot.
3. **Detectar 5 cartas con GPT-4o**.
4. Corrige cualquier carta en los selectboxes si la IA falló.
5. **Calcular HOLD óptimo** → retén las cartas en dorado.
