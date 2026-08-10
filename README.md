# Jacks or Better 9/6 — Analizador de EV óptimo

Herramienta en Python + Streamlit que calcula la **decisión matemáticamente perfecta**
en Video Póker Jacks or Better (tabla de pagos 9/6). Ideal para pausar partidas de
YouTube, cargar la mano y ver qué cartas HOLD maximizan el valor esperado (EV).

## App online

**Link permanente:** [https://5cartasbot.streamlit.app/](https://5cartasbot.streamlit.app/)

Cada `git push` a `master` actualiza la app automáticamente.

### Secrets opcionales (visión por imagen)

En Streamlit Cloud → **Settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

### Instalación local

```bash
cd 5cartasbot
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar la app

```bash
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

## Uso rápido

1. **Opción A — Manual:** elige rango y palo de las 5 cartas → *Aplicar 5 cartas*.
2. **Opción B — Imagen:** sube un screenshot del vídeo pausado.
   - Con `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` puedes detectar las cartas automáticamente.
   - Sin API, usa la imagen como referencia y carga la mano a mano.
3. Pulsa **Calcular EV óptimo (32 holds)**.
4. La UI resalta la ★ OPCIÓN ÓPTIMA y lista las 32 combinaciones ordenadas por EV.

### Variables de entorno (visión opcional)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Estructura del proyecto

| Archivo | Rol |
|---------|-----|
| `card_evaluator.py` | Parsing de cartas, clasificación de manos, tabla 9/6 |
| `math_engine.py` | EV exacto de las 32 máscaras de hold sobre el mazo restante |
| `vision_helper.py` | OpenAI / Claude Vision + parser de texto/JSON |
| `app.py` | Interfaz Streamlit |
| `requirements.txt` | Dependencias |

> `itertools` y `math` forman parte de la biblioteca estándar de Python; no van en `requirements.txt`.

## Motor matemático

Para una mano de 5 cartas el motor:

1. Genera las **32** retenciones posibles (`2^5`).
2. Para cada retención con *k* cartas, reparte las `5−k` restantes desde las **47** cartas del mazo.
3. Calcula  
   `EV = Σ payout(mano_final) / C(47, 5−k)`.
4. Ordena de mayor a menor EV y marca la(s) óptima(s).

Tabla de pagos usada (multiplicadores por unidad; royal a máximo crédito):

| Mano | Pago |
|------|------|
| Escalera Real | 800 |
| Escalera de Color | 50 |
| Póker | 25 |
| Full House | 9 |
| Color | 6 |
| Escalera | 4 |
| Trío | 3 |
| Doble Pareja | 2 |
| Pareja de Jotas o Mejor | 1 |
| Resto | 0 |

El cálculo es **exacto** (enumeración completa). En total se evalúan
`C(52,5) = 2 598 960` manos a lo largo de las 32 opciones; en CPU moderna suele
tardar unos segundos o unas decenas de segundos.

## Uso programático

```python
from math_engine import analyze_hand_fast, optimal_play

hand = ["AH", "KH", "10C", "2S", "5D"]
best = optimal_play(hand)  # usa analyze_hand por debajo
# o el camino rápido recomendado:
ranked = analyze_hand_fast(hand)
print(ranked[0].hold_description, ranked[0].ev)
```

## Notas

- Notación de cartas: `AH`, `KH`, `10C` (también `TC`), `2S`, `5D`.
- Se rechazan manos con cartas duplicadas o formato inválido.
- La royal se valora a **800** (juego a 5 monedas), estándar en estrategia de Video Póker.
