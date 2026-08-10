"""
Video Póker Jacks or Better 9/6 — Analizador de decisión óptima (EV exacto).

Ejecutar:
    streamlit run app.py

Pensado para pausar partidas de YouTube, cargar la mano (manual o por imagen)
y obtener qué cartas HOLD maximizan el valor esperado.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from card_evaluator import (
    HAND_CATEGORY_ES,
    PAYTABLE_9_6,
    RANKS,
    SUIT_NAMES,
    SUIT_SYMBOLS,
    SUITS,
    CardError,
    evaluate_hand_detail,
    format_card_pretty,
    format_hand_pretty,
    parse_cards,
)
from math_engine import analyze_hand_fast, results_to_rows
from vision_helper import analyze_image, parse_cards_from_text

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Jacks or Better 9/6 — EV Óptimo",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #5a6a5a;
        margin-bottom: 1.2rem;
    }
    .card-chip {
        display: inline-block;
        min-width: 4.2rem;
        padding: 0.85rem 0.6rem;
        margin: 0.25rem;
        border-radius: 0.55rem;
        background: #f7f4ec;
        border: 2px solid #2c3e2d;
        text-align: center;
        font-size: 1.45rem;
        font-weight: 700;
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        box-shadow: 0 2px 0 #1a261b;
    }
    .card-chip.red { color: #c0392b; }
    .card-chip.black { color: #1a1a1a; }
    .card-chip.hold {
        border-color: #d4a017;
        background: #fff6d6;
        box-shadow: 0 0 0 3px rgba(212, 160, 23, 0.35);
    }
    .optimal-box {
        background: linear-gradient(135deg, #1e3d2f 0%, #2f5d45 100%);
        color: #f5f5f0;
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0 1.25rem 0;
    }
    .optimal-box h3 {
        margin: 0 0 0.4rem 0;
        color: #f0d878;
        font-size: 1.15rem;
    }
    .optimal-ev {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .felt {
        background: radial-gradient(ellipse at center, #2f6b4f 0%, #1d4534 70%);
        padding: 1.25rem;
        border-radius: 0.85rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    div[data-testid="stDataFrame"] th {
        background-color: #1e3d2f !important;
        color: #f5f5f0 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "hand": [None, None, None, None, None],  # cada slot: "AH" | None
        "analysis_rows": None,
        "analysis_results": None,
        "last_elapsed": None,
        "vision_confidence": None,
        "vision_raw": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _card_html(card: str | None, hold: bool = False) -> str:
    if not card:
        return '<span class="card-chip black">?</span>'
    pretty = format_card_pretty(card)
    color = "red" if card[-1] in ("H", "D") else "black"
    hold_cls = " hold" if hold else ""
    return f'<span class="card-chip {color}{hold_cls}">{pretty}</span>'


def _hand_is_complete(hand: list) -> bool:
    return all(c is not None for c in hand) and len(hand) == 5


# ---------------------------------------------------------------------------
# Sidebar: tabla de pagos
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Tabla 9/6")
    st.caption("Jacks or Better — multiplicadores por unidad apostada")
    pay_df = pd.DataFrame(
        [
            {"Mano": HAND_CATEGORY_ES[k], "Pago": v}
            for k, v in PAYTABLE_9_6.items()
            if k != "nothing"
        ]
    )
    st.dataframe(pay_df, hide_index=True, use_container_width=True)
    st.markdown("---")
    st.markdown(
        "**Cómo usar con YouTube**\n\n"
        "1. Pausa el vídeo en la mano inicial (5 cartas).\n"
        "2. Carga la mano (manual o captura).\n"
        "3. Pulsa **Calcular EV óptimo**.\n"
        "4. HOLD las cartas marcadas en ★ ÓPTIMO."
    )
    st.markdown("---")
    st.caption(
        "El motor enumera las 32 retenciones y todas las combinaciones "
        "del mazo restante (EV exacto, ~2.6M evaluaciones)."
    )


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">♠ Jacks or Better 9/6 — Decisión Óptima</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Motor de EV exacto · 32 holds · baraja de 52 cartas</div>',
    unsafe_allow_html=True,
)

tab_manual, tab_vision, tab_text = st.tabs(
    ["Opción A · Selección manual", "Opción B · Análisis por imagen", "Pegar texto / JSON"]
)


# ---------------------------------------------------------------------------
# Opción A: selector manual
# ---------------------------------------------------------------------------

with tab_manual:
    st.subheader("Carga rápida de 5 cartas")
    st.caption("Elige rango y palo para cada posición (izquierda → derecha).")

    cols = st.columns(5)
    rank_options = list(RANKS)
    suit_options = [f"{SUIT_SYMBOLS[s]} {SUIT_NAMES[s]} ({s})" for s in SUITS]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Carta {i + 1}**")
            current = st.session_state.hand[i]
            default_rank_idx = 12  # As
            default_suit_idx = 0
            if current:
                default_rank_idx = rank_options.index(current[:-1])
                default_suit_idx = SUITS.index(current[-1])

            r = st.selectbox(
                "Rango",
                rank_options,
                index=default_rank_idx,
                key=f"rank_{i}",
            )
            s_label = st.selectbox(
                "Palo",
                suit_options,
                index=default_suit_idx,
                key=f"suit_{i}",
            )
            suit_code = SUITS[suit_options.index(s_label)]
            # No asignar hasta confirmar para evitar manos a medias confusas;
            # usamos botones de aplicar.
            st.session_state[f"_draft_{i}"] = f"{r}{suit_code}"

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("Aplicar 5 cartas", type="primary", use_container_width=True):
            draft = [st.session_state[f"_draft_{i}"] for i in range(5)]
            try:
                st.session_state.hand = parse_cards(draft)
                st.session_state.analysis_rows = None
                st.session_state.analysis_results = None
                st.success(f"Mano cargada: {format_hand_pretty(st.session_state.hand)}")
            except CardError as exc:
                st.error(str(exc))
    with b2:
        if st.button("Limpiar mano", use_container_width=True):
            st.session_state.hand = [None, None, None, None, None]
            st.session_state.analysis_rows = None
            st.session_state.analysis_results = None
            st.rerun()


# ---------------------------------------------------------------------------
# Opción B: visión
# ---------------------------------------------------------------------------

with tab_vision:
    st.subheader("Captura / screenshot de la mano")
    st.caption(
        "Sube una imagen del vídeo pausado. Opcionalmente usa OpenAI o Claude Vision "
        "para detectar las 5 cartas automáticamente."
    )

    uploaded = st.file_uploader(
        "Imagen (PNG, JPG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
    )

    provider = st.radio(
        "Motor de visión",
        options=["openai", "claude", "manual_confirm"],
        format_func=lambda x: {
            "openai": "OpenAI Vision (GPT-4o)",
            "claude": "Anthropic Claude Vision",
            "manual_confirm": "Solo subir imagen (confirmar a mano después)",
        }[x],
        horizontal=True,
    )

    api_key_input = ""
    if provider != "manual_confirm":
        api_key_input = st.text_input(
            "API Key (opcional si ya está en el entorno)",
            type="password",
            help="OPENAI_API_KEY o ANTHROPIC_API_KEY según el proveedor.",
        )

    if uploaded is not None:
        st.image(uploaded, caption="Captura cargada", use_container_width=True)

    if st.button("Detectar cartas en la imagen", type="primary"):
        if uploaded is None:
            st.warning("Sube una imagen primero.")
        elif provider == "manual_confirm":
            st.info(
                "Imagen lista. Usa la pestaña de selección manual o pega las cartas "
                "que veas en 'Pegar texto / JSON'."
            )
        else:
            try:
                image_bytes = uploaded.getvalue()
                with st.spinner("Consultando API de visión..."):
                    result = analyze_image(
                        image_bytes,
                        filename=uploaded.name,
                        provider=provider,
                        api_key=api_key_input or None,
                    )
                st.session_state.hand = result["cards"]
                st.session_state.vision_confidence = result.get("confidence")
                st.session_state.vision_raw = result.get("raw")
                st.session_state.analysis_rows = None
                st.session_state.analysis_results = None
                conf = result.get("confidence")
                conf_txt = f" (confianza ≈ {conf:.0%})" if isinstance(conf, float) else ""
                st.success(
                    f"Detectadas: {format_hand_pretty(result['cards'])}{conf_txt}. "
                    "Revisa y corrige en el selector manual si hace falta."
                )
            except CardError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 — mostrar error de API al usuario
                st.error(f"Error al llamar a la API de visión: {exc}")


# ---------------------------------------------------------------------------
# Pegar texto
# ---------------------------------------------------------------------------

with tab_text:
    st.subheader("Pegar mano en texto")
    sample = st.text_area(
        "Ejemplos:  AH KH 10C 2S 5D   ·   A♥ K♥ 10♣ 2♠ 5♦   ·   JSON",
        value="AH KH QC JD 10S",
        height=100,
    )
    if st.button("Cargar desde texto"):
        try:
            parsed = parse_cards_from_text(sample)
            st.session_state.hand = parsed
            st.session_state.analysis_rows = None
            st.session_state.analysis_results = None
            st.success(f"Mano cargada: {format_hand_pretty(parsed)}")
        except CardError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------------
# Vista de mano actual + cálculo
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Mano actual")

hand = st.session_state.hand
if _hand_is_complete(hand):
    chips = "".join(_card_html(c) for c in hand)
    st.markdown(f'<div class="felt">{chips}</div>', unsafe_allow_html=True)
    try:
        detail = evaluate_hand_detail(hand)
        st.caption(
            f"Si te quedas pat (HOLD las 5): **{detail['name_es']}** → pago {detail['payout']}x"
        )
    except CardError as exc:
        st.error(str(exc))
else:
    st.info("Carga las 5 cartas con alguna de las opciones de arriba.")

calc_col, info_col = st.columns([1, 2])
with calc_col:
    run = st.button(
        "Calcular EV óptimo (32 holds)",
        type="primary",
        disabled=not _hand_is_complete(hand),
        use_container_width=True,
    )
with info_col:
    st.caption(
        "Cálculo exacto sobre el mazo restante. Suele tardar unos segundos "
        "(hasta ~15–40 s según CPU)."
    )

if run and _hand_is_complete(hand):
    progress = st.progress(0.0, text="Enumerando combinaciones…")
    status = st.empty()

    def _cb(done: int, total: int) -> None:
        progress.progress(done / total, text=f"Hold {done}/{total}…")
        status.caption(f"Evaluando máscara {done - 1:05b} ({done}/{total})")

    t0 = time.perf_counter()
    try:
        results = analyze_hand_fast(hand, progress_callback=_cb)
        elapsed = time.perf_counter() - t0
        st.session_state.analysis_results = results
        st.session_state.analysis_rows = results_to_rows(results)
        st.session_state.last_elapsed = elapsed
        progress.progress(1.0, text="Listo")
        status.empty()
    except CardError as exc:
        st.error(str(exc))
        progress.empty()
        status.empty()


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

results = st.session_state.analysis_results
rows = st.session_state.analysis_rows

if results and rows:
    optimal = next(r for r in results if r.is_optimal)
    hold_positions = optimal.positions_1based
    hold_set = {i for i in range(5) if (optimal.hold_mask >> i) & 1}

    st.markdown(
        f"""
        <div class="optimal-box">
            <h3>★ OPCIÓN ÓPTIMA</h3>
            <div>{optimal.hold_description}</div>
            <div style="margin-top:0.5rem;">
                Posiciones a pulsar HOLD (1–5):
                <strong>{', '.join(str(p) for p in hold_positions) if hold_positions else 'ninguna (descarta todo)'}</strong>
            </div>
            <div class="optimal-ev">EV = {optimal.ev:.6f}</div>
            <div style="opacity:0.85; margin-top:0.35rem;">
                Draws evaluados en esta opción: {optimal.num_draws:,}
                {" · tiempo total: " + f"{st.session_state.last_elapsed:.2f}s" if st.session_state.last_elapsed else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cartas con HOLD resaltado
    chips_hold = "".join(_card_html(hand[i], hold=(i in hold_set)) for i in range(5))
    st.markdown(
        f'<div class="felt">{chips_hold}<div style="color:#f0d878;margin-top:0.6rem;">'
        f"Amarillo = HOLD</div></div>",
        unsafe_allow_html=True,
    )

    # Empates óptimos
    optima = [r for r in results if r.is_optimal]
    if len(optima) > 1:
        st.warning(
            f"Hay {len(optima)} estrategias empatadas en el EV máximo. "
            "Se muestra la primera (más cartas retenidas en caso de empate)."
        )

    st.subheader("Ranking completo de las 32 combinaciones")
    df = pd.DataFrame(rows)
    # Resaltar óptimos
    def _highlight(row: pd.Series) -> list[str]:
        if row.get("Óptimo"):
            return ["background-color: #fff3bf; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1).format({"EV": "{:.6f}"}),
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    # Top 5 compacto
    with st.expander("Top 5 alternativas"):
        top5 = df.head(5)[["Rank", "Óptimo", "HOLD (cartas)", "Posiciones HOLD", "EV"]]
        st.table(top5)

elif _hand_is_complete(hand):
    st.info("Pulsa **Calcular EV óptimo** para obtener la decisión perfecta.")
