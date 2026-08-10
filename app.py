"""
Asistente en tiempo real — Video Póker Jacks or Better 9/6.

Flujo:
  1. Sube o pega un screenshot de la mano
  2. OpenAI Vision detecta las 5 cartas
  3. Corrige con selectboxes si hace falta
  4. El motor calcula el HOLD de EV máximo (exacto)

Ejecutar:  streamlit run app.py
Online:    https://5cartasbot.streamlit.app/
"""

from __future__ import annotations

import io
import time

import pandas as pd
import streamlit as st

from poker_math import (
    HAND_CATEGORY_ES,
    PAYTABLE_9_6,
    CardError,
    all_card_codes,
    analyze_hand,
    evaluate_hand_detail,
    format_card_pretty,
    format_hand_pretty,
    parse_cards,
    top_n_strategies,
)
from vision import detect_cards_from_image, get_openai_api_key

st.set_page_config(
    page_title="5 Cartas Bot · Jacks or Better 9/6",
    page_icon="♠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-title { font-family: "Trebuchet MS", "Segoe UI", sans-serif; font-size: 2rem;
              font-weight: 700; margin-bottom: 0.15rem; }
.sub-title { color: #5a6a5a; margin-bottom: 1rem; }
.card-chip { display: inline-block; min-width: 4.4rem; padding: 0.9rem 0.55rem;
             margin: 0.2rem; border-radius: 0.55rem; background: #f7f4ec;
             border: 2px solid #2c3e2d; text-align: center; font-size: 1.5rem;
             font-weight: 700; font-family: "Trebuchet MS", "Segoe UI", sans-serif;
             box-shadow: 0 2px 0 #1a261b; }
.card-chip.red { color: #c0392b; }
.card-chip.black { color: #1a1a1a; }
.card-chip.hold { border-color: #d4a017; background: #fff6d6;
                  box-shadow: 0 0 0 3px rgba(212,160,23,.4); }
.card-chip.dump { opacity: 0.42; filter: grayscale(0.35); }
.felt { background: radial-gradient(ellipse at center, #2f6b4f 0%, #1d4534 70%);
        padding: 1.1rem; border-radius: 0.85rem; text-align: center; margin: 0.5rem 0 1rem; }
.legend { color: #f0d878; margin-top: 0.55rem; font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "hand": None,
        "hand_ready": False,
        "image_bytes": None,
        "image_name": None,
        "analysis_results": None,
        "last_elapsed": None,
        "card_options": all_card_codes(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _chip(card: str, mode: str = "neutral") -> str:
    pretty = format_card_pretty(card)
    color = "red" if card[-1] in ("H", "D") else "black"
    extra = {"hold": " hold", "dump": " dump"}.get(mode, "")
    return f'<span class="card-chip {color}{extra}">{pretty}</span>'


def _label(card: str) -> str:
    return f"{format_card_pretty(card)}  ({card})"


def _set_hand(cards: list[str]) -> None:
    st.session_state.hand = cards
    st.session_state.hand_ready = True
    st.session_state.analysis_results = None
    st.session_state.last_elapsed = None
    for i, c in enumerate(cards):
        st.session_state[f"card_sel_{i}"] = c


_init_state()
OPTIONS: list[str] = st.session_state.card_options

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("♠ 5 Cartas Bot")
    st.caption("Jacks or Better 9/6 · EV exacto")

    if get_openai_api_key():
        st.success("OpenAI API key detectada")
    else:
        st.warning("Falta OPENAI_API_KEY para Vision")

    api_key_ui = st.text_input(
        "OpenAI API Key (opcional)",
        type="password",
        help="Alternativa a `.env` o Streamlit Secrets.",
        placeholder="sk-...",
    )

    st.markdown("---")
    st.subheader("Tabla 9/6")
    st.dataframe(
        pd.DataFrame(
            [
                {"Mano": HAND_CATEGORY_ES[k], "Pago": v}
                for k, v in PAYTABLE_9_6.items()
                if k != "nothing"
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown(
        "**Flujo YouTube**\n\n"
        "1. Pausa en las 5 cartas\n"
        "2. Captura / pega screenshot\n"
        "3. Detectar → corregir\n"
        "4. Calcular → HOLD dorado"
    )

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">♠ Asistente Jacks or Better 9/6</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Screenshot → GPT-4o Vision → corrección → HOLD óptimo (EV exacto)</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1) Captura
# ---------------------------------------------------------------------------

st.subheader("1 · Captura de pantalla")
st.caption("Sube un PNG/JPG o pega desde el portapapeles (botón 📋).")

c_up, c_paste = st.columns([2, 1])
with c_up:
    uploaded = st.file_uploader(
        "Screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
with c_paste:
    try:
        from streamlit_paste_button import paste_image_button

        paste_result = paste_image_button(
            label="📋 Pegar captura",
            background_color="#2f5d45",
            hover_background_color="#3d7a5a",
            errors="ignore",
        )
        if paste_result is not None and getattr(paste_result, "image_data", None) is not None:
            buf = io.BytesIO()
            paste_result.image_data.convert("RGB").save(buf, format="PNG")
            st.session_state.image_bytes = buf.getvalue()
            st.session_state.image_name = "clipboard.png"
            st.caption("✓ Pegada del portapapeles")
    except Exception:
        st.caption("Uploader = archivo · o Ctrl+V en el diálogo del sistema")

if uploaded is not None:
    st.session_state.image_bytes = uploaded.getvalue()
    st.session_state.image_name = uploaded.name

image_bytes = st.session_state.image_bytes
if image_bytes:
    st.image(image_bytes, caption=st.session_state.image_name or "Captura", use_container_width=True)

if st.button("Detectar 5 cartas con GPT-4o", type="primary", disabled=not bool(image_bytes)):
    try:
        with st.spinner("OpenAI Vision analizando la captura…"):
            cards = detect_cards_from_image(
                image_bytes,
                filename=st.session_state.image_name,
                api_key=api_key_ui or None,
            )
        _set_hand(cards)
        st.success(f"Detectadas: **{format_hand_pretty(cards)}**")
        st.rerun()
    except CardError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error de la API de visión: {exc}")

with st.expander("¿Sin captura? Cargar mano en texto"):
    manual_txt = st.text_input("5 cartas", placeholder="AH KH 10C 2S 5D")
    if st.button("Usar esta mano"):
        try:
            cards = parse_cards(manual_txt.replace(",", " ").split())
            _set_hand(cards)
            st.rerun()
        except CardError as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# 2) Corrección
# ---------------------------------------------------------------------------

st.subheader("2 · Cartas detectadas (corrige si la IA falló)")

if not st.session_state.hand_ready:
    st.info("Todavía no hay mano. Detecta un screenshot o cárgala manualmente.")
else:
    cols = st.columns(5)
    corrected: list[str] = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Carta {i + 1}**")
            if f"card_sel_{i}" not in st.session_state:
                st.session_state[f"card_sel_{i}"] = st.session_state.hand[i]
            choice = st.selectbox(
                f"Carta posición {i + 1}",
                options=OPTIONS,
                format_func=_label,
                key=f"card_sel_{i}",
                label_visibility="collapsed",
            )
            corrected.append(choice)
            st.markdown(
                f'<div style="text-align:center">{_chip(choice)}</div>',
                unsafe_allow_html=True,
            )

    hand_ok = False
    hand: list[str] = []
    try:
        hand = parse_cards(corrected)
        hand_ok = True
        st.session_state.hand = hand
        detail = evaluate_hand_detail(hand)
        st.caption(
            f"Mano: {format_hand_pretty(hand)} · "
            f"pat → **{detail['name_es']}** ({detail['payout']}x)"
        )
    except CardError as exc:
        st.error(f"Mano inválida: {exc}")

    # -----------------------------------------------------------------------
    # 3) EV
    # -----------------------------------------------------------------------

    st.subheader("3 · Jugada matemáticamente perfecta")

    if st.button(
        "Calcular HOLD óptimo (EV exacto · 32 combinaciones)",
        type="primary",
        disabled=not hand_ok,
    ):
        progress = st.progress(0.0, text="Enumerando draws del mazo…")

        def _cb(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Evaluando hold {done}/{total}…")

        t0 = time.perf_counter()
        try:
            results = analyze_hand(hand, progress_callback=_cb)
            st.session_state.analysis_results = results
            st.session_state.last_elapsed = time.perf_counter() - t0
            progress.progress(1.0, text="Listo")
        except CardError as exc:
            st.error(str(exc))
            progress.empty()

    results = st.session_state.analysis_results
    if results and hand_ok:
        optimal = next(r for r in results if r.is_optimal)
        hold_set = {i for i in range(5) if (optimal.hold_mask >> i) & 1}
        hold_cards = [hand[i] for i in range(5) if i in hold_set]
        dump_cards = [hand[i] for i in range(5) if i not in hold_set]
        positions = optimal.positions_1based

        st.success(
            f"**★ JUGADA ÓPTIMA** — EV = **{optimal.ev:.6f}**\n\n"
            f"**HOLD:** {format_hand_pretty(hold_cards) if hold_cards else '— (descartar todas)'}\n\n"
            f"**DESCARTAR:** {format_hand_pretty(dump_cards) if dump_cards else '— (ninguna)'}\n\n"
            f"**Posiciones HOLD (1–5):** "
            f"{', '.join(map(str, positions)) if positions else 'ninguna'}"
        )

        chips = "".join(
            _chip(hand[i], "hold" if i in hold_set else "dump") for i in range(5)
        )
        st.markdown(
            f'<div class="felt">{chips}'
            f'<div class="legend">Dorado = HOLD &nbsp;|&nbsp; Atenué = descartar</div></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.last_elapsed is not None:
            st.caption(
                f"EV exacto · {optimal.num_draws:,} draws en la óptima · "
                f"{st.session_state.last_elapsed:.2f}s"
            )

        if sum(1 for r in results if r.is_optimal) > 1:
            st.warning("Hay varias estrategias empatadas en el EV máximo.")

        st.markdown("#### Top 5 combinaciones (por EV)")
        top5 = pd.DataFrame(top_n_strategies(results, 5))[
            ["Rank", "Óptimo", "HOLD (cartas)", "Posiciones HOLD", "EV"]
        ]
        st.dataframe(top5, hide_index=True, use_container_width=True)

        with st.expander("Ver las 32 combinaciones"):
            st.dataframe(
                pd.DataFrame(top_n_strategies(results, 32)),
                hide_index=True,
                use_container_width=True,
                height=420,
            )
    elif hand_ok:
        st.info("Pulsa **Calcular HOLD óptimo** para obtener la decisión perfecta.")
