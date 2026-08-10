"""
Asistente en tiempo real — Video Póker Jacks or Better 9/6.

Flujo rápido:
  Capturar pantalla del juego → Vision GPT-4o → EV óptimo (automático)

Online: https://5cartasbot.streamlit.app/
"""

from __future__ import annotations

import hashlib
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
from screen_capture import data_url_to_bytes, screen_capture_button
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
        "image_fp": None,
        "last_processed_fp": None,
        "analysis_results": None,
        "last_elapsed": None,
        "card_options": all_card_codes(),
        "auto_pipeline": True,
        "pending_auto": False,
        "last_capture_ts": None,
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


def _fingerprint(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _set_hand(cards: list[str]) -> None:
    st.session_state.hand = cards
    st.session_state.hand_ready = True
    st.session_state.analysis_results = None
    st.session_state.last_elapsed = None
    for i, c in enumerate(cards):
        st.session_state[f"card_sel_{i}"] = c


def _store_image(image_bytes: bytes, name: str, *, trigger_auto: bool = True) -> None:
    fp = _fingerprint(image_bytes)
    st.session_state.image_bytes = image_bytes
    st.session_state.image_name = name
    st.session_state.image_fp = fp
    if trigger_auto and st.session_state.auto_pipeline and fp != st.session_state.last_processed_fp:
        st.session_state.pending_auto = True


def _run_pipeline(image_bytes: bytes, api_key: str | None, *, run_ev: bool = True) -> None:
    """Vision (+ EV opcional) sobre una captura."""
    with st.spinner("Detectando cartas con GPT-4o…"):
        cards = detect_cards_from_image(
            image_bytes,
            filename=st.session_state.image_name,
            api_key=api_key,
            detail="high",
        )
    _set_hand(cards)
    st.session_state.last_processed_fp = st.session_state.image_fp
    st.session_state.pending_auto = False

    if run_ev:
        progress = st.progress(0.0, text="Calculando EV óptimo…")

        def _cb(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Hold {done}/{total}…")

        t0 = time.perf_counter()
        results = analyze_hand(cards, progress_callback=_cb)
        st.session_state.analysis_results = results
        st.session_state.last_elapsed = time.perf_counter() - t0
        progress.progress(1.0, text="Listo")


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

    st.session_state.auto_pipeline = st.toggle(
        "Procesar automáticamente",
        value=st.session_state.auto_pipeline,
        help="Al capturar/pegar/subir: detecta cartas y calcula el HOLD óptimo.",
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
        "**Flujo rápido YouTube**\n\n"
        "1. Pausa en las 5 cartas\n"
        "2. Pulsa **Capturar pantalla del juego**\n"
        "3. Elige la pestaña de YouTube/casino\n"
        "4. Espera el HOLD óptimo (dorado)"
    )

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.markdown('<div class="main-title">♠ Asistente Jacks or Better 9/6</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Captura de pantalla → GPT-4o → HOLD óptimo (automático)</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1) Captura rápida
# ---------------------------------------------------------------------------

st.subheader("1 · Capturar y analizar")
st.caption(
    "Pulsa el botón, elige la **pestaña o ventana** del Video Póker "
    "(YouTube / casino) y confirma. El navegador no puede capturar en silencio: "
    "hay que elegir la fuente una vez por captura."
)

captured = screen_capture_button(
    label="📸 Capturar pantalla del juego y analizar",
    key="screen_grab",
)
if captured and captured.get("data_url"):
    ts = captured.get("ts")
    if ts != st.session_state.last_capture_ts:
        try:
            img_bytes = data_url_to_bytes(captured["data_url"])
            st.session_state.last_capture_ts = ts
            _store_image(img_bytes, "screen_capture.jpg", trigger_auto=True)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer la captura: {exc}")

st.markdown("##### Otras formas de cargar la imagen")
c_up, c_paste = st.columns([2, 1])
with c_up:
    uploaded = st.file_uploader(
        "Screenshot archivo",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
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
            paste_result.image_data.convert("RGB").save(buf, format="JPEG", quality=85)
            new_bytes = buf.getvalue()
            if _fingerprint(new_bytes) != st.session_state.image_fp:
                _store_image(new_bytes, "clipboard.jpg", trigger_auto=True)
                st.rerun()
            else:
                st.caption("✓ Pegada del portapapeles")
    except Exception:
        st.caption("También puedes subir un archivo")

if uploaded is not None:
    new_bytes = uploaded.getvalue()
    new_fp = _fingerprint(new_bytes)
    if new_fp != st.session_state.image_fp:
        _store_image(new_bytes, uploaded.name, trigger_auto=True)
        st.rerun()

image_bytes = st.session_state.image_bytes
if image_bytes:
    st.image(image_bytes, caption=st.session_state.image_name or "Captura", use_container_width=True)

# Pipeline automático pendiente
if st.session_state.pending_auto and image_bytes:
    try:
        _run_pipeline(image_bytes, api_key_ui or None, run_ev=True)
        st.success(
            f"Detectadas: **{format_hand_pretty(st.session_state.hand)}** · EV calculado"
        )
        st.rerun()
    except CardError as exc:
        st.session_state.pending_auto = False
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.session_state.pending_auto = False
        st.error(f"Error al procesar: {exc}")

# Botón manual si auto está off
manual_cols = st.columns([1, 1, 2])
with manual_cols[0]:
    if st.button(
        "Detectar + calcular ahora",
        type="primary",
        disabled=not bool(image_bytes),
        use_container_width=True,
    ):
        try:
            _run_pipeline(image_bytes, api_key_ui or None, run_ev=True)
            st.rerun()
        except CardError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error: {exc}")

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

st.subheader("2 · Cartas (corrige si la IA falló)")

if not st.session_state.hand_ready:
    st.info("Captura la pantalla del juego para empezar.")
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
        # Si el usuario corrigió, invalidar EV previo
        if hand != st.session_state.hand:
            st.session_state.hand = hand
            st.session_state.analysis_results = None
        detail = evaluate_hand_detail(hand)
        st.caption(
            f"Mano: {format_hand_pretty(hand)} · "
            f"pat → **{detail['name_es']}** ({detail['payout']}x)"
        )
    except CardError as exc:
        st.error(f"Mano inválida: {exc}")

    st.subheader("3 · Jugada óptima")

    if hand_ok and st.session_state.analysis_results is None:
        if st.button("Recalcular EV con la mano corregida", type="primary"):
            progress = st.progress(0.0, text="Calculando…")

            def _cb(done: int, total: int) -> None:
                progress.progress(done / total, text=f"Hold {done}/{total}…")

            t0 = time.perf_counter()
            try:
                st.session_state.analysis_results = analyze_hand(hand, progress_callback=_cb)
                st.session_state.last_elapsed = time.perf_counter() - t0
                progress.progress(1.0, text="Listo")
                st.rerun()
            except CardError as exc:
                st.error(str(exc))

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

        st.markdown("#### Top 5 combinaciones")
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
