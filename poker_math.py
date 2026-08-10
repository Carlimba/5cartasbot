"""
Motor matemático de EV exacto — Jacks or Better 9/6.

API pública para la app Streamlit. Reutiliza el evaluador y el enumerador
combinatorio (32 holds × mazo restante de 47 cartas).
"""

from __future__ import annotations

from typing import Callable, Sequence

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
    full_deck,
    parse_cards,
)
from math_engine import HoldResult, analyze_hand_fast, results_to_rows

__all__ = [
    "HAND_CATEGORY_ES",
    "PAYTABLE_9_6",
    "RANKS",
    "SUIT_NAMES",
    "SUIT_SYMBOLS",
    "SUITS",
    "CardError",
    "HoldResult",
    "all_card_codes",
    "analyze_hand",
    "evaluate_hand_detail",
    "format_card_pretty",
    "format_hand_pretty",
    "optimal_hold",
    "parse_cards",
    "results_to_rows",
    "top_n_strategies",
]


def all_card_codes() -> list[str]:
    """Las 52 cartas en notación canónica (para selectboxes de corrección)."""
    return full_deck()


def analyze_hand(
    cards: Sequence[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[HoldResult]:
    """
    Calcula el EV exacto de las 32 combinaciones de HOLD.

    Returns:
        Lista ordenada de mayor a menor EV; las óptimas tienen `is_optimal=True`.
    """
    return analyze_hand_fast(cards, paytable=PAYTABLE_9_6, progress_callback=progress_callback)


def optimal_hold(cards: Sequence[str]) -> HoldResult:
    """Devuelve la estrategia de retención con mayor EV."""
    return analyze_hand(cards)[0]


def top_n_strategies(results: Sequence[HoldResult], n: int = 5) -> list[dict]:
    """Top N filas listas para `st.dataframe`."""
    rows = results_to_rows(results)
    return rows[:n]
