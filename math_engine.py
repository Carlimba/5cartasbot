"""
Motor de Valor Esperado (EV) exacto para Video Póker Jacks or Better 9/6.

Para una mano de 5 cartas:
  - Genera las 32 máscaras de retención (hold)
  - Para cada máscara, enumera TODAS las combinaciones de cartas del mazo restante
  - Calcula EV = suma(pago) / número_de_draws

La identidad combinatoria garantiza que en total se evalúan C(52,5) = 2_598_960
manos finales a lo largo de las 32 opciones (exactitud matemática completa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

from card_evaluator import (
    PAYTABLE_9_6,
    CardError,
    classify_hand,
    evaluate_payout,
    format_card_pretty,
    format_hand_pretty,
    full_deck,
    normalize_card,
    parse_cards,
)


@dataclass(frozen=True)
class HoldResult:
    """Resultado de EV para una estrategia de retención concreta."""

    hold_mask: int  # bits 0..4: 1 = retener carta en esa posición
    held_cards: tuple[str, ...]
    discarded_cards: tuple[str, ...]
    hold_labels: tuple[str, ...]  # cartas en formato pretty que se HOLD
    num_draws: int
    total_payout: int
    ev: float
    is_optimal: bool = False

    @property
    def hold_description(self) -> str:
        if not self.held_cards:
            return "DESCARTAR TODAS"
        return "HOLD: " + "  ".join(format_card_pretty(c) for c in self.held_cards)

    @property
    def positions_1based(self) -> list[int]:
        """Posiciones a pulsar HOLD (1-5, izquierda a derecha)."""
        return [i + 1 for i in range(5) if (self.hold_mask >> i) & 1]


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _mask_to_holds(mask: int, hand: Sequence[str]) -> tuple[list[str], list[str]]:
    held = [hand[i] for i in range(5) if (mask >> i) & 1]
    discarded = [hand[i] for i in range(5) if not ((mask >> i) & 1)]
    return held, discarded


def expected_value_for_hold(
    hand: Sequence[str],
    hold_mask: int,
    deck_remaining: Sequence[str] | None = None,
    paytable: dict[str, int] | None = None,
) -> HoldResult:
    """
    Calcula el EV exacto de una máscara de hold sobre `hand`.

    Args:
        hand: 5 cartas normalizadas.
        hold_mask: entero 0..31; bit i = retener hand[i].
        deck_remaining: mazo restante (si None, se calcula).
        paytable: tabla de pagos (default 9/6).
    """
    table = paytable or PAYTABLE_9_6
    held, discarded = _mask_to_holds(hold_mask, hand)
    n_draw = 5 - len(held)

    if deck_remaining is None:
        held_set = set(hand)
        deck_remaining = [c for c in full_deck() if c not in held_set]

    if n_draw == 0:
        payout = evaluate_payout(held, table)
        return HoldResult(
            hold_mask=hold_mask,
            held_cards=tuple(held),
            discarded_cards=tuple(discarded),
            hold_labels=tuple(format_card_pretty(c) for c in held),
            num_draws=1,
            total_payout=payout,
            ev=float(payout),
        )

    total = 0
    count = 0
    for draw in combinations(deck_remaining, n_draw):
        final_hand = held + list(draw)
        total += evaluate_payout(final_hand, table)
        count += 1

    assert count == _n_choose_k(len(deck_remaining), n_draw)
    ev = total / count if count else 0.0
    return HoldResult(
        hold_mask=hold_mask,
        held_cards=tuple(held),
        discarded_cards=tuple(discarded),
        hold_labels=tuple(format_card_pretty(c) for c in held),
        num_draws=count,
        total_payout=total,
        ev=ev,
    )


def analyze_hand(
    cards: Sequence[str],
    paytable: dict[str, int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[HoldResult]:
    """
    Analiza las 32 estrategias de hold y devuelve resultados ordenados por EV desc.

    La opción óptima queda marcada con `is_optimal=True` (todas las empatadas
    en el máximo EV se marcan).

    Args:
        cards: 5 cartas (acepta notación flexible; se normalizan).
        paytable: tabla de pagos; por defecto Jacks or Better 9/6.
        progress_callback: opcional `callback(done, total)` con total=32.
    """
    hand = parse_cards(cards)
    table = paytable or PAYTABLE_9_6
    hand_set = set(hand)
    deck_remaining = [c for c in full_deck() if c not in hand_set]
    if len(deck_remaining) != 47:
        raise CardError("Estado de baraja inconsistente tras quitar la mano.")

    results: list[HoldResult] = []
    total_masks = 32
    for mask in range(total_masks):
        result = expected_value_for_hold(hand, mask, deck_remaining, table)
        results.append(result)
        if progress_callback is not None:
            progress_callback(mask + 1, total_masks)

    best_ev = max(r.ev for r in results)
    # Tolerancia por redondeo flotante (los pagos son enteros; el cociente es exacto en Q)
    eps = 1e-12
    marked: list[HoldResult] = []
    for r in results:
        marked.append(
            HoldResult(
                hold_mask=r.hold_mask,
                held_cards=r.held_cards,
                discarded_cards=r.discarded_cards,
                hold_labels=r.hold_labels,
                num_draws=r.num_draws,
                total_payout=r.total_payout,
                ev=r.ev,
                is_optimal=abs(r.ev - best_ev) <= eps,
            )
        )

    marked.sort(key=lambda r: (-r.ev, -bin(r.hold_mask).count("1"), r.hold_mask))
    return marked


def optimal_play(cards: Sequence[str], paytable: dict[str, int] | None = None) -> HoldResult:
    """Atajo: devuelve la mejor estrategia de hold (primera si hay empate)."""
    ranked = analyze_hand_fast(cards, paytable=paytable)
    return ranked[0]


def hand_summary(cards: Sequence[str]) -> dict:
    """Resumen de la mano actual (antes del draw)."""
    hand = parse_cards(cards)
    category = classify_hand(hand)
    payout = PAYTABLE_9_6[category]
    return {
        "cards": hand,
        "pretty": format_hand_pretty(hand),
        "category": category,
        "payout_if_stand_pat": payout,
    }


def results_to_rows(results: Sequence[HoldResult]) -> list[dict]:
    """Convierte resultados a filas amigables para tablas Streamlit/pandas."""
    rows = []
    for rank, r in enumerate(results, start=1):
        positions = r.positions_1based
        rows.append(
            {
                "Rank": rank,
                "Óptimo": "★ SÍ" if r.is_optimal else "",
                "HOLD (cartas)": (
                    "— DESCARTAR TODAS —"
                    if not r.held_cards
                    else "  ".join(r.hold_labels)
                ),
                "Posiciones HOLD": (
                    "ninguna"
                    if not positions
                    else ", ".join(str(p) for p in positions)
                ),
                "EV": round(r.ev, 6),
                "Draws evaluados": r.num_draws,
                "Máscara": format(r.hold_mask, "05b"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluador acelerado interno (enteros) para reducir overhead de strings
# ---------------------------------------------------------------------------

def _card_to_id(card: str) -> int:
    """Mapea carta canónica a id 0..51 (suit*13 + rank)."""
    c = normalize_card(card)
    from card_evaluator import RANK_VALUES, SUITS

    rank = RANK_VALUES[c[:-1]]
    suit = SUITS.index(c[-1])
    return suit * 13 + rank


def _id_to_rank_suit(card_id: int) -> tuple[int, int]:
    return card_id % 13, card_id // 13


def _payout_from_ids(ids: Sequence[int], pay_arr: dict[str, int]) -> int:
    """Evaluación rápida de pago a partir de ids 0..51."""
    ranks = [0] * 5
    suits = [0] * 5
    for i, cid in enumerate(ids):
        ranks[i] = cid % 13
        suits[i] = cid // 13

    sorted_ranks = sorted(ranks)
    is_flush = suits[0] == suits[1] == suits[2] == suits[3] == suits[4]

    # Escalera
    is_straight = sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 5)) or (
        sorted_ranks == [0, 1, 2, 3, 12]
    )

    if is_straight and is_flush:
        if sorted_ranks == [8, 9, 10, 11, 12]:
            return pay_arr["royal_flush"]
        return pay_arr["straight_flush"]

    # Conteos de rango sin Counter (más rápido)
    count = [0] * 13
    for r in ranks:
        count[r] += 1
    freqs = sorted((c for c in count if c), reverse=True)

    if freqs[0] == 4:
        return pay_arr["four_of_a_kind"]
    if freqs == [3, 2]:
        return pay_arr["full_house"]
    if is_flush:
        return pay_arr["flush"]
    if is_straight:
        return pay_arr["straight"]
    if freqs[0] == 3:
        return pay_arr["three_of_a_kind"]
    if freqs[:2] == [2, 2]:
        return pay_arr["two_pair"]
    if freqs[0] == 2:
        pair_rank = next(i for i, c in enumerate(count) if c == 2)
        if pair_rank >= 9:  # J,Q,K,A
            return pay_arr["jacks_or_better"]
        return pay_arr["nothing"]
    return pay_arr["nothing"]


def analyze_hand_fast(
    cards: Sequence[str],
    paytable: dict[str, int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[HoldResult]:
    """
    Misma API que `analyze_hand`, pero usando ids enteros y evaluador rápido.
    Recomendada para la UI (≈2.6M evaluaciones).
    """
    hand = parse_cards(cards)
    table = paytable or PAYTABLE_9_6
    hand_ids = [_card_to_id(c) for c in hand]
    hand_id_set = set(hand_ids)
    deck_ids = [i for i in range(52) if i not in hand_id_set]

    results: list[HoldResult] = []
    for mask in range(32):
        held_idx = [i for i in range(5) if (mask >> i) & 1]
        held_ids = [hand_ids[i] for i in held_idx]
        held_cards = [hand[i] for i in held_idx]
        discarded = [hand[i] for i in range(5) if not ((mask >> i) & 1)]
        n_draw = 5 - len(held_ids)

        if n_draw == 0:
            total = _payout_from_ids(held_ids, table)
            count = 1
        else:
            total = 0
            count = 0
            base = held_ids
            for draw in combinations(deck_ids, n_draw):
                final_ids = base + list(draw)
                total += _payout_from_ids(final_ids, table)
                count += 1

        ev = total / count if count else 0.0
        results.append(
            HoldResult(
                hold_mask=mask,
                held_cards=tuple(held_cards),
                discarded_cards=tuple(discarded),
                hold_labels=tuple(format_card_pretty(c) for c in held_cards),
                num_draws=count,
                total_payout=total,
                ev=ev,
            )
        )
        if progress_callback is not None:
            progress_callback(mask + 1, 32)

    best_ev = max(r.ev for r in results)
    eps = 1e-12
    marked = [
        HoldResult(
            hold_mask=r.hold_mask,
            held_cards=r.held_cards,
            discarded_cards=r.discarded_cards,
            hold_labels=r.hold_labels,
            num_draws=r.num_draws,
            total_payout=r.total_payout,
            ev=r.ev,
            is_optimal=abs(r.ev - best_ev) <= eps,
        )
        for r in results
    ]
    marked.sort(key=lambda r: (-r.ev, -bin(r.hold_mask).count("1"), r.hold_mask))
    return marked
