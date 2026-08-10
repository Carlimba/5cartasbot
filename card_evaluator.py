"""
Evaluador de manos para Video Póker Jacks or Better (tabla 9/6).

Notación de cartas:
  Rango: A, K, Q, J, 10 (o T), 9, 8, 7, 6, 5, 4, 3, 2
  Palo:  H (corazones), D (diamantes), C (tréboles), S (picas)
  Ejemplos: 'AH', 'KH', '10C', 'TC', '2S', '5D'
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Constantes de baraja y pagos
# ---------------------------------------------------------------------------

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("H", "D", "C", "S")
SUIT_NAMES = {
    "H": "Corazones",
    "D": "Diamantes",
    "C": "Tréboles",
    "S": "Picas",
}
SUIT_SYMBOLS = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}  # 2=0 ... A=12

# Tabla 9/6 Jacks or Better (multiplicadores por unidad apostada; Royal a 5 monedas = 800)
PAYTABLE_9_6: dict[str, int] = {
    "royal_flush": 800,
    "straight_flush": 50,
    "four_of_a_kind": 25,
    "full_house": 9,
    "flush": 6,
    "straight": 4,
    "three_of_a_kind": 3,
    "two_pair": 2,
    "jacks_or_better": 1,
    "nothing": 0,
}

HAND_CATEGORY_ES = {
    "royal_flush": "Escalera Real",
    "straight_flush": "Escalera de Color",
    "four_of_a_kind": "Póker",
    "full_house": "Full House",
    "flush": "Color",
    "straight": "Escalera",
    "three_of_a_kind": "Trío",
    "two_pair": "Doble Pareja",
    "jacks_or_better": "Pareja de Jotas o Mejor",
    "nothing": "Nada",
}

# Parejas que pagan: J, Q, K, A  (valores 9..12)
HIGH_PAIR_RANKS = frozenset({9, 10, 11, 12})


class CardError(ValueError):
    """Error de validación de cartas o manos."""


# ---------------------------------------------------------------------------
# Parsing y baraja
# ---------------------------------------------------------------------------

def normalize_card(card: str) -> str:
    """Normaliza una carta a formato canónico: rango + palo (ej. '10H', 'AH')."""
    if not isinstance(card, str) or not card.strip():
        raise CardError(f"Carta inválida: {card!r}")

    raw = card.strip().upper().replace(" ", "")
    # Aceptar símbolos unicode de palo
    raw = (
        raw.replace("♥", "H")
        .replace("♦", "D")
        .replace("♣", "C")
        .replace("♠", "S")
        .replace("HEARTS", "H")
        .replace("DIAMONDS", "D")
        .replace("CLUBS", "C")
        .replace("SPADES", "S")
    )

    if len(raw) < 2:
        raise CardError(f"Carta inválida: {card!r}")

    suit = raw[-1]
    rank = raw[:-1]

    if rank == "T":
        rank = "10"
    if rank == "1":  # por si llega "1H" mal parseado
        raise CardError(f"Carta inválida: {card!r}. ¿Quisiste decir '10{suit}'?")

    if rank not in RANK_VALUES:
        raise CardError(
            f"Rango inválido en {card!r}. Rangos válidos: {', '.join(RANKS)}"
        )
    if suit not in SUITS:
        raise CardError(
            f"Palo inválido en {card!r}. Palos válidos: H, D, C, S "
            f"({', '.join(f'{k}={v}' for k, v in SUIT_NAMES.items())})"
        )
    return f"{rank}{suit}"


def parse_cards(cards: Sequence[str]) -> list[str]:
    """Parsea y valida exactamente 5 cartas sin duplicados."""
    if len(cards) != 5:
        raise CardError(f"Se requieren exactamente 5 cartas; se recibieron {len(cards)}.")

    normalized = [normalize_card(c) for c in cards]
    counts = Counter(normalized)
    dupes = [c for c, n in counts.items() if n > 1]
    if dupes:
        raise CardError(f"Cartas duplicadas: {', '.join(dupes)}")
    return normalized


def full_deck() -> list[str]:
    """Baraja francesa completa (52 cartas) en notación canónica."""
    return [f"{rank}{suit}" for suit in SUITS for rank in RANKS]


def card_to_ints(card: str) -> tuple[int, int]:
    """Devuelve (rank_value 0-12, suit_index 0-3)."""
    c = normalize_card(card)
    rank = c[:-1]
    suit = c[-1]
    return RANK_VALUES[rank], SUITS.index(suit)


def format_card_pretty(card: str) -> str:
    """Formato legible: 'A♥', '10♣', etc."""
    c = normalize_card(card)
    return f"{c[:-1]}{SUIT_SYMBOLS[c[-1]]}"


def format_hand_pretty(cards: Iterable[str]) -> str:
    return "  ".join(format_card_pretty(c) for c in cards)


# ---------------------------------------------------------------------------
# Clasificación de mano
# ---------------------------------------------------------------------------

def _is_straight(sorted_ranks: list[int]) -> bool:
    """True si los 5 rangos ordenados forman escalera (incluye rueda A-2-3-4-5)."""
    # Caso normal: cinco consecutivos
    if sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 5)):
        return True
    # Rueda: A,2,3,4,5  → valores 12,0,1,2,3 ordenados = [0,1,2,3,12]
    return sorted_ranks == [0, 1, 2, 3, 12]


def classify_hand(cards: Sequence[str]) -> str:
    """
    Clasifica una mano de 5 cartas.

    Returns:
        Clave de categoría en PAYTABLE_9_6 / HAND_CATEGORY_ES.
    """
    if len(cards) != 5:
        raise CardError(f"classify_hand requiere 5 cartas; recibió {len(cards)}.")

    ranks: list[int] = []
    suits: list[int] = []
    for card in cards:
        r, s = card_to_ints(card)
        ranks.append(r)
        suits.append(s)

    sorted_ranks = sorted(ranks)
    is_flush = len(set(suits)) == 1
    is_straight = _is_straight(sorted_ranks)

    if is_straight and is_flush:
        # Royal: 10,J,Q,K,A = valores 8,9,10,11,12
        if sorted_ranks == [8, 9, 10, 11, 12]:
            return "royal_flush"
        return "straight_flush"

    rank_counts = Counter(ranks)
    freqs = sorted(rank_counts.values(), reverse=True)

    if freqs[0] == 4:
        return "four_of_a_kind"
    if freqs == [3, 2]:
        return "full_house"
    if is_flush:
        return "flush"
    if is_straight:
        return "straight"
    if freqs[0] == 3:
        return "three_of_a_kind"
    if freqs == [2, 2, 1]:
        return "two_pair"
    if freqs[0] == 2:
        pair_rank = next(r for r, n in rank_counts.items() if n == 2)
        if pair_rank in HIGH_PAIR_RANKS:
            return "jacks_or_better"
        return "nothing"
    return "nothing"


def evaluate_payout(cards: Sequence[str], paytable: dict[str, int] | None = None) -> int:
    """Devuelve el pago (multiplicador) de la mano según la tabla 9/6."""
    table = paytable or PAYTABLE_9_6
    category = classify_hand(cards)
    return table[category]


def evaluate_hand_detail(
    cards: Sequence[str], paytable: dict[str, int] | None = None
) -> dict:
    """Detalle de evaluación: categoría, nombre ES y pago."""
    table = paytable or PAYTABLE_9_6
    category = classify_hand(cards)
    return {
        "category": category,
        "name_es": HAND_CATEGORY_ES[category],
        "payout": table[category],
        "cards": list(cards),
        "pretty": format_hand_pretty(cards),
    }
