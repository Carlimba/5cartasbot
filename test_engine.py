"""
Pruebas rápidas del evaluador y del motor de EV.
Ejecutar: python test_engine.py
"""

from __future__ import annotations

import time

from card_evaluator import CardError, classify_hand, evaluate_payout, parse_cards
from math_engine import analyze_hand_fast, expected_value_for_hold


def test_classify() -> None:
    assert classify_hand(["AH", "KH", "QH", "JH", "10H"]) == "royal_flush"
    assert evaluate_payout(["AH", "KH", "QH", "JH", "10H"]) == 800
    assert classify_hand(["9H", "10H", "JH", "QH", "KH"]) == "straight_flush"
    assert classify_hand(["AH", "AD", "AC", "AS", "2H"]) == "four_of_a_kind"
    assert classify_hand(["AH", "AD", "AC", "2S", "2H"]) == "full_house"
    assert classify_hand(["AH", "3H", "5H", "7H", "9H"]) == "flush"
    assert classify_hand(["AH", "2D", "3C", "4S", "5H"]) == "straight"  # wheel
    assert classify_hand(["AH", "AD", "AC", "5S", "9H"]) == "three_of_a_kind"
    assert classify_hand(["AH", "AD", "2C", "2S", "9H"]) == "two_pair"
    assert classify_hand(["JH", "JD", "2C", "5S", "9H"]) == "jacks_or_better"
    assert classify_hand(["10H", "10D", "2C", "5S", "9H"]) == "nothing"
    assert classify_hand(["AH", "KD", "2C", "5S", "9H"]) == "nothing"
    print("OK classify_hand / payouts")


def test_duplicates() -> None:
    try:
        parse_cards(["AH", "AH", "2C", "3D", "4S"])
        raise AssertionError("Debía fallar por duplicados")
    except CardError:
        print("OK duplicados rechazados")


def test_stand_pat_royal() -> None:
    hand = ["AH", "KH", "QH", "JH", "10H"]
    # Máscara 31 = hold todas
    r = expected_value_for_hold(hand, 31)
    assert r.ev == 800.0
    print("OK stand-pat royal EV=800")


def test_full_analysis_pair() -> None:
    # Pareja de ases + basura: lo óptimo casi siempre es HOLD los dos ases
    hand = ["AH", "AD", "3C", "7S", "9H"]
    t0 = time.perf_counter()
    ranked = analyze_hand_fast(hand)
    elapsed = time.perf_counter() - t0
    best = ranked[0]
    print(f"Mano: {hand}")
    print(f"Optimo: {best.held_cards}  EV={best.ev:.6f}  ({elapsed:.2f}s)")
    assert set(best.held_cards) == {"AH", "AD"}
    print("OK hold pareja de ases")


if __name__ == "__main__":
    test_classify()
    test_duplicates()
    test_stand_pat_royal()
    test_full_analysis_pair()
    print("\nTodas las pruebas pasaron.")
