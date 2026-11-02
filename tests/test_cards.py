import random

import pytest
from pokerkit import StandardHighHand

from pokerlab.agents.preflop import hand_class, preflop_equity, preflop_percentile
from pokerlab.engine.cards import (
    FLUSH, FULL_HOUSE, PAIR, STRAIGHT, STRAIGHT_FLUSH, TWO_PAIR,
    PK_CARDS, card_str, cards_str, category, estimate_equity, evaluate, from_pokerkit, parse_cards,
)


def test_card_roundtrip():
    for c in range(52):
        assert parse_cards(card_str(c)) == (c,)
        assert from_pokerkit(PK_CARDS[c]) == c
    assert cards_str(parse_cards("AsKd2c")) == "AsKd2c"


@pytest.mark.parametrize("cards, expected", [
    ("Ah2c3d4s5h9cKd", STRAIGHT),  # wheel
    ("AhKhQhJhTh2c3d", STRAIGHT_FLUSH),
    ("2h3h4h5h6h7c8d", STRAIGHT_FLUSH),
    ("AhAdAcKsKd2c3d", FULL_HOUSE),
    ("AhAdAcKsKdKc3d", FULL_HOUSE),  # two trips
    ("2h7h9hJhKh2c3d", FLUSH),
    ("AhAdKsKd2c2h7c", TWO_PAIR),  # three pairs
    ("AhAd2c5d9hJsKc", PAIR),
])
def test_categories(cards, expected):
    assert category(evaluate(parse_cards(cards))) == expected


def test_matches_pokerkit_on_random_hands():
    rng = random.Random(123)
    for _ in range(3000):
        cards = rng.sample(range(52), 9)
        a, b = cards[:7], cards[2:]
        mine = (evaluate(a) > evaluate(b)) - (evaluate(a) < evaluate(b))
        pa = StandardHighHand.from_game(cards_str(a))
        pb = StandardHighHand.from_game(cards_str(b))
        theirs = (pa > pb) - (pa < pb)
        assert mine == theirs, (cards_str(a), cards_str(b))


def test_equity_estimates_are_sane():
    rng = random.Random(1)
    aces = estimate_equity(parse_cards("AsAh"), (), 1, 3000, rng)
    assert 0.82 < aces < 0.88
    nuts = estimate_equity(parse_cards("AsKs"), parse_cards("QsJsTs2d3c"), 3, 200, rng)
    assert nuts == 1.0


def test_preflop_table():
    assert hand_class(parse_cards("KdAd")) == "AKs"
    assert hand_class(parse_cards("7c2d")) == "72o"
    assert preflop_percentile(parse_cards("AsAh")) == 0.0
    assert preflop_percentile(parse_cards("7c2d")) > 0.9
    assert preflop_equity(parse_cards("AsAh"), 1) > preflop_equity(parse_cards("AsAh"), 8)
