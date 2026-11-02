"""Card encoding, a fast pure-Python hand evaluator, and Monte Carlo equity.

Cards are ints 0..51: ``rank * 4 + suit`` where rank 0 = deuce .. 12 = ace and
suit 0..3 = c, d, h, s. Agents see cards in this form; the engine converts to
PokerKit ``Card`` objects only when dealing.

PokerKit decides real showdowns. This evaluator exists so agents can estimate
hand strength and equity cheaply (PokerKit's own hand objects are too slow to
call thousands of times per decision). ``tests/test_cards.py`` cross-checks the
two evaluators on random hands.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from pokerkit import Card

RANKS = "23456789TJQKA"
SUITS = "cdhs"

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "trips", "straight",
    "flush", "full house", "quads", "straight flush",
)


def card_id(text: str) -> int:
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def card_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def parse_cards(text: str) -> tuple[int, ...]:
    text = text.replace(" ", "")
    return tuple(card_id(text[i:i + 2]) for i in range(0, len(text), 2))


def cards_str(cards: Iterable[int]) -> str:
    return "".join(card_str(c) for c in cards)


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


PK_CARDS: tuple[Card, ...] = tuple(next(iter(Card.parse(card_str(c)))) for c in range(52))
_FROM_PK = {(c.rank, c.suit): i for i, c in enumerate(PK_CARDS)}


def from_pokerkit(card: Card) -> int:
    return _FROM_PK[(card.rank, card.suit)]


# --- lookup tables over 13-bit rank masks ---------------------------------

def _straight_high(mask: int) -> int:
    for high in range(12, 3, -1):
        window = 0b11111 << (high - 4)
        if mask & window == window:
            return high
    if mask & 0b1000000001111 == 0b1000000001111:  # wheel A-2-3-4-5
        return 3
    return -1


def _top_ranks_value(mask: int, count: int) -> int:
    value, taken = 0, 0
    for r in range(12, -1, -1):
        if taken == count:
            break
        if mask >> r & 1:
            value = value << 4 | r
            taken += 1
    return value << 4 * (count - taken)


STRAIGHT_HIGH = tuple(_straight_high(m) for m in range(8192))
_TOP5 = tuple(_top_ranks_value(m, 5) for m in range(8192))
_POPCOUNT = tuple(m.bit_count() for m in range(8192))


def evaluate(cards: Sequence[int]) -> int:
    """Return a comparable strength for 5-7 cards (higher is better).

    Also works for fewer cards (pairs/trips only), which is useful for
    describing what a flop or turn board makes on its own.
    """
    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suit_masks[c & 3] |= 1 << r

    for m in suit_masks:
        if _POPCOUNT[m] >= 5:
            high = STRAIGHT_HIGH[m]
            if high >= 0:
                return STRAIGHT_FLUSH << 20 | high << 16
            return FLUSH << 20 | _TOP5[m]

    quads = -1
    trips: list[int] = []
    pairs: list[int] = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quads = r
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)

    rank_mask = suit_masks[0] | suit_masks[1] | suit_masks[2] | suit_masks[3]
    if quads >= 0:
        kicker_mask = rank_mask & ~(1 << quads)
        return QUADS << 20 | quads << 16 | (_TOP5[kicker_mask] >> 16) << 12
    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1:] + pairs)
        return FULL_HOUSE << 20 | trips[0] << 16 | second << 12
    high = STRAIGHT_HIGH[rank_mask]
    if high >= 0:
        return STRAIGHT << 20 | high << 16
    if trips:
        kickers = _TOP5[rank_mask & ~(1 << trips[0])] >> 12  # top 2 kickers
        return TRIPS << 20 | trips[0] << 16 | kickers << 8
    if len(pairs) >= 2:
        kicker_mask = rank_mask & ~(1 << pairs[0]) & ~(1 << pairs[1])
        kicker = _TOP5[kicker_mask] >> 16
        return TWO_PAIR << 20 | pairs[0] << 16 | pairs[1] << 12 | kicker << 8
    if pairs:
        kickers = _TOP5[rank_mask & ~(1 << pairs[0])] >> 8  # top 3 kickers
        return PAIR << 20 | pairs[0] << 16 | kickers << 4
    return HIGH_CARD << 20 | _TOP5[rank_mask]


def category(value: int) -> int:
    return value >> 20


def estimate_equity(
    hole: Sequence[int],
    board: Sequence[int],
    opponents: int,
    samples: int,
    rng: random.Random,
) -> float:
    """Monte Carlo equity against ``opponents`` uniformly random hands."""
    if opponents <= 0:
        return 1.0
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    draw_count = need + 2 * opponents
    hole = list(hole)
    board = list(board)
    sample = rng.sample
    total = 0.0
    for _ in range(samples):
        drawn = sample(deck, draw_count)
        full_board = board + drawn[:need]
        mine = evaluate(hole + full_board)
        best, ties = 0, 0
        for i in range(need, draw_count, 2):
            value = evaluate(drawn[i:i + 2] + full_board)
            if value > best:
                best, ties = value, 1
            elif value == best:
                ties += 1
        if mine > best:
            total += 1.0
        elif mine == best:
            total += 1.0 / (ties + 1)
    return total / samples
