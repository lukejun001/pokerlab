"""Hand classes, batched dealing and a vectorized 7-card evaluator.

The solver works on the 169 strategically distinct starting-hand classes
(``AA``, ``AKs``, ``AKo``...). Preflop this is lossless: suits only matter
through card removal, which the solver keeps exactly by dealing real cards.

``evaluate_batch`` is a NumPy port of :func:`pokerlab.engine.cards.evaluate`
and returns identical values (``tests/test_solver_hands.py`` checks this).
"""

from __future__ import annotations

import numpy as np

from pokerlab.agents.preflop import all_classes, combos, hand_class
from pokerlab.engine.cards import (
    _POPCOUNT,
    _TOP5,
    FLUSH,
    FULL_HOUSE,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    STRAIGHT_HIGH,
    TRIPS,
    TWO_PAIR,
)

CLASS_LABELS: tuple[str, ...] = tuple(all_classes())
NUM_CLASSES = len(CLASS_LABELS)
CLASS_INDEX = {label: i for i, label in enumerate(CLASS_LABELS)}
CLASS_COMBOS = np.array([combos(label) for label in CLASS_LABELS], dtype=np.float64)

# (card, card) -> class index
PAIR_CLASS = np.full((52, 52), -1, dtype=np.int16)
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            PAIR_CLASS[_a, _b] = CLASS_INDEX[hand_class((_a, _b))]

_POP = np.array(_POPCOUNT, dtype=np.int64)
_TOP = np.array(_TOP5, dtype=np.int64)
_STRAIGHT = np.array(STRAIGHT_HIGH, dtype=np.int64)
_RANKS = np.arange(13, dtype=np.int64)


def class_of(hole) -> int:
    return int(PAIR_CLASS[hole[0], hole[1]])


def _highest(mask_rows: np.ndarray) -> np.ndarray:
    """Highest rank where ``mask_rows`` (N, 13 bool) is true, else -1."""
    flipped = mask_rows[:, ::-1]
    idx = 12 - flipped.argmax(axis=1)
    return np.where(mask_rows.any(axis=1), idx, -1)


def evaluate_batch(cards: np.ndarray) -> np.ndarray:
    """Strength of each row of 7 cards (N, 7) -> (N,) int64, higher is better."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    ranks = cards >> 2
    suits = cards & 3
    bits = np.left_shift(1, ranks)
    counts = (ranks[:, :, None] == _RANKS).sum(axis=1)  # (N, 13)
    suit_masks = np.stack([np.bitwise_or.reduce(np.where(suits == k, bits, 0), axis=1) for k in range(4)], axis=1)
    rank_mask = np.bitwise_or.reduce(bits, axis=1)

    out = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)

    suit_pop = _POP[suit_masks]
    flush_suit = suit_pop.argmax(axis=1)
    has_flush = suit_pop.max(axis=1) >= 5
    flush_mask = suit_masks[np.arange(n), flush_suit]
    sf_high = _STRAIGHT[flush_mask]
    sf = has_flush & (sf_high >= 0)
    out[sf] = STRAIGHT_FLUSH << 20 | sf_high[sf] << 16
    fl = has_flush & ~sf
    out[fl] = FLUSH << 20 | _TOP[flush_mask[fl]]
    done |= has_flush

    quads = _highest(counts == 4)
    is3 = counts == 3
    trips0 = _highest(is3)
    trips1 = _highest(is3 & (_RANKS < trips0[:, None]))
    pair0 = _highest(counts == 2)
    pair1 = _highest((counts == 2) & (_RANKS < pair0[:, None]))
    one = np.int64(1)

    m = ~done & (quads >= 0)
    kick = rank_mask[m] & ~(one << quads[m])
    out[m] = QUADS << 20 | quads[m] << 16 | (_TOP[kick] >> 16) << 12
    done |= m

    m = ~done & (trips0 >= 0) & ((trips1 >= 0) | (pair0 >= 0))
    second = np.maximum(trips1[m], pair0[m])
    out[m] = FULL_HOUSE << 20 | trips0[m] << 16 | second << 12
    done |= m

    st_high = _STRAIGHT[rank_mask]
    m = ~done & (st_high >= 0)
    out[m] = STRAIGHT << 20 | st_high[m] << 16
    done |= m

    m = ~done & (trips0 >= 0)
    kick = _TOP[rank_mask[m] & ~(one << trips0[m])] >> 12
    out[m] = TRIPS << 20 | trips0[m] << 16 | kick << 8
    done |= m

    m = ~done & (pair1 >= 0)
    kmask = rank_mask[m] & ~(one << pair0[m]) & ~(one << pair1[m])
    out[m] = TWO_PAIR << 20 | pair0[m] << 16 | pair1[m] << 12 | (_TOP[kmask] >> 16) << 8
    done |= m

    m = ~done & (pair0 >= 0)
    kick = _TOP[rank_mask[m] & ~(one << pair0[m])] >> 8
    out[m] = PAIR << 20 | pair0[m] << 16 | kick << 4
    done |= m

    m = ~done
    out[m] = _TOP[rank_mask[m]]
    return out


def deal_batch(rng: np.random.Generator, size: int, players: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deal ``size`` independent hands.

    Returns ``holes`` (size, players, 2), ``classes`` (size, players) and
    ``ranks`` (size, players): each player's final 7-card strength on a random
    five-card board (the checked-down showdown).
    """
    deck = rng.random((size, 52)).argsort(axis=1)[:, : 2 * players + 5]
    holes = deck[:, : 2 * players].reshape(size, players, 2)
    board = deck[:, 2 * players:]
    classes = PAIR_CLASS[holes[:, :, 0], holes[:, :, 1]].astype(np.int64)
    seven = np.concatenate([holes, np.broadcast_to(board[:, None, :], (size, players, 5))], axis=2)
    ranks = evaluate_batch(seven.reshape(size * players, 7)).reshape(size, players)
    return holes, classes, ranks
