"""Preflop hand-class equity table.

``data/preflop_equity.json`` holds Monte Carlo equity for each of the 169
starting-hand classes against 1..8 random opponents. Regenerate it with::

    python -m pokerlab.agents.preflop --samples 6000
"""

from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from pokerlab.engine.cards import RANKS, estimate_equity

DATA_PATH = Path(__file__).parent / "data" / "preflop_equity.json"
MAX_OPPONENTS = 8


def hand_class(hole: tuple[int, int] | list[int]) -> str:
    """Return the class label, e.g. ``AA``, ``AKs``, ``T9o``."""
    a, b = hole
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb = rb, ra
    if ra == rb:
        return RANKS[ra] * 2
    return RANKS[ra] + RANKS[rb] + ("s" if (a & 3) == (b & 3) else "o")


def all_classes() -> list[str]:
    labels = []
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            if hi == lo:
                labels.append(RANKS[hi] * 2)
            else:
                labels.append(RANKS[hi] + RANKS[lo] + "s")
                labels.append(RANKS[hi] + RANKS[lo] + "o")
    return labels


def combos(label: str) -> int:
    return 6 if len(label) == 2 else (4 if label[2] == "s" else 12)


def _representative(label: str) -> tuple[int, int]:
    hi, lo = RANKS.index(label[0]), RANKS.index(label[1])
    if len(label) == 2:
        return hi * 4, hi * 4 + 1
    return (hi * 4, lo * 4) if label[2] == "s" else (hi * 4, lo * 4 + 1)


def _compute(args: tuple[str, int, int]) -> tuple[str, list[float]]:
    label, samples, seed = args
    rng = random.Random(f"{seed}-{label}")
    hole = _representative(label)
    return label, [
        round(estimate_equity(hole, (), n, samples, rng), 4)
        for n in range(1, MAX_OPPONENTS + 1)
    ]


def build_table(samples: int, seed: int = 7, processes: int | None = None) -> dict:
    jobs = [(label, samples, seed) for label in all_classes()]
    with Pool(processes) as pool:
        rows = dict(pool.map(_compute, jobs))
    return {"samples": samples, "seed": seed, "equity": rows}


@lru_cache(maxsize=1)
def _table() -> dict[str, list[float]]:
    return json.loads(DATA_PATH.read_text())["equity"]


@lru_cache(maxsize=1)
def _percentiles() -> dict[str, float]:
    """Fraction of all 1326 combos ranked *above* each class.

    Ranked by average equity against 1 and 3 opponents, which keeps both
    heads-up value (big offsuit aces) and multiway playability (pairs,
    suited connectors) in view. 0.0 = best hand, approaching 1.0 = worst.
    """
    table = _table()
    order = sorted(table, key=lambda k: -(table[k][0] + table[k][2]))
    result, above = {}, 0
    for label in order:
        result[label] = above / 1326
        above += combos(label)
    return result


def preflop_equity(hole, opponents: int) -> float:
    opponents = max(1, min(MAX_OPPONENTS, opponents))
    return _table()[hand_class(hole)][opponents - 1]


def preflop_percentile(hole) -> float:
    """0.0 for AA up to ~0.99 for the worst hands. 'Top 15%' == percentile < 0.15."""
    return _percentiles()[hand_class(hole)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(build_table(args.samples, args.seed), indent=1))
    print(f"wrote {DATA_PATH}")


if __name__ == "__main__":
    main()
