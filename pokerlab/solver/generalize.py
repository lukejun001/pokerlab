"""Approximate strategies for stack vectors that were never solved.

A library of solved games (same players, blinds, rake and GTOpen abstraction;
different stacks) is matched to a target game by **effective stacks**: every
pair of players' smaller stack, compared on a log scale. A matched strategy
is translated onto the target game's tree (folds, checks and calls map to
themselves; sized raises map by rank; all-ins map to all-ins, or to the
largest sized raise when the target has no separate all-in, and back).
Several neighbors can be blended with inverse-distance weights.

Whether that is good enough is measured, not assumed: ``measure`` scores the
translated profile inside the target game with GTOpen's exact best-response
evaluation. ``Calibration`` turns those measurements into the matching
distance within which approximations are allowed, and the agent uses
nothing beyond it.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokerlab.solver.game import BettingState, PreflopGame, action_amount, action_kind
from pokerlab.solver.store import StoredStrategy, StrategyStore, family_key

APPROXIMATION_VERSION = "effstack-knn-1"


# --- GTOpen's abstract actions, in Python -------------------------------------------------


def gtopen_tokens(game: PreflopGame, state: BettingState) -> list[str]:
    """The fork's ``legal_actions_asym`` for the player to act (tokens with chip amounts).

    pokerlab's ``BettingState`` already follows PokerKit's rules (which the fork
    ports), so only GTOpen's sizing is added here. ``tests/test_generalize.py``
    checks this against the fork's own trees.
    """
    ab = game.abstraction
    i = state.actor
    stack = game.stacks[i]
    current = max(state.bets)
    owed = current - state.bets[i]
    tokens = []
    if owed > 0:
        tokens.append("f")
        if state.raises or ab.limp:
            tokens.append("c")
    else:
        tokens.append("x")
    bounds = state.raise_bounds(i, game.big_blind)
    if state.raises >= ab.max_raises or bounds is None:
        return tokens
    min_to, _ = bounds
    if state.raises == 0:
        sizes = ab.sb_open_raises if i == game.n - 2 and ab.sb_open_raises else ab.open_raises
        tos = [s * game.big_blind for s in sizes]
    else:
        mults = ab.fourbet_mults if state.raises >= 2 and ab.fourbet_mults is not None else ab.raise_mults
        tos = [current * m for m in mults]
    if ab.add_allin:
        tos.append(stack)
    seen: list[int] = []
    for to in tos:
        to = max(to, min_to)
        if to >= ab.allin_threshold * stack or to > stack:
            to = stack
        to = int(round(to))
        if to <= current or to in seen:
            continue
        seen.append(to)
        tokens.append(("a" if to == stack else "r") + str(to))
    return tokens


def replay(game: PreflopGame, history: list[str]) -> BettingState | None:
    """The state after ``history``, or None if a token is not in ``game``'s abstract tree."""
    state = BettingState.initial(game)
    for token in history:
        if state.closed or token not in gtopen_tokens(game, state):
            return None
        state.apply(action_kind(token), action_amount(token))
    return state


# --- translation between trees ------------------------------------------------------------


def _sized(tokens: list[str]) -> list[str]:
    return sorted((t for t in tokens if t[0] == "r"), key=action_amount)


def map_token(token: str, src: list[str], dst: list[str], relaxed: bool = False) -> str | None:
    """The action in ``dst`` that plays the role of ``token`` from ``src``.

    ``relaxed`` also maps across kinds when the role is missing (raise ->
    call, call <-> check, fold -> check), reaching nodes the strict mapping
    cannot; such matches are coarser and are only used as a second pass.
    """
    kind = token[0]
    if kind in "fxc":
        if token in dst:
            return token
        if not relaxed:
            return None
        for alt in {"f": ("x",), "x": ("c",), "c": ("x",)}[kind]:
            if alt in dst:
                return alt
        return None
    dst_sized, dst_allin = _sized(dst), [t for t in dst if t[0] == "a"]
    if kind == "r":
        rank = _sized(src).index(token)
        if rank < len(dst_sized):
            return dst_sized[rank]
    if dst_allin:
        return dst_allin[0]
    if dst_sized:
        return dst_sized[-1]
    if relaxed:
        return "c" if "c" in dst else ("x" if "x" in dst else None)
    return None


def translate_node(target: PreflopGame, history: list[str], source: StoredStrategy,
                   relaxed: bool = False) -> np.ndarray | None:
    """Strategy table (target actions x 169) at ``history`` of ``target``, read from ``source``.

    The target history is mapped onto the source tree token by token; then
    each source action's probability moves to its role in the target node.
    None when the line does not exist in the source tree.
    """
    state = BettingState.initial(target)
    src_hist: list[str] = []
    for token in history:
        node = source.nodes.get("-".join(src_hist))
        if node is None or state.closed:
            return None
        mapped = map_token(token, gtopen_tokens(target, state), list(node[1]), relaxed)
        if mapped is None:
            return None
        src_hist.append(mapped)
        state.apply(action_kind(token), action_amount(token))
    if state.closed:
        return None
    found = source.node_table("-".join(src_hist))
    if found is None:
        return None
    src_actions, table = found
    dst_actions = gtopen_tokens(target, state)
    out = np.zeros((len(dst_actions), table.shape[1]))
    for k, token in enumerate(src_actions):
        role = map_token(token, list(src_actions), dst_actions)
        if role is None:  # e.g. a raise where the target player can only call: treat as calling
            role = "c" if "c" in dst_actions else ("x" if "x" in dst_actions else dst_actions[0])
        out[dst_actions.index(role)] += table[k]
    total = out.sum(axis=0)
    return out / np.where(total > 0, total, 1.0)


# --- effective-stack matching ---------------------------------------------------------------


def features(stacks_bb) -> np.ndarray:
    """Log effective stack of every pair of seats (preflop order)."""
    s = np.asarray(stacks_bb, dtype=float)
    return np.array([math.log(min(s[i], s[j])) for i, j in itertools.combinations(range(len(s)), 2)])


def distance(a, b) -> float:
    """Whole-table distance: largest log difference in any pairwise effective stack."""
    return float(np.abs(features(a) - features(b)).max())


def node_distance(a, b, actor: int, live) -> float:
    """Distance at one decision: the acting player's effective stack against
    each opponent still in the hand (log scale, largest difference)."""
    worst = 0.0
    for j, alive in enumerate(live):
        if alive and j != actor:
            worst = max(worst, abs(math.log(min(a[actor], a[j])) - math.log(min(b[actor], b[j]))))
    return worst


@dataclass
class Neighbor:
    key: str
    distance: float
    stacks_bb: list[float]


class Library:
    """Solved games of one family (everything but stacks equal), for matching."""

    def __init__(self, store: StrategyStore, template: PreflopGame) -> None:
        self.store = store
        self.family = family_key(template)
        rows = store.db.execute("SELECT key, stacks_bb FROM strategies WHERE family = ?", (self.family,)).fetchall()
        self.entries = [(key, json.loads(stacks)) for key, stacks in rows]

    def __len__(self) -> int:
        return len(self.entries)

    def neighbors(self, stacks_bb, k: int = 1, exclude: str | None = None) -> list[Neighbor]:
        found = [Neighbor(key, distance(stacks_bb, s), s) for key, s in self.entries if key != exclude]
        return sorted(found, key=lambda nb: nb.distance)[:k]

    def node_table(self, target: PreflopGame, history: list[str], k: int = 1, exclude: str | None = None,
                   max_distance: float | None = None) -> tuple[np.ndarray, float] | None:
        """Per-decision match: the ``k`` library games closest at THIS decision
        (acting player's effective stacks vs live opponents) that contain the
        line, blended by inverse distance. Returns (table, distance of the nearest used)."""
        state = replay(target, history)
        if state is None or state.closed:
            return None
        actor, live = state.actor, state.live
        tb = target.stacks_bb()
        ranked = sorted((node_distance(tb, s, actor, live), key) for key, s in self.entries if key != exclude)
        tables, weights, used = [], [], []
        for d, key in ranked:
            if max_distance is not None and d > max_distance:
                break
            strategy = self.store.load(key)
            table = translate_node(target, history, strategy) if strategy else None
            if table is None:
                continue
            tables.append(table)
            weights.append(1.0 / (d + 0.05))
            used.append(d)
            if len(tables) == k:
                break
        if not tables:
            for d, key in ranked:  # coarser second pass: nearest game reachable with relaxed mapping
                if max_distance is not None and d > max_distance:
                    break
                strategy = self.store.load(key)
                table = translate_node(target, history, strategy, relaxed=True) if strategy else None
                if table is not None:
                    return table, d
            return None
        w = np.asarray(weights) / sum(weights)
        return sum(wi * t for wi, t in zip(w, tables)), used[0]


def approximate_profile(library: Library, target: PreflopGame, target_tree: dict[str, list[str]], k: int = 1,
                        exclude: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-decision approximate strategy for every node of ``target_tree``.

    Returns the profile and summary statistics: the whole-table distance to
    the nearest library game and the distribution of per-decision distances.
    """
    profile, dists, missing = {}, [], 0
    for history in target_tree:
        found = library.node_table(target, history.split("-") if history else [], k=k, exclude=exclude)
        if found is None:
            missing += 1
            continue
        profile[history], d = found
        dists.append(d)
    near = library.neighbors(target.stacks_bb(), k=1, exclude=exclude)
    d = np.asarray(dists) if dists else np.zeros(1)
    first_round = [library.node_table(target, h.split("-") if h else [], k=1, exclude=exclude)
                   for h in target_tree if h.count("-") < target.n - 1 and set(h.split("-")) <= {"f", ""}]
    return profile, {
        "distance": round(near[0].distance, 4) if near else None,
        "node_distance_mean": round(float(d.mean()), 4),
        "node_distance_p90": round(float(np.quantile(d, 0.9)), 4),
        "node_distance_max": round(float(d.max()), 4),
        "first_in_distance_max": round(max((f[1] for f in first_round if f), default=0.0), 4),
        "untranslated_nodes": missing,
    }


# --- calibration ------------------------------------------------------------------------------


@dataclass
class Calibration:
    """Where approximations were measured to be good enough."""

    family: str
    version: str
    k: int
    stat: str
    threshold_bb100: float  # largest acceptable exact NashConv of the approximate profile
    radius: float  # largest matching distance allowed
    coverage: float  # share of measured test games within the radius
    measurements: int
    within_radius_max_bb100: float | None
    note: str = ""

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> Calibration | None:
        return cls(**json.loads(path.read_text())) if path.exists() else None


def calibrate(results: list[dict[str, Any]], family: str, k: int, threshold_bb100: float,
              quantile: float = 0.9, stat: str = "node_distance_p90") -> Calibration:
    """Largest value r of the matching statistic ``stat`` such that, among
    measured games with stat <= r, at least ``quantile`` of them have an exact
    NashConv <= threshold and none exceeds twice the threshold."""
    rows = sorted((r[stat], r["nash_conv_bb100"]) for r in results if r["k"] == k)
    radius, best = 0.0, None
    for i in range(len(rows)):
        within = [nc for _, nc in rows[: i + 1]]
        ok = np.mean([nc <= threshold_bb100 for nc in within]) >= quantile and max(within) <= 2 * threshold_bb100
        if ok:
            radius, best = rows[i][0], max(within)
    coverage = float(np.mean([d <= radius for d, _ in rows])) if rows and radius > 0 else 0.0
    return Calibration(family, APPROXIMATION_VERSION, k, stat, threshold_bb100, radius, coverage, len(rows),
                       best, note=f"{quantile:.0%} of measured games within the radius meet the threshold")
