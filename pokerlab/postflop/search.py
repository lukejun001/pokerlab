"""Heads-up postflop search: re-solve each street in real time with GTOpen.

At the start of each street the two players' ranges (per combo) are known:
on the flop from the preflop strategy's reach along the preflop line, on
later streets from the previous street's solve (each player's reach at the
chance node that closed the line actually played). The street is then solved
within a time budget by the fork's ``postflop-cli`` and every node of the
street is exported, so the whole street plays from one solve.

The flop is depth-limited (``Spot::new_depth_limited`` in the fork): betting
ends with the flop and each closing line is valued at an equity leaf, the
check-down value over all runouts. That solves a richer flop tree to
convergence in about a second, but the leaf ignores later betting; it is the
slot a learned value function would fill. Turn and river trees are full.
Exploitability is reported for each solve in its own tree (the flop number
therefore ignores later-street play).

Opponent bet sizes are translated to the nearest size in the tree (log
distance of the street-total amount); all-ins map to the all-in action.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pokerlab.engine.cards import card_str
from pokerlab.solver.gtopen import GTOPEN_DIR
from pokerlab.solver.hands import CLASS_LABELS

BINARY = GTOPEN_DIR / "target" / "release" / "postflop-cli"
STREETS = ("flop", "turn", "river")


@dataclass
class SearchSettings:
    flop_bet: tuple[float, ...] = (33.0, 75.0)
    flop_raise: tuple[float, ...] = (60.0,)
    turn_bet: tuple[float, ...] = (75.0,)
    turn_raise: tuple[float, ...] = (60.0,)
    river_bet: tuple[float, ...] = (75.0,)
    river_raise: tuple[float, ...] = (60.0,)
    max_raises: int = 3
    budget_s: dict[str, float] = field(default_factory=lambda: {"flop": 10.0, "turn": 15.0, "river": 5.0})
    target_exploit_pct: float = 0.3
    threads: int = 2
    flop_depth_limited: bool = True  # equity leaves after the flop (else the full three-street tree)


def class_combos() -> list[list[tuple[int, int]]]:
    """Card pairs (pokerlab ids) of each of the 169 classes."""
    ranks = "23456789TJQKA"
    out = []
    for label in CLASS_LABELS:
        hi, lo = ranks.index(label[0]), ranks.index(label[1])
        pairs = []
        for s1 in range(4):
            for s2 in range(4):
                a, b = hi * 4 + s1, lo * 4 + s2
                if label[0] == label[1]:
                    if s1 < s2:
                        pairs.append((a, b))
                elif label.endswith("s"):
                    if s1 == s2:
                        pairs.append((a, b))
                elif s1 != s2:
                    pairs.append((a, b))
        out.append(pairs)
    return out


CLASS_COMBOS = class_combos()


def combo_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a > b else (b, a)


def range_from_classes(reach: np.ndarray, board: tuple[int, ...]) -> dict[tuple[int, int], float]:
    """Per-class reach (169) -> per-combo weights, without combos that hit the board."""
    dead = set(board)
    out = {}
    for k, pairs in enumerate(CLASS_COMBOS):
        if reach[k] <= 0:
            continue
        for a, b in pairs:
            if a not in dead and b not in dead:
                out[combo_key(a, b)] = float(reach[k])
    return out


def range_string(weights: dict[tuple[int, int], float], floor: float = 1e-4) -> str:
    top = max(weights.values(), default=0.0)
    if top <= 0:
        return ""
    return ",".join(f"{card_str(a)}{card_str(b)}:{w / top:.5f}" for (a, b), w in weights.items() if w / top >= floor)


def _sizing(bet, raise_) -> dict[str, Any]:
    return {"bet": [{"PotPct": float(x)} for x in bet], "raise": [{"PotPct": float(x)} for x in raise_], "donk": []}


@dataclass
class StreetSolve:
    street: str
    board: tuple[int, ...]
    views: dict[tuple[int, ...], dict]
    metrics: dict[str, Any]

    def view(self, path: tuple[int, ...]) -> dict | None:
        return self.views.get(path)

    def reach(self, path: tuple[int, ...], player: int) -> dict[tuple[int, int], float]:
        v = self.views[path]
        return {combo_key(h["c1"], h["c2"]): h["reach"] for h in v["players"][player]["hands"] if h["reach"] > 0}


def solve_street(street: str, board: tuple[int, ...], ranges: tuple[dict, dict], pot_bb: float, stack_bb: float,
                 rake_pct: float, rake_cap_bb: float, settings: SearchSettings) -> StreetSolve:
    """Solve the street starting now (ranges: OOP, IP) and export all its nodes."""
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} not found: cargo build --release -p solver --bin postflop-cli")
    s = settings
    sizes = {"flop": _sizing(s.flop_bet, s.flop_raise), "turn": _sizing(s.turn_bet, s.turn_raise),
             "river": _sizing(s.river_bet, s.river_raise)}
    per_street = [sizes["flop"], sizes["turn"], sizes["river"]]
    depth_limited = street == "flop" and s.flop_depth_limited
    spot = {
        "board": "".join(card_str(c) for c in board),
        "range_oop": range_string(ranges[0]), "range_ip": range_string(ranges[1]),
        "tree": {"starting_pot": pot_bb, "effective_stack": stack_bb, "rake_pct": rake_pct,
                 "rake_cap": rake_cap_bb, "oop": per_street, "ip": per_street,
                 "allin_threshold": 0.85, "add_allin": True, "max_raises": s.max_raises},
    }
    request = {"spot": spot, "max_iterations": 1000, "time_limit_s": s.budget_s[street],
               "target_exploit_pct": s.target_exploit_pct, "check_every": 10, "export_street": True,
               "depth_limited": depth_limited}
    env = dict(os.environ, SOLVER_THREADS=str(s.threads))
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(request))
        proc = subprocess.run([str(BINARY), str(src), str(dst)], cwd=GTOPEN_DIR, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"postflop-cli failed: {proc.stderr[-1500:]}")
        out = json.loads(dst.read_text())
    views = {tuple(v["path"]): v for v in out["street_views"]}
    return StreetSolve(street, board, views, out["metrics"])


def match_action(view: dict, kind: str, to_bb: float | None, all_in: bool) -> int:
    """Index of the tree action playing the role of an observed action.

    ``kind``: fold/check/call/bet/raise; ``to_bb``: street-total amount for bets and raises.
    """
    actions = view["actions"]
    if kind in ("fold", "check", "call"):
        for i, a in enumerate(actions):
            if a["kind"] == kind:
                return i
        alt = {"check": "call", "call": "check", "fold": "check"}[kind]
        for i, a in enumerate(actions):
            if a["kind"] == alt:
                return i
        return 0
    aggressive = [(i, a) for i, a in enumerate(actions) if a["kind"] in ("bet", "raise")]
    if not aggressive:
        return next((i for i, a in enumerate(actions) if a["kind"] in ("call", "check")), 0)
    if all_in:
        return max(aggressive, key=lambda x: x[1]["amount"])[0]
    return min(aggressive, key=lambda x: abs(math.log(max(x[1]["amount"], 1e-6) / max(to_bb or 1e-6, 1e-6))))[0]


def hand_strategy(view: dict, player: int, hole: tuple[int, int]) -> np.ndarray | None:
    key = combo_key(*hole)
    for h in view["players"][player]["hands"]:
        if combo_key(h["c1"], h["c2"]) == key:
            return np.asarray(h["strategy"]) if h.get("strategy") is not None else None
    return None
