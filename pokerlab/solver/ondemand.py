"""Solve unsolved stack vectors on demand.

``solve_on_demand`` solves a stack vector the store does not have and stores
the result as an ordinary exact strategy (which also joins the library).

Warm starts from the library (the per-decision approximation of
``generalize.approximate_profile`` handed to GTOpen as a starting profile)
were measured in ``docs/phase3_report.md`` ("Solving on demand"): they win
only for short budgets. On a held-out vector the borrowed profile was
12.8 bb/100; warm runs reached 13.7 bb/100 at 20 iterations against 57.8 cold,
but cold DCFR overtook every warm setting by about 70 iterations and ended
at 0.50 bb/100 after 200 against 1.32 for the best warm setting. So
``warm="auto"`` warm-starts only budgets of at most ``WARM_MAX_ITERATIONS``.

Solves take minutes, so an agent at the table cannot wait for one. It can
record the game in a pending queue (``PendingSolves``) and keep falling back;
``pokerlab strategies solve-pending`` works through the queue later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pokerlab.engine.rake import RakeConfig
from pokerlab.solver import gtopen
from pokerlab.solver.config import SolverConfig, SolveSettings
from pokerlab.solver.game import BettingAbstraction, PreflopGame
from pokerlab.solver.generalize import Library, approximate_profile
from pokerlab.solver.store import StrategyStore, strategy_key

ON_DEMAND_VERSION = "ondemand-1"
DEFAULT_ITERATIONS = 200
# Warm-start knobs (gtopen.WarmStart; tuned in reports/phase3/warm/): best of
# the settings measured at 60+ iterations.
WARM_DEFAULTS = {"regret_scale": 10.0, "sum_scale": 30.0, "iterations": 60}
# Above this budget a cold start converged faster in every measurement.
WARM_MAX_ITERATIONS = 60


def game_to_dict(game: PreflopGame) -> dict[str, Any]:
    r = game.rake
    return {"stacks": list(game.stacks), "small_blind": game.small_blind, "big_blind": game.big_blind,
            "rake": {"percentage": r.percentage, "cap": r.cap, "no_flop_no_drop": r.no_flop_no_drop,
                     "cap_by_players": {str(k): v for k, v in r.cap_by_players.items()}},
            "abstraction": game.abstraction.to_dict()}


def game_from_dict(data: dict[str, Any]) -> PreflopGame:
    ab = dict(data["abstraction"])
    abstraction = (gtopen.GTOpenAbstraction.from_dict({k: v for k, v in ab.items() if k != "engine"})
                   if ab.get("engine") == "gtopen" else BettingAbstraction.from_dict(ab))
    return PreflopGame(tuple(data["stacks"]), data["small_blind"], data["big_blind"],
                       RakeConfig.from_dict(data["rake"]), abstraction)


def warm_profile(store: StrategyStore, game: PreflopGame, k: int = 1) -> tuple[dict | None, dict[str, Any]]:
    """The library approximation of ``game`` to start from, and where it came from."""
    library = Library(store, game)
    if not len(library):
        return None, {"library_size": 0}
    t0 = time.perf_counter()
    profile, info = approximate_profile(library, game, gtopen.tree(game), k=k)
    nearest = library.neighbors(game.stacks_bb(), k=1)[0]
    return profile, {"library_size": len(library), "nearest_key": nearest.key, "nearest_stacks_bb": nearest.stacks_bb,
                     "k": k, "approximation_s": round(time.perf_counter() - t0, 1), **info}


def solve_on_demand(store: StrategyStore, game: PreflopGame, iterations: int = DEFAULT_ITERATIONS,
                    threads: int | None = None, warm: str = "auto", warm_knobs: dict[str, Any] | None = None,
                    k: int = 1, progress: bool = False, name: str | None = None) -> dict[str, Any]:
    """Solve ``game`` (GTOpen abstractions only) and store it. Returns the solve
    metrics; an already-stored game is returned as is (``"cached": True``).

    ``warm``: ``"auto"`` warm-starts from the library only when ``iterations``
    is at most ``WARM_MAX_ITERATIONS`` (and the library has same-family games);
    ``"on"`` / ``"off"`` force it."""
    if warm not in ("auto", "on", "off"):
        raise ValueError("warm must be auto, on or off")
    if not isinstance(game.abstraction, gtopen.GTOpenAbstraction):
        raise ValueError("on-demand solving needs a GTOpen abstraction")
    existing = store.lookup(game)
    if existing is not None:
        return {**existing.strategy.meta["metrics"], "key": existing.strategy.key, "cached": True}
    use_warm = warm == "on" or (warm == "auto" and iterations <= WARM_MAX_ITERATIONS)
    profile, source = warm_profile(store, game, k=k) if use_warm else (None, {})
    knobs = {**WARM_DEFAULTS, **(warm_knobs or {})}
    start = gtopen.WarmStart(profile, **knobs) if profile else None
    key = strategy_key(game)
    config = SolverConfig(name=name or f"ondemand_{key[:12]}", game=game, chips_per_bb=game.big_blind,
                          solve=SolveSettings(iterations=iterations, eval_every=iterations))
    extra = {"on_demand": {"version": ON_DEMAND_VERSION, "warm_started": start is not None,
                           "warm_knobs": knobs if start else None, **source}}
    return gtopen.generate(config, store, iterations=iterations, threads=threads, progress=progress,
                           warm_start=start, extra_metrics=extra)


class PendingSolves:
    """Games an agent met without a stored strategy, waiting to be solved.

    An append-only JSON-lines file next to the store (safe for several
    simulation workers appending at once); duplicates collapse on read.
    """

    def __init__(self, store: StrategyStore) -> None:
        self.store = store
        self.path = Path(store.root) / "pending.jsonl"

    def add(self, game: PreflopGame) -> None:
        line = json.dumps({"key": strategy_key(game), "stacks_bb": list(game.stacks_bb()), "game": game_to_dict(game)})
        with self.path.open("a") as f:
            f.write(line + "\n")

    def items(self) -> list[dict[str, Any]]:
        """Unsolved pending games, first-seen order, with how often each was met."""
        if not self.path.exists():
            return []
        seen: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in seen:
                seen[row["key"]]["count"] += 1
            else:
                seen[row["key"]] = {**row, "count": 1}
        return [row for row in seen.values() if not self.store.has(row["key"])]

    def compact(self) -> None:
        """Drop solved and duplicate entries."""
        rows = self.items()
        self.path.write_text("".join(json.dumps({k: r[k] for k in ("key", "stacks_bb", "game")}) + "\n" for r in rows))
