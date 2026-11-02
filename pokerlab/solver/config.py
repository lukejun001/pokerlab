"""Solver configuration files (``configs/6max_*.yaml``).

A solver config describes one concrete game (players, blinds, an independent
starting stack per position, rake), the betting abstraction, the solve
budget and, optionally, an offline simulation that uses the result::

    game:
      variant: no_limit_texas_holdem
      max_players: 6
      active_players: 6
      small_blind_bb: 0.5
      big_blind_bb: 1.0
      ante_bb: 0
      chips_per_bb: 5000            # integer chips; 5000 = cents at $25/$50
      starting_stacks_bb: {UTG: 100, HJ: 150, CO: 70, BTN: 115, SB: 40, BB: 200}
    rake: {percentage: 0.045, cap_bb: 0.06, no_flop_no_drop: true, cap_basis: players_dealt_in}
    strategy: {model: approximate_equilibrium, preflop_solver: pokerlab-cfr, ...}
    abstraction: {...}               # BettingAbstraction fields
    solve: {iterations: 1500, batch_size: 4096, workers: 8, seed: 1, ...}
    simulation: {hands: 10000, seed: 42, lineup: [...]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pokerlab.engine.rake import RakeConfig
from pokerlab.simulation.configurations import CONFIG_DIR, resolve_config_path
from pokerlab.solver.game import GAME_VARIANT, BettingAbstraction, PreflopGame, position_names

STRATEGY_MODEL_VERSION = "approx-eq-preflop-1"
SUPPORTED_SOLVERS = ("pokerlab-cfr",)


@dataclass
class SolveSettings:
    iterations: int = 1000
    batch_size: int = 4096
    workers: int = 8
    seed: int = 1
    eval_every: int = 0  # 0 = evaluate only at the end
    eval_deals: int = 30_000
    time_limit_s: float | None = None


@dataclass
class SolverConfig:
    name: str
    game: PreflopGame
    chips_per_bb: int
    solve: SolveSettings = field(default_factory=SolveSettings)
    simulation: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def stacks_by_position_bb(self) -> dict[str, float]:
        return dict(zip(self.game.positions, self.game.stacks_bb()))


def _rake(data: dict[str, Any], chips_per_bb: int) -> RakeConfig:
    data = dict(data or {})
    basis = data.pop("cap_basis", "players_dealt_in")
    if basis != "players_dealt_in":
        raise ValueError("only cap_basis: players_dealt_in is supported")
    cap_bb = data.pop("cap_bb", None)
    caps_bb = data.pop("cap_by_players_bb", None) or {}
    cap = None if cap_bb is None else int(round(cap_bb * chips_per_bb))
    by_players = {int(k): int(round(v * chips_per_bb)) for k, v in caps_bb.items()}
    unknown = set(data) - {"percentage", "no_flop_no_drop"}
    if unknown:
        raise ValueError(f"unknown rake keys: {sorted(unknown)}")
    return RakeConfig(percentage=float(data.get("percentage", 0.0)), cap=cap, cap_by_players=by_players,
                      no_flop_no_drop=bool(data.get("no_flop_no_drop", True)))


def load_solver_config(name_or_path: str | Path) -> SolverConfig:
    path = resolve_config_path(str(name_or_path))
    data = yaml.safe_load(path.read_text()) or {}
    return solver_config_from_dict(data, name=data.get("name", path.stem), path=path)


def solver_config_from_dict(data: dict[str, Any], name: str = "solver", path: Path | None = None) -> SolverConfig:
    g = data["game"]
    if g.get("variant", GAME_VARIANT) != GAME_VARIANT:
        raise ValueError(f"unsupported variant {g.get('variant')!r}")
    n = int(g.get("active_players", g.get("max_players", 6)))
    chips_per_bb = int(g.get("chips_per_bb", 100))
    if float(g.get("big_blind_bb", 1.0)) != 1.0:
        raise ValueError("big_blind_bb must be 1.0 (stacks and sizes are measured in big blinds)")
    sb = int(round(float(g.get("small_blind_bb", 0.5)) * chips_per_bb))
    ante = int(round(float(g.get("ante_bb", 0)) * chips_per_bb))
    names = position_names(n)
    stacks_cfg = g["starting_stacks_bb"]
    if isinstance(stacks_cfg, dict):
        missing = set(names) - set(stacks_cfg)
        extra = set(stacks_cfg) - set(names)
        if missing or extra:
            raise ValueError(f"starting_stacks_bb must name exactly {names}; missing {sorted(missing)}, extra {sorted(extra)}")
        stacks_bb = [float(stacks_cfg[p]) for p in names]
    else:
        stacks_bb = [float(s) for s in stacks_cfg]
        if len(stacks_bb) != n:
            raise ValueError(f"need {n} stacks, got {len(stacks_bb)}")
    strategy = dict(data.get("strategy") or {})
    solver_name = strategy.get("preflop_solver", "pokerlab-cfr")
    if solver_name == "gtopen":
        raise ValueError("preflop_solver: gtopen is not usable here: GTOpen has no license (all rights reserved) "
                         "and supports only equal stacks. Use pokerlab-cfr; see docs/phase3_report.md.")
    if solver_name not in SUPPORTED_SOLVERS:
        raise ValueError(f"preflop_solver must be one of {SUPPORTED_SOLVERS}")
    if strategy.get("opponent_specific_exploitation"):
        raise ValueError("opponent_specific_exploitation is not supported by the equilibrium model")
    if strategy.get("realtime_solving"):
        raise ValueError("realtime_solving is not supported; strategies are precomputed")
    game = PreflopGame(
        stacks=tuple(int(round(s * chips_per_bb)) for s in stacks_bb),
        small_blind=sb,
        big_blind=chips_per_bb,
        rake=_rake(data.get("rake") or {}, chips_per_bb),
        abstraction=BettingAbstraction.from_dict(data.get("abstraction")),
        ante=ante,
    )
    solve = SolveSettings(**(data.get("solve") or {}))
    return SolverConfig(name=name, game=game, chips_per_bb=chips_per_bb, solve=solve,
                        simulation=dict(data.get("simulation") or {}), strategy=strategy, path=path)


def is_solver_config(name_or_path: str) -> bool:
    try:
        path = resolve_config_path(name_or_path)
    except FileNotFoundError:
        return False
    data = yaml.safe_load(path.read_text()) or {}
    return isinstance(data, dict) and "game" in data


__all__ = ["CONFIG_DIR", "STRATEGY_MODEL_VERSION", "SolverConfig", "SolveSettings", "is_solver_config",
           "load_solver_config", "solver_config_from_dict"]
