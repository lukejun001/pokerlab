"""Strategy interface for the experimental agent.

A ``Strategy`` turns an ``Observation`` into an ``Action``. The experimental
agent delegates every decision to one, so a lookup-table, CFR, solver-backed,
search-based or neural strategy plugs in without touching the engine or the
batch controller: implement ``decide`` and name the class in the config::

    experimental:
      strategy: mypackage.cfr:CFRStrategy
      strategy_params: {table_path: tables/cfr_100bb.npz}

Native (C++/Rust) solvers would be wrapped behind a ``Strategy`` subclass
through their Python bindings.
"""

from __future__ import annotations

import importlib
import random
from abc import ABC, abstractmethod
from typing import Any

from pokerlab.engine.actions import Action
from pokerlab.engine.observations import Observation, PublicHandRecord


class Strategy(ABC):
    name = "strategy"

    def __init__(self, seed: int | str = 0, **params: Any) -> None:
        self.params = params
        self.rng = random.Random(seed)

    def reset(self, seed: int | str) -> None:
        self.rng.seed(seed)

    @abstractmethod
    def decide(self, observation: Observation) -> Action: ...

    def observe_hand(self, record: PublicHandRecord) -> None:
        """Public result of each finished hand, for opponent modeling."""


class AgentStrategy(Strategy):
    """Runs any agent as a strategy, e.g. ``archetype:TAG`` as a control group."""

    def __init__(self, agent, seed: int | str = 0) -> None:
        super().__init__(seed)
        self.agent = agent
        self.name = f"archetype:{agent.archetype}"

    def reset(self, seed: int | str) -> None:
        self.agent.reset(seed)

    def decide(self, observation: Observation) -> Action:
        return self.agent.act(observation)

    def observe_hand(self, record: PublicHandRecord) -> None:
        self.agent.observe_hand(record)


def load_strategy(spec: str, seed: int | str = 0, **params: Any) -> Strategy:
    """Build a strategy from ``equity`` (built-in default), ``solver``
    (precomputed preflop strategies, see ``strategies/solver.py``),
    ``archetype:<Name>`` or an import path ``package.module:ClassName``."""
    if spec in ("equity", "heuristic", "default"):
        from pokerlab.strategies.heuristic import EquityStrategy

        return EquityStrategy(seed=seed, **params)
    if spec == "solver":
        from pokerlab.strategies.solver import SolverStrategy

        return SolverStrategy(seed=seed, **params)
    if spec.startswith("archetype:"):
        from pokerlab.agents.opponents import ARCHETYPES

        name = spec.split(":", 1)[1]
        if name not in ARCHETYPES:
            raise ValueError(f"unknown archetype {name!r}; choose from {sorted(ARCHETYPES)}")
        return AgentStrategy(ARCHETYPES[name](agent_id=name, seed=seed, **params), seed)
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"strategy {spec!r} must be 'equity', 'archetype:<Name>' or 'module:Class'")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(f"{spec} is not a Strategy subclass")
    return cls(seed=seed, **params)
