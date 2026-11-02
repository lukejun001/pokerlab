"""Agent registry: build any agent from its config name."""

from __future__ import annotations

import importlib
from typing import Any

from pokerlab.agents.base import Agent
from pokerlab.agents.experimental import ExperimentalAgent
from pokerlab.agents.opponents import ARCHETYPES

AGENT_TYPES: dict[str, type[Agent]] = {**ARCHETYPES, "Experimental": ExperimentalAgent}


def create_agent(kind: str, agent_id: str, seed: int | str = 0, params: dict[str, Any] | None = None) -> Agent:
    """``kind`` is a registered name (``TAG``, ``Experimental``...) or ``module:Class``."""
    if kind in AGENT_TYPES:
        cls = AGENT_TYPES[kind]
    elif ":" in kind:
        module_name, class_name = kind.split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
    else:
        raise ValueError(f"unknown agent type {kind!r}; choose from {sorted(AGENT_TYPES)} or module:Class")
    return cls(agent_id=agent_id, seed=seed, **(params or {}))


__all__ = ["Agent", "AGENT_TYPES", "create_agent", "ExperimentalAgent"]
