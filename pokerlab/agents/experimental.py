"""The agent under study. All of its decisions come from a pluggable ``Strategy``."""

from __future__ import annotations

from typing import Any

from pokerlab.agents.base import Agent
from pokerlab.engine.actions import Action
from pokerlab.engine.observations import Observation, PublicHandRecord
from pokerlab.strategies.base import load_strategy


class ExperimentalAgent(Agent):
    archetype = "Experimental"

    def __init__(
        self,
        agent_id: str | None = None,
        seed: int | str = 0,
        strategy: str = "equity",
        strategy_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_id, seed, strategy=strategy, strategy_params=strategy_params)
        self.strategy = load_strategy(strategy, seed, **(strategy_params or {}))

    def reset(self, seed: int | str) -> None:
        super().reset(seed)
        self.strategy.reset(seed)

    def act(self, observation: Observation) -> Action:
        return self.strategy.decide(observation)

    def observe_hand(self, record: PublicHandRecord) -> None:
        self.strategy.observe_hand(record)
