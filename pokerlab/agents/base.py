"""The interface every player implements."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from pokerlab.engine.actions import Action
from pokerlab.engine.observations import Observation, PublicHandRecord


class Agent(ABC):
    """A poker player.

    The engine calls ``act`` whenever it is this agent's turn and
    ``observe_hand`` with the public record of every finished hand. Agents get
    no other access to the game: ``Observation`` is all they can see.

    ``agent_id`` names this seat's player in reports (e.g. ``TAG_2``);
    ``archetype`` groups agents of the same kind.
    """

    archetype: str = "Agent"

    def __init__(self, agent_id: str | None = None, seed: int | str = 0, **params: Any) -> None:
        self.agent_id = agent_id or self.archetype
        self.params = params
        self.rng = random.Random(seed)

    def reset(self, seed: int | str) -> None:
        """Called at the start of every independent simulation chunk."""
        self.rng.seed(seed)

    @abstractmethod
    def act(self, observation: Observation) -> Action: ...

    def observe_hand(self, record: PublicHandRecord) -> None:
        """Public summary of a finished hand (shown cards only)."""
