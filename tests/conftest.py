"""Shared helpers: scripted agents and rigged decks."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import pytest

from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.cards import parse_cards
from pokerlab.engine.game import play_hand
from pokerlab.engine.observations import GameInfo, Observation
from pokerlab.engine.rake import RakeConfig, make_rake_function


class RiggedDeck:
    """Stands in for ``random.Random``: puts chosen cards on top of the deck.

    Deal order: two hole cards per player in PokerKit order (left of the button
    first, button last; heads-up: big blind first), then burn, flop x3, burn,
    turn, burn, river.
    """

    def __init__(self, cards: str) -> None:
        self.top = list(parse_cards(cards))
        if len(set(self.top)) != len(self.top):
            raise ValueError(f"rigged deck repeats a card: {cards}")

    def shuffle(self, deck: list[int]) -> None:
        rest = [c for c in deck if c not in self.top]
        deck[:] = self.top + rest


def rigged(holes: Sequence[str], board: str = "", burns: str = "9s8s7s") -> RiggedDeck:
    """Hole cards in PokerKit order plus a 5-card board, with burn cards interleaved."""
    b = parse_cards(board) if board else ()
    burn = parse_cards(burns)
    from pokerlab.engine.cards import cards_str

    sequence = "".join(holes)
    if b:
        sequence += cards_str((burn[0], *b[:3], burn[1], b[3], burn[2], b[4]))
    return RiggedDeck(sequence)


class Scripted:
    """Agent driven by a function ``(obs) -> Action``; records observations."""

    def __init__(self, policy: Callable[[Observation], Action]) -> None:
        self.policy = policy
        self.observations: list[Observation] = []

    def act(self, obs: Observation) -> Action:
        self.observations.append(obs)
        return self.policy(obs)

    def observe_hand(self, record) -> None:
        pass


def passive(obs: Observation) -> Action:
    return Action.check() if obs.can(ActionType.CHECK) else Action.call()


def folder(obs: Observation) -> Action:
    return Action.check() if obs.can(ActionType.CHECK) else Action.fold()


def jammer(obs: Observation) -> Action:
    for kind in (ActionType.BET, ActionType.RAISE):
        if obs.can(kind):
            return Action(kind, obs.max_raise_to)
    return passive(obs)


def random_policy(rng: random.Random) -> Callable[[Observation], Action]:
    def policy(obs: Observation) -> Action:
        x = rng.random()
        for kind in (ActionType.BET, ActionType.RAISE):
            if obs.can(kind) and x < 0.2:
                if rng.random() < 0.3:
                    return Action(kind, obs.max_raise_to)
                return Action(kind, rng.randint(obs.min_raise_to, obs.max_raise_to))
        if obs.can(ActionType.FOLD) and x < 0.45:
            return Action.fold()
        return passive(obs)

    return policy


def game_info(players: int, rake: RakeConfig | None = None, stack: int = 5000) -> GameInfo:
    rake = rake or RakeConfig()
    return GameInfo(players, 25, 50, stack, rake.percentage, rake.cap, rake.no_flop_no_drop, "independent")


def run(agents: dict, *, stacks=None, button=None, deck=None, rake: RakeConfig | None = None, **kw):
    seats = sorted(agents)
    n = max(seats) + 1
    rake = rake or RakeConfig()
    return play_hand(
        agents=agents,
        seats=seats,
        button_seat=seats[-1] if button is None else button,
        stacks=stacks or {s: 5000 for s in seats},
        game=game_info(n, rake),
        rake_function=make_rake_function(rake),
        deck_rng=deck or random.Random(0),
        **kw,
    )


@pytest.fixture
def run_hand():
    return run
