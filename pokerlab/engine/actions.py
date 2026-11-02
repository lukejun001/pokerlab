"""Agent actions and their validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class Action:
    """An agent decision. ``amount`` is the *raise-to* total for this street
    (PokerKit semantics) and is required only for BET and RAISE."""

    type: ActionType
    amount: int | None = None

    @staticmethod
    def fold() -> Action:
        return Action(ActionType.FOLD)

    @staticmethod
    def check() -> Action:
        return Action(ActionType.CHECK)

    @staticmethod
    def call() -> Action:
        return Action(ActionType.CALL)

    @staticmethod
    def bet(amount: int) -> Action:
        return Action(ActionType.BET, int(amount))

    @staticmethod
    def raise_to(amount: int) -> Action:
        return Action(ActionType.RAISE, int(amount))


class InvalidActionError(ValueError):
    pass


def validate(action: object, legal: frozenset[ActionType], min_to: int | None, max_to: int | None) -> Action:
    """Return ``action`` if it is legal in the current spot, else raise."""
    if not isinstance(action, Action):
        raise InvalidActionError(f"agent returned {type(action).__name__}, expected Action")
    if action.type not in legal:
        raise InvalidActionError(f"{action.type.value} is not legal here (legal: {sorted(a.value for a in legal)})")
    if action.type in (ActionType.BET, ActionType.RAISE):
        if not isinstance(action.amount, int) or isinstance(action.amount, bool):
            raise InvalidActionError(f"{action.type.value} requires an integer amount")
        if not min_to <= action.amount <= max_to:
            raise InvalidActionError(
                f"{action.type.value} to {action.amount} outside legal range [{min_to}, {max_to}]"
            )
    elif action.amount is not None:
        raise InvalidActionError(f"{action.type.value} must not carry an amount")
    return action
