"""What an agent is allowed to see.

Everything here is built from public information plus the acting player's own
hole cards. Observations are immutable snapshots: they hold plain ints, tuples
and frozen dataclasses, never a reference to the PokerKit ``State``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from pokerlab.engine.actions import ActionType


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


_MIDDLE_LABELS = ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO")


def position_labels(n: int) -> tuple[str, ...]:
    """Labels in PokerKit player order (index 0 acts first postflop).

    9 players -> SB BB UTG UTG1 UTG2 LJ HJ CO BTN; heads-up -> BB BTN.
    """
    if n == 2:
        return ("BB", "BTN")
    middle = _MIDDLE_LABELS[len(_MIDDLE_LABELS) - (n - 3):] if n > 3 else ()
    return ("SB", "BB", *middle, "BTN")


def position_group(label: str) -> str:
    if label in ("SB", "BB"):
        return "blind"
    if label in ("CO", "BTN"):
        return "late"
    if label in ("LJ", "HJ"):
        return "middle"
    return "early"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    seat: int
    street: Street
    type: ActionType
    amount: int  # chips added for CALL; raise-to total for BET/RAISE; 0 otherwise
    all_in: bool = False


@dataclass(frozen=True, slots=True)
class GameInfo:
    num_seats: int
    small_blind: int
    big_blind: int
    starting_stack: int
    rake_percentage: float
    rake_cap: int | None
    no_flop_no_drop: bool
    mode: str


@dataclass(frozen=True, slots=True)
class PublicHandRecord:
    """Public summary of a finished hand. Only cards actually shown at
    showdown appear in ``shown_cards``."""

    hand_id: int
    button_seat: int
    seats: tuple[int, ...]
    board: tuple[int, ...]
    actions: tuple[ActionRecord, ...]
    shown_cards: dict[int, tuple[int, ...]]
    net: dict[int, int]
    rake: int


@dataclass(frozen=True, slots=True)
class PlayerPublicStats:
    """Opponent tendencies derived only from public actions."""

    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    postflop_aggressive: int = 0
    postflop_calls: int = 0
    postflop_folds: int = 0
    postflop_checks: int = 0
    saw_flop: int = 0
    showdowns: int = 0

    @property
    def vpip(self) -> float:
        return self.vpip_hands / self.hands if self.hands else 0.0

    @property
    def pfr(self) -> float:
        return self.pfr_hands / self.hands if self.hands else 0.0

    @property
    def aggression_frequency(self) -> float:
        total = self.postflop_aggressive + self.postflop_calls + self.postflop_folds + self.postflop_checks
        return self.postflop_aggressive / total if total else 0.0

    @property
    def fold_frequency(self) -> float:
        facing = self.postflop_calls + self.postflop_folds + self.postflop_aggressive
        return self.postflop_folds / facing if facing else 0.0


@dataclass(frozen=True, slots=True)
class Observation:
    hand_id: int
    seat: int
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    street: Street
    pot: int  # every chip committed this hand, including current-street bets
    stack: int
    stacks: tuple[int, ...]  # by seat; 0 for empty / busted seats
    bets: tuple[int, ...]  # current-street bets by seat
    in_hand: tuple[bool, ...]  # dealt in and not folded
    all_in: tuple[bool, ...]
    button_seat: int
    position: str
    position_index: int  # 0 = first to act postflop .. num_players-1 = last
    num_players: int  # players dealt into this hand
    legal_actions: frozenset[ActionType]
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    history: tuple[ActionRecord, ...]
    seat_positions: dict[int, str]
    game: GameInfo
    previous_hands: tuple[PublicHandRecord, ...] = ()
    player_stats: dict[int, PlayerPublicStats] = field(default_factory=dict)

    @property
    def big_blind(self) -> int:
        return self.game.big_blind

    @property
    def current_bet(self) -> int:
        return max(self.bets)

    @property
    def pot_odds(self) -> float:
        return self.to_call / (self.pot + self.to_call) if self.to_call else 0.0

    @property
    def opponents_in_hand(self) -> int:
        return sum(self.in_hand) - 1

    @property
    def position_factor(self) -> float:
        """0.0 = first to act postflop, 1.0 = button."""
        return self.position_index / (self.num_players - 1)

    @property
    def street_actions(self) -> tuple[ActionRecord, ...]:
        return tuple(a for a in self.history if a.street == self.street)

    def can(self, action_type: ActionType) -> bool:
        return action_type in self.legal_actions
