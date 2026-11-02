"""Rake rules and the PokerKit rake callback.

PokerKit calls ``rake(amount, state)`` once per main/side pot and applies any
cap *per pot*, so a hand with side pots can be raked past the cap. We hand
PokerKit our own callback instead: it computes the rake for the *whole hand*
(percentage, cap, cap schedule by players dealt, no-flop-no-drop) and gives
each pot its proportional share, rounded down. PokerKit then records each
pot's ``raked_amount`` and pays out only the unraked remainder, so rake comes
straight out of the winners' chips.

Because each pot's share is rounded down, the chips actually raked can be up
to ``(number of pots - 1)`` below the nominal hand rake and never above it.
The engine reports the exact amount PokerKit removed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

RakeFunction = Callable[[int, Any], tuple[int, int]]


@dataclass(frozen=True)
class RakeConfig:
    percentage: float = 0.0  # 0.04 == 4%
    cap: int | None = None  # chips; None = uncapped
    # Players dealt in -> cap, e.g. {2: 50, 3: 100, 5: 150}: the largest key
    # <= players dealt in applies. Overrides ``cap`` when it matches.
    cap_by_players: Mapping[int, int] = field(default_factory=dict)
    no_flop_no_drop: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.percentage <= 1:
            raise ValueError(f"rake percentage {self.percentage} must be within [0, 1]")
        if self.cap is not None and self.cap < 0:
            raise ValueError("rake cap must be non-negative")

    def cap_for(self, players_dealt: int) -> int | None:
        applicable = [k for k in self.cap_by_players if k <= players_dealt]
        if applicable:
            return self.cap_by_players[max(applicable)]
        return self.cap

    def hand_rake(self, total_pot: int, players_dealt: int, flop_dealt: bool) -> int:
        """Nominal rake for a hand whose pots total ``total_pot`` chips."""
        if self.percentage <= 0 or total_pot <= 0:
            return 0
        if self.no_flop_no_drop and not flop_dealt:
            return 0
        rake = int(total_pot * self.percentage + 1e-9)
        cap = self.cap_for(players_dealt)
        return rake if cap is None else min(rake, cap)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RakeConfig:
        data = dict(data or {})
        caps = {int(k): int(v) for k, v in (data.pop("cap_by_players", None) or {}).items()}
        cap = data.pop("cap", None)
        return cls(cap=None if cap is None else int(cap), cap_by_players=caps, **data)


def make_rake_function(config: RakeConfig) -> RakeFunction:
    if config.percentage <= 0:
        return lambda amount, state: (0, amount)

    def rake(amount: int, state) -> tuple[int, int]:
        # Chips already collected into pots == sum of every pot PokerKit is
        # about to build. (Uncalled bets have been refunded by this point.)
        total = 0
        for payoff, bet in zip(state.payoffs, state.bets):
            total += -payoff - bet
        flop_dealt = any(state.board_cards)
        hand_rake = config.hand_rake(total, state.player_count, flop_dealt)
        if not hand_rake:
            return 0, amount
        share = amount * hand_rake // total
        return share, amount - share

    return rake
