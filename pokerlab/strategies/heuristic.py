"""Default experimental strategy: equity-driven play with simple opponent modeling.

Preflop it uses the shared range model. Postflop it estimates Monte Carlo
equity against ranges inferred from opponents' public VPIP and preflop
actions, then adjusts to observed tendencies:

* against players who fold often it bluffs more; against calling stations it
  stops bluffing and value-bets thinner and bigger;
* against frequent bettors it calls down wider.

This is a reasonable starting point to beat, not a strong poker AI.
"""

from __future__ import annotations

from typing import Any

from pokerlab.agents.heuristics import (
    PreflopSpot,
    analyze_hand,
    bet_pot_fraction,
    check_or_call,
    check_or_fold,
    equity_vs_ranges,
    estimated_ranges,
    preflop_percentile,
    ramp,
    street_raises,
    table_aggression,
    table_fold_frequency,
    was_preflop_aggressor,
)
from pokerlab.agents.opponents import HeuristicAgent
from pokerlab.engine.actions import Action
from pokerlab.engine.observations import Observation, Street
from pokerlab.strategies.base import Strategy

PREFLOP_DEFAULTS = {
    "open_range": {"early": 0.13, "middle": 0.18, "late": 0.32, "sb": 0.25},
    "call_range": 0.08,
    "threebet_range": 0.045,
    "threebet_bluff_frequency": 0.15,
    "call_3bet_range": 0.05,
    "fourbet_range": 0.02,
    "mix_width": 0.02,
}


class _PreflopModel(HeuristicAgent):
    archetype = "PreflopModel"


class EquityStrategy(Strategy):
    name = "equity"

    def __init__(
        self,
        seed: int | str = 0,
        equity_samples: int = 200,
        bluff_frequency: float = 0.15,
        value_threshold: float = 0.6,
        exploit_strength: float = 1.0,
        preflop: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(seed)
        self.equity_samples = equity_samples
        self.bluff_frequency = bluff_frequency
        self.value_threshold = value_threshold
        self.exploit = exploit_strength
        self.ranges = _PreflopModel(seed=seed, **{**PREFLOP_DEFAULTS, **(preflop or {})})
        self.ranges.rng = self.rng  # one RNG stream for the whole strategy

    def reset(self, seed: int | str) -> None:
        self.rng.seed(seed)

    def decide(self, obs: Observation) -> Action:
        if obs.street == Street.PREFLOP:
            return self.ranges.preflop(obs, PreflopSpot.from_obs(obs), preflop_percentile(obs.hole_cards))
        return self.postflop(obs)

    def postflop(self, obs: Observation) -> Action:
        rng = self.rng
        features = analyze_hand(obs.hole_cards, obs.board)
        opponents = obs.opponents_in_hand
        equity = equity_vs_ranges(
            obs.hole_cards, obs.board, estimated_ranges(obs)[:4], self.equity_samples, rng
        )
        fold_freq = table_fold_frequency(obs)  # population prior 0.4
        aggression = table_aggression(obs)  # population prior 0.3

        if not obs.to_call:
            # Thinner value against stations, which fold less.
            threshold = self.value_threshold + 0.04 * min(opponents - 1, 3)
            threshold -= self.exploit * 0.15 * (0.4 - fold_freq)
            if rng.random() < ramp(equity, threshold, 0.04):
                size = 0.55 + 0.35 * min(1.0, (equity - threshold) * 3)
                size += self.exploit * max(0.0, 0.4 - fold_freq)  # bigger into callers
                return bet_pot_fraction(obs, size)
            if opponents <= 2 and equity < 0.4:
                bluff = self.bluff_frequency * (1 + self.exploit * 2.5 * (fold_freq - 0.4))
                bluff *= 1.5 if was_preflop_aggressor(obs) else 1.0
                bluff *= 1.5 if features.any_draw and obs.street != Street.RIVER else 1.0
                if rng.random() < max(0.0, bluff) / opponents:
                    return bet_pot_fraction(obs, 0.6)
            return check_or_fold(obs)

        raises = street_raises(obs)
        if equity >= 0.82 or (equity >= 0.7 and raises <= 1 and rng.random() < 0.6):
            return bet_pot_fraction(obs, 0.8)
        implied = 0.04 if features.any_draw and obs.street != Street.RIVER else 0.0
        bluff_catch = self.exploit * 0.12 * (aggression - 0.3)
        if rng.random() < ramp(equity + implied + bluff_catch, obs.pot_odds + 0.02 * (raises - 1), 0.03):
            return check_or_call(obs)
        return check_or_fold(obs)
