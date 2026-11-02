"""Eight heuristic opponent archetypes.

All share a parameterized preflop range model (``HeuristicAgent.preflop``) and
each has its own postflop decision logic. Every parameter in ``defaults`` can
be overridden per agent from the YAML lineup, e.g.::

    - {type: TAG, params: {aggression: 0.8, open_range: {late: 0.35}}}

The numbers are illustrative behavioral knobs, not validated frequencies. The
decisions react to hand strength, draws, position, pot odds, bet size,
number of opponents and observed opponent tendencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from pokerlab.agents.base import Agent
from pokerlab.agents.heuristics import (
    HandFeatures,
    PreflopSpot,
    all_in,
    analyze_hand,
    bet_pot_fraction,
    bet_to_pot_ratio,
    check_or_call,
    check_or_fold,
    equity_vs_ranges,
    estimated_ranges,
    heuristic_equity,
    in_position,
    open_raise,
    position_bucket,
    preflop_percentile,
    ramp,
    reraise,
    street_raises,
    table_aggression,
    table_fold_frequency,
    was_preflop_aggressor,
)
from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.observations import Observation, Street

BASE_DEFAULTS: dict[str, Any] = {
    # behavioral knobs used by postflop logic (0..1)
    "tightness": 0.5,
    "aggression": 0.5,
    "bluff_frequency": 0.1,
    "calling_tendency": 0.4,
    # preflop ranges as fractions of all starting hands (0.10 = top 10%)
    "open_range": {"early": 0.12, "middle": 0.16, "late": 0.28, "sb": 0.22},
    "open_limp_frequency": 0.0,  # share of opening hands limped instead of raised
    "limp_range": 0.0,  # extra hands (beyond open range) played by limping
    "call_range": 0.08,  # flat a single raise
    "threebet_range": 0.035,
    "threebet_bluff_frequency": 0.0,  # 3-bet hands just outside the calling range
    "call_3bet_range": 0.04,
    "fourbet_range": 0.015,
    "bb_defense": 1.8,  # big blind call-range multiplier
    "open_size_bb": 2.5,
    "threebet_multiplier": 3.2,
    "mix_width": 0.015,  # blend actions near range boundaries
}


@dataclass(slots=True)
class Spot:
    """Postflop decision context."""

    features: HandFeatures
    opponents: int
    facing_bet: bool
    bet_ratio: float  # bet faced / pot before it
    pot_odds: float
    raises: int  # bets + raises this street
    aggressor: bool  # we made the last preflop raise
    in_position: bool
    equity: float


class HeuristicAgent(Agent):
    archetype = "Heuristic"
    defaults: dict[str, Any] = {}

    def __init__(self, agent_id: str | None = None, seed: int | str = 0, **params: Any) -> None:
        merged = copy.deepcopy(BASE_DEFAULTS)
        for source in (self.defaults, params):
            for key, value in source.items():
                if key not in merged:
                    raise ValueError(f"{type(self).__name__}: unknown parameter {key!r}")
                if isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        super().__init__(agent_id, seed, **merged)
        self.p = merged

    def chance(self, probability: float) -> bool:
        return self.rng.random() < probability

    # --- dispatch ------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        if obs.street == Street.PREFLOP:
            return self.preflop(obs, PreflopSpot.from_obs(obs), preflop_percentile(obs.hole_cards))
        return self.postflop(obs, self.spot(obs))

    def spot(self, obs: Observation) -> Spot:
        features = analyze_hand(obs.hole_cards, obs.board)
        opponents = obs.opponents_in_hand
        return Spot(
            features=features,
            opponents=opponents,
            facing_bet=obs.to_call > 0,
            bet_ratio=bet_to_pot_ratio(obs),
            pot_odds=obs.pot_odds,
            raises=street_raises(obs),
            aggressor=was_preflop_aggressor(obs),
            in_position=in_position(obs),
            equity=heuristic_equity(features, opponents),
        )

    # --- preflop (shared range model) ---------------------------------------

    def in_range(self, pct: float, width: float) -> bool:
        """Is a hand at percentile ``pct`` inside a ``width`` range, mixing near the edge?"""
        return self.chance(ramp(width, pct, self.p["mix_width"]))

    def preflop(self, obs: Observation, spot: PreflopSpot, pct: float) -> Action:
        p = self.p
        bucket = position_bucket(obs)
        bb = obs.big_blind

        if spot.situation in ("unopened", "limped"):
            if bucket == "bb":
                if self.in_range(pct, p["threebet_range"] * 2):
                    return open_raise(obs, spot, p["open_size_bb"] + 1)
                return check_or_fold(obs)
            open_width = p["open_range"][bucket] * (0.75 if spot.limpers else 1.0)
            if self.in_range(pct, open_width):
                if pct > p["threebet_range"] and self.chance(p["open_limp_frequency"]):
                    return check_or_call(obs)
                return self.sized(obs, open_raise(obs, spot, p["open_size_bb"]))
            if pct < open_width + p["limp_range"] or (bucket == "sb" and self.chance(p["limp_range"])):
                return check_or_call(obs)
            return check_or_fold(obs)

        raise_bb = spot.raise_to / bb
        if spot.situation == "facing_open":
            size_factor = min(1.0, 3.0 / max(raise_bb, 1.0)) ** 0.5
            call_width = p["call_range"] * size_factor
            if bucket in ("late", "sb"):
                call_width *= 1.25
            elif bucket == "bb":
                call_width *= p["bb_defense"]
            call_width += 0.01 * spot.callers  # better price multiway
            threebet_width = p["threebet_range"] * (1.2 if bucket in ("late", "sb", "bb") else 1.0)
            if self.in_range(pct, threebet_width):
                return self.sized(obs, reraise(obs, spot, p["threebet_multiplier"]))
            if self.in_range(pct, threebet_width + call_width):
                return check_or_call(obs)
            if pct < threebet_width + call_width + 0.12 and self.chance(p["threebet_bluff_frequency"]):
                return self.sized(obs, reraise(obs, spot, p["threebet_multiplier"]))
            return check_or_fold(obs)

        if spot.situation == "facing_3bet":
            if self.in_range(pct, p["fourbet_range"]):
                return self.sized(obs, reraise(obs, spot, 2.3))
            if self.in_range(pct, p["call_3bet_range"] * (1.2 if spot.i_raised else 0.8)):
                return check_or_call(obs)
            return check_or_fold(obs)

        # facing a 4-bet or more
        if self.in_range(pct, p["fourbet_range"] * 0.7):
            return all_in(obs) if obs.can(ActionType.RAISE) else check_or_call(obs)
        return check_or_fold(obs)

    def sized(self, obs: Observation, action: Action) -> Action:
        """Shove instead of a raise that would commit 40%+ of the stack anyway."""
        if action.type in (ActionType.BET, ActionType.RAISE) and obs.max_raise_to is not None:
            committed = obs.bets[obs.seat]
            if action.amount - committed >= 0.4 * (obs.stack + committed):
                return all_in(obs)
        return action

    # --- postflop (overridden per archetype) ---------------------------------

    def postflop(self, obs: Observation, s: Spot) -> Action:
        raise NotImplementedError


class Nit(HeuristicAgent):
    """Plays few hands, bets only strong ones, almost never bluffs."""

    archetype = "Nit"
    defaults = {
        "tightness": 0.9, "aggression": 0.35, "bluff_frequency": 0.03, "calling_tendency": 0.2,
        "open_range": {"early": 0.05, "middle": 0.07, "late": 0.11, "sb": 0.07},
        "call_range": 0.04, "threebet_range": 0.02, "call_3bet_range": 0.02,
        "fourbet_range": 0.01, "bb_defense": 1.4,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.tier == "monster" or (f.tier == "strong" and s.opponents <= 2):
                return bet_pot_fraction(obs, 0.6)
            if s.aggressor and s.opponents == 1 and not f.board_wet and self.chance(p["bluff_frequency"]):
                return bet_pot_fraction(obs, 0.4)
            return check_or_fold(obs)
        if f.tier == "monster":
            return bet_pot_fraction(obs, 0.8) if self.chance(p["aggression"]) else check_or_call(obs)
        if f.tier == "strong" and (s.raises <= 1 or f.made_strength >= 0.8) and s.bet_ratio <= 1.0:
            return check_or_call(obs)
        if f.tier == "medium" and s.bet_ratio <= 0.4 and s.opponents == 1 and s.raises <= 1:
            return check_or_call(obs)
        if f.strong_draw and obs.street == Street.FLOP and f.draw_equity >= s.pot_odds + 0.05:
            return check_or_call(obs)
        return check_or_fold(obs)


class TAG(HeuristicAgent):
    """Tight-aggressive: position-aware ranges, c-bets, value bets, disciplined calls."""

    archetype = "TAG"
    defaults = {
        "tightness": 0.75, "aggression": 0.65, "bluff_frequency": 0.15, "calling_tendency": 0.35,
        "open_range": {"early": 0.11, "middle": 0.15, "late": 0.28, "sb": 0.2},
        "call_range": 0.07, "threebet_range": 0.04, "threebet_bluff_frequency": 0.1,
        "call_3bet_range": 0.045, "fourbet_range": 0.018,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.tier in ("strong", "monster"):
                return bet_pot_fraction(obs, 0.66)
            if s.aggressor and s.opponents <= 2:
                if f.tier == "medium" and self.chance(0.5):
                    return bet_pot_fraction(obs, 0.5)
                if f.any_draw and self.chance(p["aggression"]):
                    return bet_pot_fraction(obs, 0.6)
                bluff = p["bluff_frequency"] * (1.6 if not f.board_wet else 1.0) * (1.0 if s.opponents == 1 else 0.5)
                if obs.street == Street.FLOP and self.chance(bluff * 2):
                    return bet_pot_fraction(obs, 0.4)
            return check_or_fold(obs)
        if f.tier == "monster" and self.chance(p["aggression"]):
            return bet_pot_fraction(obs, 0.75)
        # Call wider against opponents who bet a lot, tighter against passive ones.
        required = s.pot_odds + 0.03 + 0.05 * (s.raises - 1) - 0.1 * (table_aggression(obs) - 0.3)
        if s.opponents == 1 and f.strong_draw and obs.street == Street.FLOP and self.chance(p["aggression"] * 0.3):
            return bet_pot_fraction(obs, 0.75)
        if s.equity >= required:
            return check_or_call(obs)
        return check_or_fold(obs)


class LAG(HeuristicAgent):
    """Loose-aggressive: wide ranges, frequent c-bets and barrels, raises draws."""

    archetype = "LAG"
    defaults = {
        "tightness": 0.4, "aggression": 0.8, "bluff_frequency": 0.3, "calling_tendency": 0.45,
        "open_range": {"early": 0.18, "middle": 0.24, "late": 0.42, "sb": 0.35},
        "call_range": 0.11, "threebet_range": 0.065, "threebet_bluff_frequency": 0.3,
        "call_3bet_range": 0.07, "fourbet_range": 0.025, "open_size_bb": 2.5,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.tier in ("medium", "strong", "monster"):
                return bet_pot_fraction(obs, 0.75)
            if f.any_draw and self.chance(p["aggression"]):
                return bet_pot_fraction(obs, 0.7)
            barrel = p["bluff_frequency"] * (2.0 if s.aggressor else 1.0) * (1.0 if s.opponents == 1 else 0.4)
            barrel *= 0.5 + table_fold_frequency(obs)  # bluff more into players who fold
            if self.chance(barrel):
                return bet_pot_fraction(obs, 0.66)
            return check_or_fold(obs)
        if f.tier in ("strong", "monster") and self.chance(p["aggression"] * (0.9 if f.tier == "monster" else 0.4)):
            return bet_pot_fraction(obs, 0.85)
        if f.any_draw and obs.street != Street.RIVER and s.raises <= 1 and self.chance(p["aggression"] * 0.35):
            return bet_pot_fraction(obs, 0.85)
        if s.opponents == 1 and s.raises <= 1 and self.chance(p["bluff_frequency"] * 0.4):
            return bet_pot_fraction(obs, 0.9)
        if s.equity + 0.03 * p["calling_tendency"] >= s.pot_odds:
            return check_or_call(obs)
        return check_or_fold(obs)


class CallingStation(HeuristicAgent):
    """Calls with any pair, any draw and sometimes nothing; rarely raises."""

    archetype = "CallingStation"
    defaults = {
        "tightness": 0.2, "aggression": 0.15, "bluff_frequency": 0.02, "calling_tendency": 0.85,
        "open_range": {"early": 0.03, "middle": 0.04, "late": 0.05, "sb": 0.04},
        "open_limp_frequency": 0.3, "limp_range": 0.4, "call_range": 0.4,
        "threebet_range": 0.02, "call_3bet_range": 0.15, "fourbet_range": 0.008, "bb_defense": 1.5,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.tier == "monster" or (f.tier == "strong" and self.chance(0.5)):
                return bet_pot_fraction(obs, 0.4)
            return check_or_fold(obs)
        if f.tier == "monster" and self.chance(0.3):
            return bet_pot_fraction(obs, 0.6)
        if f.made_strength >= 0.3 or f.any_draw:
            return check_or_call(obs)
        if f.overcards or f.made_strength >= 0.14:  # ace-high / overcards
            if s.bet_ratio <= 0.75 and self.chance(p["calling_tendency"]):
                return check_or_call(obs)
        elif s.bet_ratio <= 0.5 and self.chance(p["calling_tendency"] * 0.4):
            return check_or_call(obs)
        return check_or_fold(obs)


class Maniac(HeuristicAgent):
    """Raises and re-raises constantly with wide ranges and large sizes."""

    archetype = "Maniac"
    defaults = {
        "tightness": 0.1, "aggression": 0.95, "bluff_frequency": 0.55, "calling_tendency": 0.6,
        "open_range": {"early": 0.5, "middle": 0.55, "late": 0.7, "sb": 0.65},
        "call_range": 0.2, "threebet_range": 0.25, "threebet_bluff_frequency": 0.5,
        "call_3bet_range": 0.3, "fourbet_range": 0.12, "open_size_bb": 4.0, "threebet_multiplier": 3.5,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        size = 0.8 + 0.4 * self.rng.random()
        if not s.facing_bet:
            if f.tier != "nothing" or f.any_draw or self.chance(p["bluff_frequency"] + 0.25):
                return bet_pot_fraction(obs, size)
            return check_or_fold(obs)
        if f.tier in ("strong", "monster"):
            return all_in(obs) if self.chance(0.3) else bet_pot_fraction(obs, size)
        if s.raises <= 2 and self.chance(p["bluff_frequency"] * 0.7):
            return bet_pot_fraction(obs, size)
        if f.made_strength >= 0.3 or f.any_draw or self.chance(p["calling_tendency"] * 0.5):
            return check_or_call(obs)
        return check_or_fold(obs)


class LoosePassive(HeuristicAgent):
    """Limps and calls a lot preflop, calls light postflop, rarely bets."""

    archetype = "LoosePassive"
    defaults = {
        "tightness": 0.25, "aggression": 0.2, "bluff_frequency": 0.04, "calling_tendency": 0.65,
        "open_range": {"early": 0.04, "middle": 0.05, "late": 0.08, "sb": 0.06},
        "open_limp_frequency": 0.5, "limp_range": 0.35, "call_range": 0.28,
        "threebet_range": 0.025, "call_3bet_range": 0.08, "fourbet_range": 0.01,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.tier == "monster" or (f.tier == "strong" and self.chance(p["aggression"] * 2)):
                return bet_pot_fraction(obs, 0.45)
            return check_or_fold(obs)
        if f.tier == "monster" and self.chance(p["aggression"]):
            return bet_pot_fraction(obs, 0.6)
        if f.made_strength >= 0.5:
            return check_or_call(obs)
        if (f.made_strength >= 0.3 or f.any_draw) and s.bet_ratio <= 1.0 and self.chance(p["calling_tendency"] + 0.2):
            return check_or_call(obs)
        return check_or_fold(obs)


class FitOrFold(HeuristicAgent):
    """Continues only when it hits; folds to aggression when it misses."""

    archetype = "FitOrFold"
    defaults = {
        "tightness": 0.6, "aggression": 0.55, "bluff_frequency": 0.05, "calling_tendency": 0.3,
        "open_range": {"early": 0.14, "middle": 0.18, "late": 0.3, "sb": 0.22},
        "call_range": 0.12, "threebet_range": 0.035, "call_3bet_range": 0.04,
    }

    def postflop(self, obs, s):
        f, p = s.features, self.p
        if not s.facing_bet:
            if f.hit and self.chance(0.85):
                return bet_pot_fraction(obs, 0.6)
            if s.aggressor and obs.street == Street.FLOP and self.chance(p["bluff_frequency"]):
                return bet_pot_fraction(obs, 0.5)
            return check_or_fold(obs)
        if not f.hit:
            return check_or_fold(obs)
        if f.tier in ("strong", "monster") and self.chance(p["aggression"] * 0.6):
            return bet_pot_fraction(obs, 0.8)
        if f.made_strength >= 0.5 or (f.strong_draw and f.draw_equity >= s.pot_odds - 0.05):
            return check_or_call(obs)
        return check_or_fold(obs)


class Balanced(HeuristicAgent):
    """Equity-driven baseline: Monte Carlo equity against estimated ranges,
    pot-odds calls, value bets sized by strength, and bluffs chosen in
    proportion to bet size. Mixes near thresholds. Not GTO."""

    archetype = "Balanced"
    defaults = {
        "tightness": 0.65, "aggression": 0.6, "bluff_frequency": 0.18, "calling_tendency": 0.45,
        "open_range": {"early": 0.13, "middle": 0.17, "late": 0.3, "sb": 0.24},
        "call_range": 0.08, "threebet_range": 0.045, "threebet_bluff_frequency": 0.15,
        "call_3bet_range": 0.05, "fourbet_range": 0.02, "mix_width": 0.025,
    }
    equity_samples = 150

    def postflop(self, obs, s):
        f, p = s.features, self.p
        equity = equity_vs_ranges(obs.hole_cards, obs.board, estimated_ranges(obs)[:4], self.equity_samples, self.rng)
        value_threshold = 0.62 + 0.04 * min(s.opponents - 1, 3)
        if not s.facing_bet:
            if self.chance(ramp(equity, value_threshold, 0.05)):
                return bet_pot_fraction(obs, 0.5 + 0.4 * min(1.0, (equity - value_threshold) * 3))
            size = 0.66
            # Bluff often enough to balance value bets of this size: bluffs/value ~= size / (1 + 2*size)
            bluff_rate = p["bluff_frequency"] * (size / (1 + 2 * size)) / 0.28
            if s.opponents <= 2 and equity < 0.35:
                if f.any_draw and obs.street != Street.RIVER and self.chance(bluff_rate * 2):
                    return bet_pot_fraction(obs, size)
                if self.chance(bluff_rate * (2 if s.aggressor else 1) / s.opponents):
                    return bet_pot_fraction(obs, size)
            return check_or_fold(obs)
        implied = 0.04 if f.any_draw and obs.street != Street.RIVER else 0.0
        if equity >= 0.8 or (equity >= 0.7 and s.raises <= 1 and self.chance(p["aggression"])):
            return bet_pot_fraction(obs, 0.8)
        if s.opponents == 1 and f.strong_draw and obs.street == Street.FLOP and s.raises <= 1 \
                and self.chance(p["bluff_frequency"] * 0.5):
            return bet_pot_fraction(obs, 0.8)
        if self.chance(ramp(equity + implied, s.pot_odds, 0.03)):
            return check_or_call(obs)
        return check_or_fold(obs)


ARCHETYPES: dict[str, type[HeuristicAgent]] = {
    cls.archetype: cls for cls in (Nit, TAG, LAG, CallingStation, Maniac, LoosePassive, FitOrFold, Balanced)
}
