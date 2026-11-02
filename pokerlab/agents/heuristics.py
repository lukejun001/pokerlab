"""Shared poker reasoning used by the heuristic agents and strategies.

Everything here works from an ``Observation`` only. Strength numbers are
rough approximations of equity against a random hand; they are heuristics for
test opponents, not solver output.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass

from pokerlab.agents.preflop import hand_class, preflop_equity, preflop_percentile
from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.cards import (
    FLUSH,
    FULL_HOUSE,
    PAIR,
    STRAIGHT,
    STRAIGHT_HIGH,
    TRIPS,
    TWO_PAIR,
    evaluate,
)
from pokerlab.engine.observations import Observation, PlayerPublicStats, Street, position_group

# --- preflop ---------------------------------------------------------------

# All 1326 two-card combos, strongest first, for fast range sampling: the
# top ``w`` fraction of hands is a prefix of this list.
_COMBOS_BY_STRENGTH = sorted(
    ((a, b) for a in range(52) for b in range(a + 1, 52)), key=preflop_percentile
)
_SORTED_PERCENTILES = [preflop_percentile(combo) for combo in _COMBOS_BY_STRENGTH]


def range_size(width: float) -> int:
    """Number of combos in the top ``width`` fraction of starting hands."""
    if width >= 1.0:
        return len(_COMBOS_BY_STRENGTH)
    return max(1, bisect_left(_SORTED_PERCENTILES, width))


def preflop_strength(hole) -> float:
    """1.0 for AA down to ~0.0 for 72o (1 - percentile)."""
    return 1.0 - preflop_percentile(hole)


@dataclass(frozen=True, slots=True)
class PreflopSpot:
    raises: int  # voluntary raises so far (the big blind is not a raise)
    limpers: int  # calls of the big blind before any raise
    callers: int  # calls of the current raise
    raise_to: int  # current bet level in chips
    last_raiser: int | None
    i_raised: bool
    situation: str  # unopened | limped | facing_open | facing_3bet | facing_4bet

    @staticmethod
    def from_obs(obs: Observation) -> PreflopSpot:
        raises = limpers = callers = 0
        last_raiser = None
        i_raised = False
        for a in obs.history:
            if a.street != Street.PREFLOP:
                break
            if a.type is ActionType.RAISE:
                raises += 1
                callers = 0
                last_raiser = a.seat
                i_raised = i_raised or a.seat == obs.seat
            elif a.type is ActionType.CALL:
                if raises:
                    callers += 1
                else:
                    limpers += 1
        if raises == 0:
            situation = "limped" if limpers else "unopened"
        else:
            situation = ("facing_open", "facing_3bet")[raises - 1] if raises <= 2 else "facing_4bet"
        return PreflopSpot(raises, limpers, callers, obs.current_bet, last_raiser, i_raised, situation)


def position_bucket(obs: Observation) -> str:
    if obs.position == "SB":
        return "sb"
    if obs.position == "BB":
        return "bb"
    return position_group(obs.position)


# --- postflop hand features ------------------------------------------------

@dataclass(frozen=True, slots=True)
class HandFeatures:
    category: int
    made_strength: float  # ~equity vs one random hand from the made hand alone
    tier: str  # nothing | weak | medium | strong | monster
    outs: int  # approximate clean outs to improve
    flush_draw: bool
    nut_flush_draw: bool
    straight_draw: str  # none | gutshot | oesd
    overcards: int
    draw_equity: float
    board_paired: bool
    board_flush_threat: bool  # 3+ of a suit on board
    board_wet: bool

    @property
    def strong_draw(self) -> bool:
        return self.outs >= 8

    @property
    def any_draw(self) -> bool:
        return self.outs >= 4

    @property
    def hit(self) -> bool:
        """Has something worth continuing with: medium+ made hand or a real draw."""
        return self.made_strength >= 0.5 or self.outs >= 8


def _tier(strength: float) -> str:
    if strength >= 0.86:
        return "monster"
    if strength >= 0.68:
        return "strong"
    if strength >= 0.5:
        return "medium"
    if strength >= 0.3:
        return "weak"
    return "nothing"


def analyze_hand(hole: tuple[int, int], board: tuple[int, ...]) -> HandFeatures:
    cards = (*hole, *board)
    value = evaluate(cards)
    cat = value >> 20
    board_value = evaluate(board)
    board_cat = board_value >> 20
    board_ranks = sorted({c >> 2 for c in board}, reverse=True)
    top_board = board_ranks[0]
    h1, h2 = sorted((hole[0] >> 2, hole[1] >> 2), reverse=True)
    pocket_pair = h1 == h2

    suit_counts = [0, 0, 0, 0]
    for c in board:
        suit_counts[c & 3] += 1
    board_flush_threat = max(suit_counts) >= 3
    board_paired = board_cat >= PAIR
    board_mask = 0
    for c in board:
        board_mask |= 1 << (c >> 2)
    connected = sum(1 for r in board_ranks if r - 1 in board_ranks or r - 2 in board_ranks)
    board_wet = board_flush_threat or (max(suit_counts) == 2 and len(board) == 3) or connected >= 2

    # --- made hand strength
    if cat >= STRAIGHT:
        if len(board) == 5 and value == board_value:
            strength = 0.25  # playing the board
        elif cat == FLUSH:
            flush_suit = max(range(4), key=lambda s: suit_counts[s] + sum(1 for c in hole if c & 3 == s))
            high = max((c >> 2 for c in hole if c & 3 == flush_suit), default=0)
            strength = 0.95 if high >= 11 else (0.88 if high >= 8 else 0.8)
            if suit_counts[flush_suit] >= 4:
                strength -= 0.08
        elif cat == STRAIGHT:
            strength = 0.86 if board_flush_threat or board_paired else 0.9
        elif cat == FULL_HOUSE and board_cat == FULL_HOUSE:
            strength = 0.3
        else:
            strength = 0.97 if cat > board_cat else 0.5
    elif cat == TRIPS:
        if pocket_pair:
            strength = 0.9  # set
        elif board_cat == TRIPS:
            strength = 0.25 + 0.05 * (h1 >= 11)
        else:
            trips_rank = value >> 16 & 0xF
            kicker = h2 if h1 == trips_rank else h1
            strength = 0.8 + 0.05 * (kicker >= 10)
    elif cat == TWO_PAIR and board_cat < PAIR:
        top_two = h1 in board_ranks and h2 in board_ranks and {h1, h2} >= {board_ranks[0], board_ranks[1]}
        strength = 0.86 if top_two else 0.8
    else:
        # One real pair at most: find our pair rank ignoring pairs on the board.
        board_pair_ranks = {r for r in board_ranks if sum(1 for c in board if c >> 2 == r) >= 2}
        pair_rank = None
        if pocket_pair and h1 not in board_ranks:
            pair_rank = h1
        else:
            for h in (h1, h2):
                if h in board_ranks and h not in board_pair_ranks:
                    pair_rank = h
                    break
        if pair_rank is None:
            # No pair of our own: ace/king high.
            strength = 0.12 + 0.03 * (h1 >= 12) + 0.02 * (h1 >= 11)
        elif pocket_pair and pair_rank > top_board:
            strength = 0.76 + 0.06 * (pair_rank >= 10)
        elif pair_rank == top_board:
            kicker = h2 if h1 == pair_rank else h1
            strength = 0.62 + 0.08 * (kicker >= 10) + 0.04 * (kicker == 12 or pair_rank == 12)
        elif len(board_ranks) > 1 and pair_rank >= board_ranks[1]:
            strength = 0.5
        else:
            strength = 0.36
        if board_flush_threat and len(board) >= 4 and max(suit_counts) >= 4:
            strength -= 0.1
        if board_paired and pair_rank is not None:
            strength -= 0.04

    # --- draws (flop and turn only)
    flush_draw = nut_flush_draw = False
    straight_draw = "none"
    outs = 0
    overcards = 0
    if len(board) < 5 and cat < FLUSH:
        for suit in range(4):
            mine = [c for c in hole if c & 3 == suit]
            if mine and suit_counts[suit] + len(mine) == 4:
                flush_draw = True
                nut_flush_draw = any(c >> 2 == 12 for c in mine)
        if cat < STRAIGHT:
            all_mask = board_mask | 1 << h1 | 1 << h2
            completing = 0
            for r in range(13):
                bit = 1 << r
                if all_mask & bit:
                    continue
                if STRAIGHT_HIGH[all_mask | bit] >= 0 and STRAIGHT_HIGH[board_mask | bit] < 0:
                    completing += 1
            straight_draw = "oesd" if completing >= 2 else ("gutshot" if completing == 1 else "none")
        outs = (9 if flush_draw else 0) + {"oesd": 8, "gutshot": 4}.get(straight_draw, 0)
        if flush_draw and straight_draw != "none":
            outs -= 2
        if strength < 0.3:
            overcards = (h1 > top_board) + (h2 > top_board)
            outs += 2 * overcards  # discounted: pairing may not be good enough
    per_out = 0.04 if len(board) == 3 else 0.022
    draw_equity = min(0.65, outs * per_out)

    strength = max(0.0, min(1.0, strength))
    return HandFeatures(
        category=cat,
        made_strength=strength,
        tier=_tier(strength),
        outs=outs,
        flush_draw=flush_draw,
        nut_flush_draw=nut_flush_draw,
        straight_draw=straight_draw,
        overcards=overcards,
        draw_equity=draw_equity,
        board_paired=board_paired,
        board_flush_threat=board_flush_threat,
        board_wet=board_wet,
    )


def heuristic_equity(features: HandFeatures, opponents: int) -> float:
    """Cheap equity estimate from features, discounted for extra opponents."""
    base = max(features.made_strength, features.draw_equity)
    if features.made_strength >= 0.3 and features.outs:
        base = min(0.97, base + 0.5 * features.draw_equity)
    return base ** (1 + 0.5 * max(0, opponents - 1))


# --- Monte Carlo equity against estimated ranges -----------------------------

def equity_vs_ranges(
    hole: tuple[int, int],
    board: tuple[int, ...],
    range_widths: list[float],
    samples: int,
    rng: random.Random,
) -> float:
    """Equity against opponents whose hands are drawn from the top
    ``range_widths[i]`` fraction of starting hands (1.0 = any two cards).

    Opponent hands are drawn uniformly from the strongest combos (by preflop
    percentile), rejecting ones that collide with known or already-drawn
    cards, so the estimate reflects who continued in the hand rather than
    assuming random holdings.
    """
    if not range_widths:
        return 1.0
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    hole_l, board_l = list(hole), list(board)
    sizes = [range_size(w) for w in range_widths]
    combos = _COMBOS_BY_STRENGTH
    random_ = rng.random
    sample = rng.sample
    total = 0.0
    for _ in range(samples):
        runout = sample(deck, need) if need else []
        blocked = dead.union(runout)
        opp_hands = []
        for size in sizes:
            for _attempt in range(50):
                a, b = combos[int(random_() * size)]
                if a not in blocked and b not in blocked:
                    break
            else:
                while True:  # range exhausted by blockers: take any live hand
                    a, b = sample(deck, 2)
                    if a not in blocked and b not in blocked:
                        break
            blocked.add(a)
            blocked.add(b)
            opp_hands.append((a, b))
        full_board = board_l + runout
        mine = evaluate(hole_l + full_board)
        best, ties = 0, 0
        for a, b in opp_hands:
            v = evaluate([a, b, *full_board])
            if v > best:
                best, ties = v, 1
            elif v == best:
                ties += 1
        if mine > best:
            total += 1.0
        elif mine == best:
            total += 1.0 / (ties + 1)
    return total / samples


# --- opponent tendencies -------------------------------------------------------

PRIOR_HANDS = 30  # shrink sparse observed stats toward population defaults


def _shrunk(observed: float, n: int, prior: float) -> float:
    return (observed * n + prior * PRIOR_HANDS) / (n + PRIOR_HANDS)


def opponent_vpip(stats: PlayerPublicStats | None) -> float:
    if stats is None:
        return 0.3
    return _shrunk(stats.vpip, stats.hands, 0.3)


def opponent_aggression(stats: PlayerPublicStats | None) -> float:
    if stats is None:
        return 0.3
    actions = stats.postflop_aggressive + stats.postflop_calls + stats.postflop_folds + stats.postflop_checks
    return _shrunk(stats.aggression_frequency, actions, 0.3)


def opponent_fold_frequency(stats: PlayerPublicStats | None) -> float:
    if stats is None:
        return 0.4
    facing = stats.postflop_calls + stats.postflop_folds + stats.postflop_aggressive
    return _shrunk(stats.fold_frequency, facing, 0.4)


def active_opponents(obs: Observation) -> list[int]:
    return [s for s, alive in enumerate(obs.in_hand) if alive and s != obs.seat]


def table_aggression(obs: Observation) -> float:
    opps = active_opponents(obs)
    if not opps:
        return 0.3
    return sum(opponent_aggression(obs.player_stats.get(s)) for s in opps) / len(opps)


def table_fold_frequency(obs: Observation) -> float:
    opps = active_opponents(obs)
    if not opps:
        return 0.4
    return sum(opponent_fold_frequency(obs.player_stats.get(s)) for s in opps) / len(opps)


def estimated_ranges(obs: Observation) -> list[float]:
    """Range width per active opponent from their VPIP and preflop actions."""
    preflop_raisers = {a.seat for a in obs.history if a.street == Street.PREFLOP and a.type is ActionType.RAISE}
    widths = []
    for seat in active_opponents(obs):
        width = max(0.12, min(1.0, opponent_vpip(obs.player_stats.get(seat)) * 1.2))
        if seat in preflop_raisers:
            width *= 0.6
        if obs.seat_positions.get(seat) == "BB" and not any(
            a.seat == seat and a.street == Street.PREFLOP and a.type is not ActionType.CHECK for a in obs.history
        ):
            width = 1.0  # checked option in the big blind: any two cards
        widths.append(width)
    return widths


# --- street context ------------------------------------------------------------

def street_raises(obs: Observation) -> int:
    return sum(1 for a in obs.history if a.street == obs.street and a.type in (ActionType.BET, ActionType.RAISE))


def was_preflop_aggressor(obs: Observation) -> bool:
    last = None
    for a in obs.history:
        if a.street == Street.PREFLOP and a.type is ActionType.RAISE:
            last = a.seat
    return last == obs.seat


def bet_to_pot_ratio(obs: Observation) -> float:
    """Size of the bet we face relative to the pot before it was made."""
    if not obs.to_call:
        return 0.0
    return obs.to_call / max(1, obs.pot - obs.to_call)


def _postflop_order(obs: Observation, seat: int) -> int:
    # Postflop action starts left of the button and ends on the button.
    return (seat - obs.button_seat - 1) % obs.game.num_seats


def in_position(obs: Observation) -> bool:
    """True if no active opponent acts after us postflop."""
    me = _postflop_order(obs, obs.seat)
    return all(_postflop_order(obs, s) < me for s in active_opponents(obs))


# --- action helpers ---------------------------------------------------------

def check_or_fold(obs: Observation) -> Action:
    return Action.check() if obs.can(ActionType.CHECK) else Action.fold()


def check_or_call(obs: Observation) -> Action:
    return Action.check() if obs.can(ActionType.CHECK) else Action.call()


def raise_to(obs: Observation, amount: float) -> Action:
    """Bet/raise to ``amount`` clamped to the legal range, or call/check if
    raising is not allowed."""
    for kind in (ActionType.BET, ActionType.RAISE):
        if obs.can(kind):
            return Action(kind, int(max(obs.min_raise_to, min(obs.max_raise_to, amount))))
    return check_or_call(obs)


def bet_pot_fraction(obs: Observation, fraction: float) -> Action:
    """Bet ``fraction`` of the pot, or raise by ``fraction`` of the pot after calling."""
    pot_after_call = obs.pot + obs.to_call
    return raise_to(obs, obs.current_bet + fraction * pot_after_call)


def all_in(obs: Observation) -> Action:
    if obs.max_raise_to is None:
        return check_or_call(obs)
    return raise_to(obs, obs.max_raise_to)


def open_raise(obs: Observation, spot: PreflopSpot, size_bb: float) -> Action:
    return raise_to(obs, obs.big_blind * (size_bb + spot.limpers))


def reraise(obs: Observation, spot: PreflopSpot, multiplier: float) -> Action:
    return raise_to(obs, spot.raise_to * multiplier + spot.callers * spot.raise_to)


def ramp(value: float, threshold: float, width: float) -> float:
    """Probability that rises linearly from 0 to 1 across [threshold - width, threshold + width].

    Used to mix actions near decision boundaries instead of switching hard.
    """
    if width <= 0:
        return 1.0 if value >= threshold else 0.0
    return max(0.0, min(1.0, (value - threshold + width) / (2 * width)))


__all__ = [
    "PreflopSpot",
    "HandFeatures",
    "analyze_hand",
    "heuristic_equity",
    "equity_vs_ranges",
    "preflop_strength",
    "preflop_percentile",
    "preflop_equity",
    "hand_class",
]
