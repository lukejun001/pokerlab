"""Plays one No-Limit Hold'em hand through PokerKit.

PokerKit owns the rules: blinds, betting rounds, legal raise sizes, all-ins,
side pots, showdown evaluation, split pots and stack accounting. This module
deals cards from a seeded deck (PokerKit's own shuffle uses the global
``random`` module, which is not reproducible per hand), asks agents for
actions through ``Observation`` snapshots, validates those actions, and
summarizes the finished hand.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from pokerkit import (
    Automation,
    ChipsPushing,
    HoleCardsShowingOrMucking,
    Mode,
    NoLimitTexasHoldem,
    StandardHighHand,
)

from pokerlab.engine.actions import Action, ActionType, InvalidActionError, validate
from pokerlab.engine.cards import PK_CARDS
from pokerlab.engine.observations import (
    ActionRecord,
    GameInfo,
    Observation,
    PlayerPublicStats,
    PublicHandRecord,
    Street,
    position_labels,
)
from pokerlab.engine.rake import RakeFunction

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

_EMPTY_STATS: Mapping[int, PlayerPublicStats] = MappingProxyType({})


class _MemoizedHighHand(StandardHighHand):
    """PokerKit's own high-hand evaluator, memoized.

    During showdown PokerKit re-evaluates the same hole+board combination
    many times (show/muck decisions, hand killing, each pot). Caching the
    result removes most of that cost without replacing PokerKit's logic.
    """

    _cache: dict = {}

    @classmethod
    def from_game(cls, hole_cards, board_cards=()):
        hole_cards, board_cards = tuple(hole_cards), tuple(board_cards)
        key = (hole_cards, board_cards)
        hand = cls._cache.get(key)
        if hand is None:
            hand = super().from_game(hole_cards, board_cards)
            if len(cls._cache) > 20_000:
                cls._cache.clear()
            cls._cache[key] = hand
        return hand


class _HoldemGame(NoLimitTexasHoldem):
    hand_types = (_MemoizedHighHand,)


class StalledHandError(RuntimeError):
    pass


class ChipConservationError(RuntimeError):
    pass


@dataclass
class HandResult:
    hand_id: int
    button_seat: int
    seats: tuple[int, ...]  # seats dealt in, in PokerKit order (SB first; BB first heads-up)
    positions: dict[int, str]
    hole_cards: dict[int, tuple[int, int]]  # private: for logs/analysis only, never shown to agents
    board: tuple[int, ...]
    actions: tuple[ActionRecord, ...]
    starting_stacks: dict[int, int]
    final_stacks: dict[int, int]
    net: dict[int, int]
    rake: int  # chips PokerKit actually removed
    rake_attributed: dict[int, float]  # rake borne by each pot winner, pro rata
    winnings: dict[int, int]  # chips pushed to each seat from pots (after rake)
    shown_cards: dict[int, tuple[int, ...]]
    showdown_seats: tuple[int, ...]
    flop_seats: tuple[int, ...]
    decisions: int
    invalid_actions: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    decision_time: dict[int, float] = field(default_factory=dict)

    def public_record(self) -> PublicHandRecord:
        return PublicHandRecord(
            hand_id=self.hand_id,
            button_seat=self.button_seat,
            seats=self.seats,
            board=self.board,
            actions=self.actions,
            shown_cards=MappingProxyType(dict(self.shown_cards)),
            net=MappingProxyType(dict(self.net)),
            rake=self.rake,
        )


def seat_order(seats: Sequence[int], button_seat: int) -> tuple[int, ...]:
    """Seats in PokerKit order: starting left of the button, button last."""
    ordered = sorted(seats)
    if button_seat not in ordered:
        raise ValueError(f"button seat {button_seat} is not dealt in")
    k = ordered.index(button_seat)
    return tuple(ordered[k + 1:] + ordered[:k + 1])


def play_hand(
    *,
    agents: Mapping[int, object],
    seats: Sequence[int],
    button_seat: int,
    stacks: Mapping[int, int],
    game: GameInfo,
    rake_function: RakeFunction,
    deck_rng: random.Random,
    hand_id: int = 0,
    previous_hands: tuple[PublicHandRecord, ...] = (),
    player_stats: Mapping[int, PlayerPublicStats] = _EMPTY_STATS,
    max_actions: int = 500,
    strict: bool = False,
) -> HandResult:
    """Play one hand to completion and return its summary.

    ``agents`` maps seat -> agent (anything with ``act(observation)``).
    ``stacks`` maps seat -> chips for every seat in ``seats``.
    """
    order = seat_order(seats, button_seat)
    n = len(order)
    if n < 2:
        raise ValueError("a hand needs at least two players")
    if any(stacks[s] <= 0 for s in order):
        raise ValueError("every dealt-in player needs chips")

    labels = position_labels(n)
    positions = {seat: labels[i] for i, seat in enumerate(order)}
    starting = {seat: stacks[seat] for seat in order}
    num_seats = game.num_seats

    state = _HoldemGame.create_state(
        AUTOMATIONS,
        True,
        0,
        (game.small_blind, game.big_blind),
        game.big_blind,
        tuple(starting[s] for s in order),
        n,
        mode=Mode.CASH_GAME,
        rake=rake_function,
    )

    deck = list(range(52))
    deck_rng.shuffle(deck)
    hole_cards = {}
    for i, seat in enumerate(order):
        cards = (deck[2 * i], deck[2 * i + 1])
        hole_cards[seat] = cards
        state.deal_hole((PK_CARDS[cards[0]], PK_CARDS[cards[1]]))
    next_card = 2 * n
    board: list[int] = []

    history: list[ActionRecord] = []
    folded: set[int] = set()
    flop_seats: tuple[int, ...] = ()
    invalid: dict[int, int] = {}
    errors: list[str] = []
    decision_time: dict[int, float] = {}
    decisions = 0
    steps = 0
    seat_positions = MappingProxyType(dict(positions))

    while state.status:
        steps += 1
        if steps > max_actions:
            raise StalledHandError(f"hand {hand_id} exceeded {max_actions} steps")

        if state.can_burn_card():
            state.burn_card(PK_CARDS[deck[next_card]])
            next_card += 1
            continue
        if state.can_deal_board():
            count = state.street.board_dealing_count
            cards = deck[next_card:next_card + count]
            next_card += count
            state.deal_board(tuple(PK_CARDS[c] for c in cards))
            board.extend(cards)
            if len(board) == 3:
                flop_seats = tuple(s for s in order if s not in folded)
            continue

        actor = state.actor_index
        if actor is None:
            raise StalledHandError(f"hand {hand_id}: no actor and no pending dealing")

        seat = order[actor]
        to_call = state.checking_or_calling_amount
        legal = {ActionType.FOLD, ActionType.CALL} if to_call else {ActionType.CHECK}
        min_to = max_to = None
        if state.can_complete_bet_or_raise_to():
            max_to = state.max_completion_betting_or_raising_to_amount
            min_to = min(state.min_completion_betting_or_raising_to_amount, max_to)
            legal.add(ActionType.RAISE if max(state.bets) > 0 else ActionType.BET)
        legal = frozenset(legal)

        stacks_by_seat = [0] * num_seats
        bets_by_seat = [0] * num_seats
        in_hand = [False] * num_seats
        for i, s in enumerate(order):
            stacks_by_seat[s] = state.stacks[i]
            bets_by_seat[s] = state.bets[i]
            in_hand[s] = state.statuses[i]
        all_in = tuple(in_hand[s] and stacks_by_seat[s] == 0 for s in range(num_seats))

        observation = Observation(
            hand_id=hand_id,
            seat=seat,
            hole_cards=hole_cards[seat],
            board=tuple(board),
            street=Street(state.street_index),
            pot=-sum(state.payoffs),
            stack=state.stacks[actor],
            stacks=tuple(stacks_by_seat),
            bets=tuple(bets_by_seat),
            in_hand=tuple(in_hand),
            all_in=all_in,
            button_seat=button_seat,
            position=positions[seat],
            position_index=actor,
            num_players=n,
            legal_actions=legal,
            to_call=to_call,
            min_raise_to=min_to,
            max_raise_to=max_to,
            history=tuple(history),
            seat_positions=seat_positions,
            game=game,
            previous_hands=previous_hands,
            player_stats=player_stats,
        )

        decisions += 1
        started = time.perf_counter()
        try:
            action = validate(agents[seat].act(observation), legal, min_to, max_to)
        except Exception as exc:  # agent bugs must not crash a million-hand run
            if strict:
                raise
            invalid[seat] = invalid.get(seat, 0) + 1
            if len(errors) < 5:
                errors.append(f"seat {seat}: {type(exc).__name__}: {exc}")
            action = Action.check() if ActionType.CHECK in legal else Action.fold()
        decision_time[seat] = decision_time.get(seat, 0.0) + time.perf_counter() - started

        street = Street(state.street_index)
        if action.type is ActionType.FOLD:
            state.fold()
            folded.add(seat)
            amount = 0
        elif action.type in (ActionType.CHECK, ActionType.CALL):
            amount = state.check_or_call().amount
        else:
            amount = state.complete_bet_or_raise_to(action.amount).amount
        history.append(ActionRecord(seat, street, action.type, amount, state.stacks[actor] == 0))

    final = {seat: state.stacks[i] for i, seat in enumerate(order)}
    pots = list(state.pots)
    rake = sum(p.raked_amount for p in pots)
    if sum(starting.values()) != sum(final.values()) + rake:
        raise ChipConservationError(
            f"hand {hand_id}: {sum(starting.values())} != {sum(final.values())} + rake {rake}"
        )

    pushed_by_pot: dict[int, dict[int, int]] = {}
    shown: dict[int, tuple[int, ...]] = {}
    for op in state.operations:
        if isinstance(op, ChipsPushing):
            by_seat = pushed_by_pot.setdefault(op.pot_index, {})
            for i, amount in enumerate(op.amounts):
                if amount:
                    by_seat[order[i]] = by_seat.get(order[i], 0) + amount
        elif isinstance(op, HoleCardsShowingOrMucking) and op.hole_cards:
            shown[order[op.player_index]] = hole_cards[order[op.player_index]]

    winnings: dict[int, int] = {}
    rake_attributed: dict[int, float] = {}
    for pot_index, by_seat in pushed_by_pot.items():
        pot_total = sum(by_seat.values())
        raked = pots[pot_index].raked_amount
        for seat, amount in by_seat.items():
            winnings[seat] = winnings.get(seat, 0) + amount
            if raked:
                rake_attributed[seat] = rake_attributed.get(seat, 0.0) + raked * amount / pot_total

    remaining = tuple(s for s in order if s not in folded)
    return HandResult(
        hand_id=hand_id,
        button_seat=button_seat,
        seats=order,
        positions=positions,
        hole_cards=hole_cards,
        board=tuple(board),
        actions=tuple(history),
        starting_stacks=starting,
        final_stacks=final,
        net={s: final[s] - starting[s] for s in order},
        rake=rake,
        rake_attributed=rake_attributed,
        winnings=winnings,
        shown_cards=shown,
        showdown_seats=remaining if len(remaining) > 1 else (),
        flop_seats=flop_seats,
        decisions=decisions,
        invalid_actions=invalid,
        errors=errors,
        decision_time=decision_time,
    )
