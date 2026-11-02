import random

import pytest
from conftest import RiggedDeck, Scripted, folder, jammer, passive, random_policy, rigged

from pokerlab.engine.actions import Action, ActionType, InvalidActionError, validate
from pokerlab.engine.cards import parse_cards
from pokerlab.engine.game import StalledHandError, seat_order
from pokerlab.engine.observations import Street, position_labels


def test_position_labels():
    assert position_labels(9) == ("SB", "BB", "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN")
    assert position_labels(6) == ("SB", "BB", "LJ", "HJ", "CO", "BTN")
    assert position_labels(3) == ("SB", "BB", "BTN")
    assert position_labels(2) == ("BB", "BTN")


def test_seat_order_rotates_from_button():
    assert seat_order(range(9), 4) == (5, 6, 7, 8, 0, 1, 2, 3, 4)
    assert seat_order([0, 2, 5], 2) == (5, 0, 2)
    with pytest.raises(ValueError):
        seat_order([0, 1], 3)


def test_complete_nine_player_hand(run_hand):
    agents = {s: Scripted(passive) for s in range(9)}
    r = run_hand(agents, button=8)
    assert r.positions[0] == "SB" and r.positions[1] == "BB" and r.positions[8] == "BTN"
    assert len(r.board) == 5
    assert set(r.showdown_seats) == set(range(9))
    assert sum(r.net.values()) == 0
    # Every street was played in order and first-to-act preflop was UTG (seat 2).
    streets = [a.street for a in r.actions]
    assert streets == sorted(streets)
    assert r.actions[0].seat == 2
    assert {a.street for a in r.actions} == {Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER}


def test_preflop_fold_to_big_blind(run_hand):
    agents = {s: Scripted(folder) for s in range(9)}
    r = run_hand(agents, button=8)
    assert r.board == ()
    assert r.net[0] == -25 and r.net[1] == 25
    assert all(r.net[s] == 0 for s in range(2, 9))
    assert r.showdown_seats == () and r.flop_seats == ()
    assert r.shown_cards == {}


def test_heads_up_blinds_and_order(run_hand):
    agents = {0: Scripted(passive), 1: Scripted(passive)}
    r = run_hand(agents, button=1)
    assert r.positions == {0: "BB", 1: "BTN"}
    assert r.actions[0].seat == 1 and r.actions[0].street == Street.PREFLOP  # button/SB acts first preflop
    flop = [a for a in r.actions if a.street == Street.FLOP]
    assert flop[0].seat == 0  # big blind acts first postflop
    first_obs = agents[1].observations[0]
    assert first_obs.to_call == 25 and first_obs.bets[1] == 25 and first_obs.bets[0] == 50


def test_check_call_amounts_and_legal_actions(run_hand):
    agents = {s: Scripted(passive) for s in range(3)}
    r = run_hand(agents, button=2)
    btn_obs = agents[2].observations[0]
    assert btn_obs.legal_actions == {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
    assert btn_obs.to_call == 50 and btn_obs.min_raise_to == 100 and btn_obs.max_raise_to == 5000
    bb_obs = agents[1].observations[0]
    assert ActionType.CHECK in bb_obs.legal_actions and ActionType.FOLD not in bb_obs.legal_actions
    flop_obs = agents[0].observations[1]
    assert flop_obs.street == Street.FLOP
    assert ActionType.BET in flop_obs.legal_actions and ActionType.RAISE not in flop_obs.legal_actions
    assert r.actions[0].amount == 50  # button calls 50


def test_raise_validation():
    legal = frozenset({ActionType.FOLD, ActionType.CALL, ActionType.RAISE})
    assert validate(Action.raise_to(200), legal, 100, 5000) == Action.raise_to(200)
    with pytest.raises(InvalidActionError):
        validate(Action.raise_to(99), legal, 100, 5000)
    with pytest.raises(InvalidActionError):
        validate(Action.raise_to(5001), legal, 100, 5000)
    with pytest.raises(InvalidActionError):
        validate(Action.bet(200), legal, 100, 5000)  # must be RAISE when facing a bet
    with pytest.raises(InvalidActionError):
        validate(Action.check(), legal, 100, 5000)
    with pytest.raises(InvalidActionError):
        validate(Action(ActionType.RAISE, None), legal, 100, 5000)
    with pytest.raises(InvalidActionError):
        validate("fold", legal, 100, 5000)


def test_invalid_action_falls_back_and_is_counted(run_hand):
    def bad(obs):
        return Action.raise_to(1) if obs.can(ActionType.RAISE) else Action.bet(10**9)

    agents = {0: Scripted(passive), 1: Scripted(passive), 2: Scripted(bad)}
    r = run_hand(agents, button=2)
    assert r.invalid_actions[2] >= 1
    assert r.actions[0].type is ActionType.FOLD  # button faced a bet: fallback is fold
    assert "outside legal range" in r.errors[0]


def test_invalid_action_strict_mode_raises(run_hand):
    agents = {0: Scripted(passive), 1: Scripted(passive), 2: Scripted(lambda o: Action.check())}
    with pytest.raises(InvalidActionError):
        run_hand(agents, button=2, strict=True)


def test_agent_exception_is_contained(run_hand):
    def boom(obs):
        raise RuntimeError("agent bug")

    agents = {0: Scripted(boom), 1: Scripted(passive), 2: Scripted(passive)}
    r = run_hand(agents, button=2)
    assert r.invalid_actions[0] >= 1
    assert "RuntimeError: agent bug" in r.errors[0]
    assert sum(r.net.values()) == 0


def test_all_in_side_pots_and_stacks(run_hand):
    # PokerKit order for button=2: seat 0 (SB), seat 1 (BB), seat 2 (BTN).
    # Seat 0 has the nuts, seat 1 second best, seat 2 worst; stacks 1000 / 3000 / 5000.
    deck = rigged(["AsAh", "KsKh", "7c2d"], board="Ad8c9dJh3s")
    agents = {s: Scripted(jammer) for s in range(3)}
    r = run_hand(agents, button=2, stacks={0: 1000, 1: 3000, 2: 5000}, deck=deck)
    assert r.final_stacks == {0: 3000, 1: 4000, 2: 2000}
    assert r.net == {0: 2000, 1: 1000, 2: -3000}
    assert set(r.shown_cards) == {0, 1, 2}
    assert any(a.all_in for a in r.actions)


def test_split_pot_on_board(run_hand):
    deck = rigged(["2c3d", "4h5h"], board="AsKsQsJsTs")  # royal flush on board
    agents = {0: Scripted(passive), 1: Scripted(passive)}
    r = run_hand(agents, button=1, deck=deck)
    assert r.net == {0: 0, 1: 0}
    assert r.winnings == {0: 50, 1: 50}


def test_three_way_split_with_odd_chips(run_hand):
    from pokerlab.engine.rake import RakeConfig

    # Everyone plays the board; 150-chip pot raked 5% -> 7, leaving 143 to split three ways.
    deck = rigged(["2c3d", "2d3h", "2h3s"], board="AsKsQsJsTs")
    agents = {s: Scripted(passive) for s in range(3)}
    r = run_hand(agents, button=2, deck=deck, rake=RakeConfig(percentage=0.05))
    assert r.rake == 7
    assert sum(r.winnings.values()) == 143
    assert sorted(r.winnings.values()) in ([47, 48, 48], [47, 47, 49])
    assert sum(r.final_stacks.values()) + r.rake == 15000


def test_dead_seats_are_skipped(run_hand):
    agents = {0: Scripted(passive), 3: Scripted(passive), 7: Scripted(passive)}
    r = run_hand(agents, button=3)
    assert r.seats == (7, 0, 3)
    assert r.positions == {7: "SB", 0: "BB", 3: "BTN"}


def test_stall_guard(run_hand):
    agents = {s: Scripted(passive) for s in range(9)}
    with pytest.raises(StalledHandError):
        run_hand(agents, max_actions=5)


def test_hidden_information_isolation(run_hand):
    """Agents only ever see their own hole cards and dealt board cards."""
    rng = random.Random(5)
    for hand in range(40):
        agents = {s: Scripted(random_policy(rng)) for s in range(6)}
        r = run_hand(agents, button=hand % 6, deck=random.Random(hand))
        for seat, agent in agents.items():
            others = {c for s, cards in r.hole_cards.items() if s != seat for c in cards}
            for obs in agent.observations:
                assert obs.hole_cards == r.hole_cards[seat]
                visible = set(obs.hole_cards) | set(obs.board)
                assert not visible & others
                expected_board = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[obs.street]
                assert obs.board == r.board[:expected_board]
                _assert_plain_data(obs)
        # The public record reveals only cards shown at showdown.
        record = r.public_record()
        assert set(record.shown_cards) <= set(r.showdown_seats)


def _assert_plain_data(value, depth=0):
    """Observations must be inert data with no path back to the PokerKit state."""
    import dataclasses
    from enum import Enum
    from types import MappingProxyType

    assert depth < 8
    if value is None or isinstance(value, (int, float, str, bool, Enum)):
        return
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            _assert_plain_data(getattr(value, f.name), depth + 1)
        return
    if isinstance(value, (dict, MappingProxyType)):
        for k, v in value.items():
            _assert_plain_data(k, depth + 1)
            _assert_plain_data(v, depth + 1)
        return
    assert isinstance(value, (tuple, frozenset)), f"unexpected type in observation: {type(value)}"
    for item in value:
        _assert_plain_data(item, depth + 1)


def test_observations_are_immutable(run_hand):
    agents = {s: Scripted(passive) for s in range(3)}
    run_hand(agents)
    obs = agents[0].observations[0]
    with pytest.raises(Exception):
        obs.hole_cards = parse_cards("AsAh")
    with pytest.raises(TypeError):
        obs.seat_positions[0] = "BTN"


def test_deterministic_dealing(run_hand):
    def play(seed):
        agents = {s: Scripted(passive) for s in range(4)}
        return run_hand(agents, deck=random.Random(seed))

    a, b, c = play(7), play(7), play(8)
    assert a.hole_cards == b.hole_cards and a.board == b.board
    assert a.hole_cards != c.hole_cards


def test_rigged_deck_helper():
    deck = list(range(52))
    RiggedDeck("AsKs").shuffle(deck)
    assert deck[:2] == list(parse_cards("AsKs")) and sorted(deck) == list(range(52))
