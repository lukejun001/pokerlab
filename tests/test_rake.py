import random

import pytest
from conftest import Scripted, folder, jammer, passive, random_policy, rigged

from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.rake import RakeConfig

RAKE = RakeConfig(percentage=0.04, cap=150, no_flop_no_drop=True)


def test_hand_rake_rules():
    assert RakeConfig().hand_rake(10_000, 9, True) == 0
    assert RAKE.hand_rake(1000, 9, True) == 40
    assert RAKE.hand_rake(1010, 9, True) == 40  # rounds down
    assert RAKE.hand_rake(10_000, 9, True) == 150  # capped
    assert RAKE.hand_rake(10_000, 9, False) == 0  # no flop, no drop
    assert RakeConfig(percentage=0.04, no_flop_no_drop=False).hand_rake(1000, 9, False) == 40
    assert RakeConfig(percentage=0.04).hand_rake(1_000_000, 9, True) == 40_000  # uncapped


def test_cap_by_players():
    rake = RakeConfig(percentage=0.05, cap=999, cap_by_players={2: 50, 3: 100, 5: 150})
    assert rake.cap_for(2) == 50
    assert rake.cap_for(4) == 100
    assert rake.cap_for(9) == 150
    assert rake.hand_rake(10_000, 2, True) == 50
    assert RakeConfig(percentage=0.05, cap=999, cap_by_players={3: 100}).cap_for(2) == 999


def test_config_validation_and_parsing():
    with pytest.raises(ValueError):
        RakeConfig(percentage=1.5)
    parsed = RakeConfig.from_dict({"percentage": 0.04, "cap": 150, "cap_by_players": {"2": 50}})
    assert parsed.cap_by_players == {2: 50} and parsed.cap == 150


def test_single_pot_rake_comes_out_of_winner(run_hand):
    # Heads-up limp/check to showdown: pot 100 -> rake 4, winner nets +46.
    deck = rigged(["AsAh", "7c2d"], board="Kd8c9dJh3s")
    agents = {0: Scripted(passive), 1: Scripted(passive)}
    r = run_hand(agents, button=1, deck=deck, rake=RAKE)
    assert r.rake == 4
    assert r.net == {0: 46, 1: -50}
    assert r.rake_attributed == {0: 4.0}


def test_no_flop_no_drop(run_hand):
    agents = {s: Scripted(folder) for s in range(9)}
    r = run_hand(agents, rake=RAKE)
    assert r.rake == 0 and sum(r.net.values()) == 0


def test_preflop_all_in_with_board_is_raked(run_hand):
    # All-in preflop still deals a flop, so the pot is raked.
    agents = {0: Scripted(jammer), 1: Scripted(jammer)}
    r = run_hand(agents, button=1, rake=RAKE)
    assert len(r.board) == 5
    assert r.rake == 150


def test_cap_applies_to_whole_hand_with_side_pots(run_hand):
    """PokerKit's own rake caps each pot separately; ours caps the hand."""
    deck = rigged(["AsAh", "KsKh", "7c2d"], board="Ad8c9dJh3s")
    agents = {s: Scripted(jammer) for s in range(3)}
    r = run_hand(agents, button=2, stacks={0: 1000, 1: 3000, 2: 5000}, deck=deck, rake=RAKE)
    # Pots: main 3000 (all three), side 4000 (seats 1, 2). Nominal hand rake = min(280, 150) = 150.
    assert 149 <= r.rake <= 150
    assert sum(r.final_stacks.values()) + r.rake == 9000
    # Rake split pro rata across the pots won: main pot -> seat 0, side pot -> seat 1.
    assert set(r.rake_attributed) == {0, 1}
    assert sum(r.rake_attributed.values()) == pytest.approx(r.rake)
    assert r.net[2] == -3000  # the loser pays no rake


def test_split_pot_rake_attribution(run_hand):
    deck = rigged(["2c3d", "4h5d"], board="AsKsQsJsTs")
    agents = {0: Scripted(passive), 1: Scripted(passive)}
    r = run_hand(agents, button=1, deck=deck, rake=RAKE)
    assert r.rake == 4
    assert r.rake_attributed == {0: 2.0, 1: 2.0}
    assert r.net == {0: -2, 1: -2}


def test_uncalled_bet_is_not_raked(run_hand):
    def bet_flop(obs):
        if obs.street > 0 and obs.can(ActionType.BET):
            return Action.bet(1000)
        return passive(obs) if obs.street == 0 else Action.fold()

    agents = {0: Scripted(bet_flop), 1: Scripted(bet_flop)}
    r = run_hand(agents, button=1, rake=RAKE)
    # Pot was 100 when the flop bet went uncalled: rake 4 on 100, not on 1100.
    assert r.rake == 4
    assert sorted(r.net.values()) == [-50, 46]


@pytest.mark.parametrize("players", [2, 3, 6, 9])
def test_randomized_conservation_and_cap(run_hand, players):
    rng = random.Random(players)
    rake = RakeConfig(percentage=0.05, cap=120, no_flop_no_drop=True)
    for hand in range(150):
        agents = {s: Scripted(random_policy(rng)) for s in range(players)}
        stacks = {s: rng.choice([60, 400, 1500, 5000, 12000]) for s in range(players)}
        r = run_hand(agents, button=hand % players, stacks=stacks, deck=random.Random(hand), rake=rake)
        assert sum(stacks.values()) == sum(r.final_stacks.values()) + r.rake
        assert sum(r.net.values()) == -r.rake
        assert 0 <= r.rake <= 120
        if not r.board:
            assert r.rake == 0
        total_pot = sum(r.winnings.values()) + r.rake
        nominal = rake.hand_rake(total_pot, players, bool(r.board))
        assert nominal - players <= r.rake <= nominal
        assert sum(r.rake_attributed.values()) == pytest.approx(r.rake)
