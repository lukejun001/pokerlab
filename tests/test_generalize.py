"""Approximate strategies across stack vectors (pokerlab/solver/generalize.py)."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from pokerlab.solver import gtopen
from pokerlab.solver.game import BettingState, PreflopGame, action_amount, action_kind
from pokerlab.solver.generalize import (
    calibrate,
    distance,
    features,
    gtopen_tokens,
    map_token,
    node_distance,
    replay,
)

needs_fork = pytest.mark.skipif(not gtopen.BINARY.exists(), reason="GTOpen fork not built")


def game(stacks_bb) -> PreflopGame:
    from pokerlab.engine.rake import RakeConfig

    return PreflopGame(tuple(int(s * 5000) for s in stacks_bb), 2500, 5000,
                       RakeConfig(percentage=0.045, cap=300), gtopen.GTOpenAbstraction())


@needs_fork
def test_python_sizing_matches_the_fork():
    rng = random.Random(5)
    for _ in range(3):
        g = game([rng.randint(20, 250) for _ in range(6)])
        tree = gtopen.tree(g)
        for history, actions in tree.items():
            state = BettingState.initial(g)
            for t in filter(None, history.split("-")):
                state.apply(action_kind(t), action_amount(t))
            assert gtopen_tokens(g, state) == actions, history


def test_replay_rejects_off_tree_lines():
    g = game([100, 150, 70, 115, 40, 200])
    assert replay(g, ["r12500"]) is not None
    assert replay(g, ["r15000"]) is None  # UTG has no 3bb open in this abstraction


def test_token_roles():
    src = ["f", "c", "r37500", "a500000"]
    dst = ["f", "c", "a200000"]  # a short stack: the 3-bet size is already all-in
    assert map_token("r37500", src, dst) == "a200000"
    assert map_token("a500000", src, dst) == "a200000"
    assert map_token("c", src, dst) == "c"
    assert map_token("r37500", src, ["f", "c"]) is None
    assert map_token("r37500", src, ["f", "c"], relaxed=True) == "c"
    assert map_token("f", ["f", "c"], ["x", "r12500"], relaxed=True) == "x"


def test_effective_stack_distances():
    a = [100, 150, 70, 115, 40, 200]
    rot = [200, 100, 150, 70, 115, 40]
    assert distance(a, a) == 0 and len(features(a)) == 15
    # SB vs BB: effective 40bb in both, so the decision distance is 0 there
    live = [False, False, False, False, True, True]
    assert node_distance(a, rot, 4, live) == pytest.approx(0)
    assert node_distance(a, rot, 0, [True] * 6) > 0.5


@needs_fork
def test_self_translation_is_exact():
    from pokerlab.solver.config import load_solver_config
    from pokerlab.solver.generalize import translate_node
    from pokerlab.solver.store import StrategyStore, strategy_key

    g = load_solver_config("6max_gtopen_asymmetric").game
    strategy = StrategyStore().load(strategy_key(g))
    if strategy is None:
        pytest.skip("asymmetric strategy not in the store")
    for history in list(strategy.nodes)[:400]:
        table = translate_node(g, history.split("-") if history else [], strategy)
        assert np.allclose(table, strategy.node_table(history)[1])


def test_calibration_picks_the_radius_that_meets_the_threshold():
    rows = [{"k": 1, "node_distance_p90": d, "nash_conv_bb100": nc}
            for d, nc in [(0.1, 0.4), (0.2, 0.8), (0.3, 0.9), (0.4, 3.5), (0.5, 9.0)]]
    cal = calibrate(rows, "fam", 1, threshold_bb100=1.0)
    assert cal.radius == 0.3 and cal.within_radius_max_bb100 == 0.9
    assert cal.coverage == pytest.approx(0.6)
    strict = calibrate(rows, "fam", 1, threshold_bb100=0.1)
    assert strict.radius == 0.0 and strict.coverage == 0.0


def test_library_games_share_a_family():
    from pokerlab.solver.store import family_key, strategy_key

    a, b = game([100] * 6), game([50, 60, 70, 80, 90, 100])
    assert family_key(a) == family_key(b) and strategy_key(a) != strategy_key(b)
    c = dataclasses.replace(b, small_blind=2000)
    assert family_key(c) != family_key(a)
