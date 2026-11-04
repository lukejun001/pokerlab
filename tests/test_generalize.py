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


@needs_fork
@pytest.mark.parametrize("mode, expected", [("always", "approximate:"), ("validated", "fallback:no_validated_approximation"),
                                            ("off", "fallback:no_matching_strategy")])
def test_agent_borrows_only_as_configured(tmp_path, mode, expected):
    from pokerlab.simulation.batch import run_batch
    from pokerlab.simulation.configurations import SimulationConfig
    from pokerlab.solver.config import solver_config_from_dict
    from pokerlab.solver.store import StrategyStore

    ab = {"name": "hu-test", "open_raises": [2.5], "sb_open_raises": [], "raise_mults": [3.0], "max_raises": 2}
    store = StrategyStore(tmp_path)
    for stacks in ({"SB": 30, "BB": 60}, {"SB": 50, "BB": 50}):
        cfg = solver_config_from_dict({"game": {"active_players": 2, "chips_per_bb": 100, "starting_stacks_bb": stacks},
                                       "rake": {"percentage": 0.05, "cap_bb": 3}, "strategy": {"preflop_solver": "gtopen"},
                                       "abstraction": ab, "solve": {"iterations": 20, "workers": 2}})
        gtopen.generate(cfg, store, progress=False)
    params = {"strategy": "solver", "strategy_params": {
        "abstraction": {**ab, "engine": "gtopen"}, "store": str(tmp_path), "approximate": mode}}
    sim = SimulationConfig(players=2, hands=60, chunk_size=60, workers=1, strict=True, small_blind=50, big_blind=100,
                           stacks_by_position_bb={"SB": 40, "BB": 55},
                           rake={"percentage": 0.05, "cap": 300, "no_flop_no_drop": True},
                           lineup=[{"type": "Experimental", "id": "Hero", "params": params}, {"type": "TAG"}])
    result = run_batch(sim, progress=False)
    sources = result.stats.agents["Hero"].decision_sources
    assert any(k.startswith(expected) for k in sources), sources
    assert not any(k.startswith("exact:") for k in sources)
    assert result.stats.table.invalid_actions == 0


# --- on-demand solving with warm starts --------------------------------------------------------


def _hu_config(stacks, iterations=30):
    from pokerlab.solver.config import solver_config_from_dict

    ab = {"name": "hu-test", "open_raises": [2.5], "sb_open_raises": [], "raise_mults": [3.0], "max_raises": 2}
    return solver_config_from_dict({"game": {"active_players": 2, "chips_per_bb": 100, "starting_stacks_bb": stacks},
                                    "rake": {"percentage": 0.05, "cap_bb": 3}, "strategy": {"preflop_solver": "gtopen"},
                                    "abstraction": ab, "solve": {"iterations": iterations, "workers": 2}}), ab


@needs_fork
def test_warm_start_from_the_solution_stays_converged():
    """Seeded with its own converged strategy, a solve starts near equilibrium;
    seeded with nothing it starts from uniform play."""
    cfg, _ = _hu_config({"SB": 40, "BB": 55})
    game = cfg.game
    solved = gtopen.run(game, 300, progress=False, threads=2)
    exact = solved["metrics"]["checkpoints"][-1]["nash_conv_bb"]
    order = [solved["class_labels"].index(label) for label in gtopen.CLASS_LABELS]
    profile = {n["history"]: np.asarray(n["strategy"])[:, order] for n in solved["nodes"]}
    warm = gtopen.run(game, 5, check_every=1, progress=False, threads=2,
                      warm_start=gtopen.WarmStart(profile, regret_scale=5, sum_scale=5, iterations=100))
    cold = gtopen.run(game, 5, check_every=1, progress=False, threads=2)
    w = [c["nash_conv_bb"] for c in warm["metrics"]["checkpoints"]]
    c = [c["nash_conv_bb"] for c in cold["metrics"]["checkpoints"]]
    assert warm["metrics"]["warm_start"]["seeded_nodes"] == len(profile)
    assert w[0] < c[0] / 5 and w[-1] < c[-1]
    assert w[-1] < 5 * exact + 0.01


def test_game_round_trips_through_the_queue_format():
    from pokerlab.solver.ondemand import game_from_dict, game_to_dict
    from pokerlab.solver.store import strategy_key

    g = game([100, 150, 70, 115, 40, 200])
    assert strategy_key(game_from_dict(game_to_dict(g))) == strategy_key(g)


@needs_fork
def test_on_demand_solve_warm_starts_from_the_library(tmp_path):
    from pokerlab.solver.ondemand import PendingSolves, solve_on_demand
    from pokerlab.solver.store import StrategyStore

    store = StrategyStore(tmp_path)
    cfg, _ = _hu_config({"SB": 30, "BB": 60})
    gtopen.generate(cfg, store, progress=False)
    target, _ = _hu_config({"SB": 40, "BB": 55})
    queue = PendingSolves(store)
    queue.add(target.game)
    queue.add(target.game)
    assert [r["count"] for r in queue.items()] == [2]
    m = solve_on_demand(store, target.game, iterations=20, threads=2)  # short budget: warm by default
    assert m["on_demand"]["warm_started"] and m["warm_start"]["seeded_nodes"] > 0
    assert store.lookup(target.game) is not None and queue.items() == []
    assert solve_on_demand(store, target.game)["cached"]
    other, _ = _hu_config({"SB": 45, "BB": 50})
    m = solve_on_demand(store, other.game, iterations=80, threads=2)  # long budget: cold by default
    assert not m["on_demand"]["warm_started"] and m["warm_start"] is None


@needs_fork
@pytest.mark.parametrize("mode", ["queue", "solve"])
def test_agent_on_demand_modes(tmp_path, mode):
    from pokerlab.simulation.batch import run_batch
    from pokerlab.simulation.configurations import SimulationConfig
    from pokerlab.solver.ondemand import PendingSolves
    from pokerlab.solver.store import StrategyStore

    store = StrategyStore(tmp_path)
    cfg, ab = _hu_config({"SB": 30, "BB": 60}, iterations=20)
    gtopen.generate(cfg, store, progress=False)
    params = {"strategy": "solver", "strategy_params": {
        "abstraction": {**ab, "engine": "gtopen"}, "store": str(tmp_path), "approximate": "off",
        "on_demand": mode, "on_demand_iterations": 20, "on_demand_threads": 2}}
    sim = SimulationConfig(players=2, hands=20, chunk_size=20, workers=1, strict=True, small_blind=50, big_blind=100,
                           stacks_by_position_bb={"SB": 40, "BB": 55},
                           rake={"percentage": 0.05, "cap": 300, "no_flop_no_drop": True},
                           lineup=[{"type": "Experimental", "id": "Hero", "params": params}, {"type": "TAG"}])
    result = run_batch(sim, progress=False)
    sources = result.stats.agents["Hero"].decision_sources
    pending = PendingSolves(store).items()
    if mode == "queue":
        assert any(k.startswith("fallback:") for k in sources) and not any(k.startswith("exact:") for k in sources)
        assert len(pending) >= 1
    else:
        assert any(k.startswith("exact:") for k in sources) and pending == []
    assert result.stats.table.invalid_actions == 0
