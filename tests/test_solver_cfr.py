"""Solver, strategy store and experimental-agent integration."""

from __future__ import annotations

import numpy as np
import pytest

from pokerlab.engine.rake import RakeConfig
from pokerlab.simulation.batch import run_batch
from pokerlab.simulation.configurations import SimulationConfig
from pokerlab.solver.cfr import Deals, PreflopSolver
from pokerlab.solver.config import solver_config_from_dict
from pokerlab.solver.game import BettingAbstraction, PreflopGame, PreflopTree
from pokerlab.solver.hands import CLASS_COMBOS, CLASS_INDEX
from pokerlab.solver.pipeline import simulation_config
from pokerlab.solver.store import StrategyStore, strategy_key

JAM_FOLD = BettingAbstraction(name="jam-fold", max_sized_raises=0, max_raises=1)


def solve(game: PreflopGame, iterations: int, batch: int = 8192, seed: int = 1) -> PreflopSolver:
    solver = PreflopSolver(PreflopTree(game), batch_size=batch, seed=seed)
    for _ in range(iterations):
        solver.iterate()
    return solver


def freq(table: np.ndarray, action: int) -> float:
    return float((table[:, action] * CLASS_COMBOS).sum() / CLASS_COMBOS.sum())


@pytest.fixture(scope="module")
def push_fold():
    return solve(PreflopGame((1000, 1000), 50, 100, RakeConfig(), JAM_FOLD), 150)


def test_heads_up_push_fold_matches_known_equilibrium(push_fold):
    """10bb heads-up push/fold: published Nash solutions jam ~58% from the SB and call ~37% in the BB."""
    s = push_fold
    jam = s.node_strategy(0)
    call = s.node_strategy(s.tree.by_key["a1000"])
    assert 0.54 < freq(jam, 1) < 0.61
    assert 0.33 < freq(call, 1) < 0.42
    assert jam[CLASS_INDEX["AA"], 1] > 0.99 and call[CLASS_INDEX["AA"], 1] > 0.99
    assert jam[CLASS_INDEX["72o"], 1] < 0.05 and call[CLASS_INDEX["72o"], 1] < 0.05
    assert jam[CLASS_INDEX["K9o"], 1] > 0.95  # clear jam, but not a clear call
    report = s.evaluate(Deals.sample(np.random.default_rng(5), 60_000, 2), Deals.sample(np.random.default_rng(6), 60_000, 2))
    assert report["nash_conv_bb100"] < 15 and report["holdout_nash_conv_bb100"] < 5
    assert sum(report["ev_bb"]) == pytest.approx(0, abs=1e-9)  # zero-sum without rake


def test_solver_is_deterministic():
    game = PreflopGame((2000, 2500, 1500), 50, 100, RakeConfig(), JAM_FOLD)
    a, b = solve(game, 5, batch=2048), solve(game, 5, batch=2048)
    assert np.array_equal(a.strategy_sum, b.strategy_sum)
    c = solve(game, 5, batch=2048, seed=2)
    assert not np.array_equal(a.strategy_sum, c.strategy_sum)


def test_terminal_values_conserve_chips_after_rake():
    rake = RakeConfig(percentage=0.045, cap=30)
    game = PreflopGame((4000, 6000, 2800, 4600, 1600, 8000), 50, 100, rake,
                       BettingAbstraction(max_sized_raises=2, max_raises=4, max_calls_per_raise=1))
    s = PreflopSolver(PreflopTree(game), batch_size=64)
    deals = Deals.sample(np.random.default_rng(0), 500, 6)
    for nd in s.tree.terminals:
        values = s._terminal(nd.id, deals.ranks)
        assert np.allclose(values.sum(axis=1), -nd.rake / 100)


def test_rotating_stacks_changes_the_solution():
    """Test E: the same stack vector in different seats is a different game and solves differently."""
    base = PreflopGame((800, 2500, 3000), 50, 100, RakeConfig(), JAM_FOLD)
    jams = []
    for k in range(3):
        s = solve(base.rotated(k), 60, batch=4096)
        jams.append(freq(s.node_strategy(0), 1))  # button jam frequency
    # Short button (8bb) jams far wider than a deep button facing short blinds.
    assert max(jams) - min(jams) > 0.10


def test_store_round_trip_and_exact_keys(tmp_path, push_fold):
    store = StrategyStore(tmp_path)
    key = store.save(push_fold, {"note": "test"})
    assert key == strategy_key(push_fold.game)
    hit = store.lookup(push_fold.game)
    assert hit.kind == "exact"
    actions, probs = hit.strategy.action_probabilities("", CLASS_INDEX["Q5o"])
    assert actions == ("f", "a1000")
    assert probs.sum() == pytest.approx(1)
    original = push_fold.node_strategy(0)[CLASS_INDEX["Q5o"]]
    assert probs == pytest.approx(original, abs=1e-6)  # mixed strategies preserved
    # Any change to stacks or rake is a miss, never a silent reuse.
    g = push_fold.game
    assert store.lookup(PreflopGame((1000, 1100), 50, 100, g.rake, g.abstraction)) is None
    assert store.lookup(PreflopGame(g.stacks, 50, 100, RakeConfig(percentage=0.05, cap=10), g.abstraction)) is None
    assert store.lookup(PreflopGame(g.stacks, 50, 100, g.rake, BettingAbstraction())) is None
    near = store.nearest(PreflopGame((1000, 1100), 50, 100, g.rake, g.abstraction))
    assert near == [(1.0, key)]  # foundation for approximation: reported with its distance


def _solver_config(stacks_bb):
    return solver_config_from_dict({
        "game": {"active_players": 3, "chips_per_bb": 100, "starting_stacks_bb": dict(zip(("BTN", "SB", "BB"), stacks_bb))},
        "rake": {"percentage": 0.05, "cap_bb": 3, "no_flop_no_drop": True},
        "abstraction": {"name": "jam-fold", "max_sized_raises": 0, "max_raises": 1},
        "simulation": {"hands": 300, "seed": 3, "chunk_size": 150, "workers": 1, "strict": True,
                       "lineup": [{"type": "Experimental", "id": "Hero"}, {"type": "Experimental", "id": "Villain"},
                                  {"type": "TAG"}]},
    }, name="jam3")


def test_experimental_agent_uses_cached_strategy(tmp_path):
    config = _solver_config((12, 20, 30))
    store = StrategyStore(tmp_path)
    s = solve(config.game, 40, batch=4096)
    store.save(s)
    sim = simulation_config(config)
    for entry in sim.lineup:
        if entry["type"] == "Experimental":
            entry["params"]["strategy_params"] = {"abstraction": config.game.abstraction.to_dict(), "store": str(tmp_path)}
    result = run_batch(sim, progress=False)
    hero = result.stats.agents["Hero"].decision_sources
    exact = sum(v for k, v in hero.items() if k.startswith("exact:"))
    assert exact > 0
    assert all(k.startswith(("exact:", "fallback:")) for k in hero)
    assert result.stats.table.invalid_actions == 0
    assert sum(a.net for a in result.stats.agents.values()) == -result.stats.table.rake
    again = run_batch(sim, progress=False)
    assert {k: v.net for k, v in again.stats.agents.items()} == {k: v.net for k, v in result.stats.agents.items()}


def test_agent_falls_back_when_no_strategy_matches(tmp_path):
    config = _solver_config((12, 20, 30))
    store = StrategyStore(tmp_path)
    store.save(solve(config.game, 5, batch=1024))
    other = _solver_config((12, 20, 31))  # one stack differs by 1bb
    sim = simulation_config(other)
    for entry in sim.lineup:
        if entry["type"] == "Experimental":
            entry["params"]["strategy_params"] = {"abstraction": other.game.abstraction.to_dict(), "store": str(tmp_path)}
    result = run_batch(sim, progress=False)
    hero = result.stats.agents["Hero"].decision_sources
    assert not any(k.startswith("exact:") for k in hero)
    assert hero.get("fallback:no_matching_strategy", 0) > 0


def test_simulation_assigns_stacks_by_position():
    config = SimulationConfig(players=6, hands=12, chunk_size=12, workers=1, strict=True,
                              stacks_by_position_bb={"UTG": 100, "HJ": 150, "CO": 70, "BTN": 115, "SB": 40, "BB": 200})
    import pokerlab.simulation.controller as controller

    seen = []
    original = controller.play_hand

    def spy(**kwargs):
        result = original(**kwargs)
        seen.append({result.positions[s]: kwargs["stacks"][s] for s in result.seats})
        return result

    controller.play_hand = spy
    try:
        controller.run_chunk(config, 0, 0, 12)
    finally:
        controller.play_hand = original
    # The simulator labels six-max UTG "LJ".
    expected = {"LJ": 5000, "HJ": 7500, "CO": 3500, "BTN": 5750, "SB": 2000, "BB": 10000}
    assert all(s == expected for s in seen)
