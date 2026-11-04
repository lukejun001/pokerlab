"""Heads-up postflop search (pokerlab/postflop) and the preflop network's agent hooks."""

from __future__ import annotations

import numpy as np
import pytest

from pokerlab.engine.cards import parse_cards
from pokerlab.postflop import search as S

needs_postflop = pytest.mark.skipif(not S.BINARY.exists(), reason="postflop-cli not built")


def test_class_combos_cover_the_deck():
    sizes = [len(c) for c in S.CLASS_COMBOS]
    assert sum(sizes) == 1326 and set(sizes) == {4, 6, 12}
    reach = np.zeros(169)
    reach[0] = 1.0  # first class
    board = parse_cards("2c7d9h")
    r = S.range_from_classes(reach, board)
    assert all(a not in board and b not in board for a, b in r)


def test_action_matching_by_role_and_size():
    view = {"actions": [{"kind": "fold", "amount": 0}, {"kind": "call", "amount": 10},
                        {"kind": "raise", "amount": 40}, {"kind": "raise", "amount": 200}]}
    assert S.match_action(view, "call", None, False) == 1
    assert S.match_action(view, "raise", 50.0, False) == 2
    assert S.match_action(view, "raise", 150.0, False) == 3
    assert S.match_action(view, "raise", 30.0, True) == 3  # all-in maps to the largest
    assert S.match_action({"actions": [{"kind": "check", "amount": 0}, {"kind": "bet", "amount": 5}]},
                          "call", None, False) == 0


@needs_postflop
def test_turn_solve_exports_the_street_and_next_ranges():
    board = parse_cards("Ks7h2d4c")
    oop = S.range_from_classes(np.ones(169), board)
    ip = dict(oop)
    settings = S.SearchSettings(threads=2, budget_s={"flop": 2, "turn": 3, "river": 2})
    solve = S.solve_street("turn", board, (oop, ip), pot_bb=10.0, stack_bb=50.0, rake_pct=0.0,
                           rake_cap_bb=0.0, settings=settings)
    root = solve.view(())
    assert root["node_type"] == "action" and root["player"] == 0
    assert solve.metrics["iterations"] > 0
    check_check = (0, 0)
    assert solve.view(check_check)["node_type"] == "chance"
    reach = solve.reach(check_check, 1)
    assert 0 < len(reach) <= len(ip) and all(0 <= w <= 1 for w in reach.values())
    strat = S.hand_strategy(root, 0, parse_cards("AsAd"))
    assert strat is not None and abs(strat.sum() - 1) < 1e-3


@needs_postflop
def test_heads_up_agent_searches_postflop(tmp_path):
    from pokerlab.simulation.batch import run_batch
    from pokerlab.simulation.configurations import SimulationConfig
    from pokerlab.solver import gtopen
    from pokerlab.solver.config import solver_config_from_dict
    from pokerlab.solver.store import StrategyStore

    ab = {"name": "hu-test", "open_raises": [2.5], "sb_open_raises": [], "raise_mults": [3.0], "max_raises": 2}
    store = StrategyStore(tmp_path)
    cfg = solver_config_from_dict({"game": {"active_players": 2, "chips_per_bb": 100,
                                            "starting_stacks_bb": {"SB": 40, "BB": 55}},
                                   "rake": {"percentage": 0.05, "cap_bb": 3}, "strategy": {"preflop_solver": "gtopen"},
                                   "abstraction": ab, "solve": {"iterations": 30, "workers": 2}})
    gtopen.generate(cfg, store, progress=False)
    params = {"strategy": "solver", "strategy_params": {
        "abstraction": {**ab, "engine": "gtopen"}, "store": str(tmp_path), "approximate": "off",
        "postflop": "search", "search_budget": {"flop": 2, "turn": 1, "river": 1}, "search_threads": 2}}
    sim = SimulationConfig(players=2, hands=12, chunk_size=12, workers=1, strict=True, small_blind=50, big_blind=100,
                           stacks_by_position_bb={"SB": 40, "BB": 55},
                           rake={"percentage": 0.05, "cap": 300, "no_flop_no_drop": True},
                           lineup=[{"type": "Experimental", "id": "Hero", "params": params},
                                   {"type": "CallingStation"}])
    result = run_batch(sim, progress=False)
    sources = result.stats.agents["Hero"].decision_sources
    assert any(k.startswith("search:") for k in sources), sources
    assert result.stats.table.invalid_actions == 0
