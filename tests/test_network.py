"""Preflop strategy network: features, model shapes, and the agent's network mode."""

from __future__ import annotations

import numpy as np
import pytest

from pokerlab.solver import gtopen
from pokerlab.solver import network as net

torch = pytest.importorskip("torch")

AB = {"engine": "gtopen", "name": "gtopen-6max-v1", "open_raises": [2.5], "sb_open_raises": [3.0],
      "raise_mults": [3.0], "max_raises": 3, "add_allin": True, "allin_threshold": 0.5, "limp": False,
      "realization": "balanced", "multiway_equity_model": "coupled_deck_v1"}


def game(stacks_bb):
    from pokerlab.engine.rake import RakeConfig
    from pokerlab.solver.game import PreflopGame

    return PreflopGame(tuple(int(s * 5000) for s in stacks_bb), 2500, 5000, RakeConfig(percentage=0.045, cap=300),
                       gtopen.GTOpenAbstraction())


@pytest.mark.parametrize("arch", ["bilinear", "mlp"])
def test_profile_covers_the_tree_with_distributions(arch):
    from pokerlab.solver.generalize import python_tree

    g = game([63, 82, 206, 65, 72, 88])
    tree = python_tree(g)
    model = net.build_model(hidden=32, dim=16, layers=1, arch=arch)
    profile = net.profile_for(model, g, tree)
    assert set(profile) == set(tree)
    for h in list(tree)[:200]:
        t = profile[h]
        assert t.shape == (len(tree[h]), 169) and np.allclose(t.sum(axis=0), 1, atol=1e-4)


def test_features_are_fixed_length_for_any_table_size():
    from pokerlab.solver.generalize import python_tree

    for stacks in ([40, 60], [30, 90, 150], [63, 82, 206, 65, 72, 88]):
        g = game(stacks)
        tree = python_tree(g)
        feats, masks = net.encode_nodes(g, list(tree), list(tree.values()))
        assert feats.shape == (len(tree), net.NUM_FEATURES) and np.isfinite(feats).all()


@pytest.mark.parametrize("residual", [False, True])
def test_agent_plays_unsolved_stacks_from_the_network(tmp_path, residual):
    from pokerlab.simulation.batch import run_batch
    from pokerlab.simulation.configurations import SimulationConfig

    path = tmp_path / "net.pt"
    net.save_model(net.build_model(hidden=32, dim=16, layers=1, arch="mlp", residual=residual), path)
    params = {"strategy": "solver", "strategy_params": {"abstraction": AB, "approximate": "off", "network": str(path),
                                                        "store": str(tmp_path / "store")}}
    sim = SimulationConfig(players=6, hands=12, chunk_size=12, workers=1, strict=True, small_blind=2500,
                           big_blind=5000, stacks_by_position_bb={"UTG": 63, "HJ": 82, "CO": 206, "BTN": 65,
                                                                  "SB": 72, "BB": 88},
                           rake={"percentage": 0.045, "cap": 300, "no_flop_no_drop": True},
                           lineup=[{"type": "Experimental", "id": "Hero", "params": params}, {"type": "TAG"},
                                   {"type": "LAG"}, {"type": "CallingStation"}, {"type": "Nit"}, {"type": "Balanced"}])
    result = run_batch(sim, progress=False)
    sources = result.stats.agents["Hero"].decision_sources
    assert any(k.startswith("network:") for k in sources), sources
    assert result.stats.table.invalid_actions == 0
