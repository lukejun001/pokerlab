import json

import pytest

from pokerlab.analysis.reporting import agent_table, export, render_report
from pokerlab.cli import main
from pokerlab.simulation.batch import chunk_plan, run_batch
from pokerlab.simulation.configurations import list_presets, load_config
from pokerlab.simulation.controller import run_chunk, seat_lineup


def small(**overrides):
    base = {"hands": 120, "chunk_size": 40, "workers": 1, "strict": True}
    return load_config(None, {**base, **overrides})


def summaries(result):
    """Everything except wall-clock timing."""
    out = {}
    for agent_id, stats in result.stats.agents.items():
        summary = stats.summary(result.config.big_blind)
        summary.pop("avg_decision_ms")
        out[agent_id] = summary
    return out


def test_chunk_plan():
    assert chunk_plan(10, 4) == [(0, 0, 4), (1, 4, 4), (2, 8, 2)]


def test_batch_runs_and_conserves_chips():
    result = run_batch(small(), progress=False)
    t = result.stats.table
    assert t.hands == 120
    assert t.stalled_hands == t.engine_errors == t.invalid_actions == 0
    assert sum(s.net for s in result.stats.agents.values()) == -t.rake
    assert all(s.hands == 120 for s in result.stats.agents.values())


def test_seed_reproducibility_and_worker_independence():
    a = run_batch(small(seed=9), progress=False)
    b = run_batch(small(seed=9), progress=False)
    c = run_batch(small(seed=9, workers=3), progress=False)
    d = run_batch(small(seed=10), progress=False)
    assert summaries(a) == summaries(b)
    assert {k: v["net_chips"] for k, v in summaries(a).items()} == {k: v["net_chips"] for k, v in summaries(c).items()}
    assert summaries(a) != summaries(d)


def test_matched_rake_comparison_differs_only_by_rake():
    """Same seed with and without rake: identical decisions, so each player's
    net difference equals the rake attributed to them. (Pro-rata attribution of
    a split pot's odd chips can be off by a fraction of a chip.)"""
    no_rake = run_batch(small(seed=3, rake={"percentage": 0.0, "cap": None}), progress=False)
    raked = run_batch(small(seed=3, rake={"percentage": 0.05, "cap": 150}), progress=False)
    assert raked.stats.table.decisions == no_rake.stats.table.decisions
    assert raked.stats.table.rake > 0
    for agent_id, s in raked.stats.agents.items():
        assert abs(s.net + s.rake_attributed - no_rake.stats.agents[agent_id].net) < 1.0
    total_attributed = sum(s.rake_attributed for s in raked.stats.agents.values())
    assert total_attributed == pytest.approx(raked.stats.table.rake)


def test_button_rotation_and_independent_stacks():
    config = small(hands=18, chunk_size=18, hand_log=True)
    records = []
    import pokerlab.simulation.controller as controller

    original = controller.play_hand

    def spy(**kwargs):
        records.append((kwargs["hand_id"], kwargs["button_seat"], dict(kwargs["stacks"])))
        return original(**kwargs)

    controller.play_hand = spy
    try:
        run_chunk(config, 0, 0, 18)
    finally:
        controller.play_hand = original
    assert [button for _, button, _ in records] == [h % 9 for h in range(18)]
    assert all(set(stacks.values()) == {5000} for _, _, stacks in records)


def test_continuous_mode_rebuys_and_conservation():
    config = small(mode="continuous", hands=300, chunk_size=300, rebuy_below_bb=60, rebuy_to_bb=100)
    result = run_batch(config, progress=False)  # the controller asserts session conservation every hand
    t = result.stats.table
    assert t.hands == 300 and t.rebuy_chips > 0
    assert sum(s.rebuys for s in result.stats.agents.values()) > 0


def test_continuous_mode_without_rebuys_sits_out_busted_players():
    config = small(mode="continuous", hands=400, chunk_size=400, rebuy_enabled=False,
                   lineup=[{"type": "Maniac"}, {"type": "Maniac"}, {"type": "CallingStation"}])
    result = run_batch(config, progress=False)
    t = result.stats.table
    assert t.hands == 400 and t.engine_errors == 0
    hands = [s.hands for s in result.stats.agents.values()]
    assert min(hands) < 400 or t.session_resets > 0


def test_seat_assignment_modes():
    rotate = small(seat_assignment="rotate")
    assert seat_lineup(rotate, 1)[1]["id"] == "Hero"
    assert seat_lineup(rotate, 9)[0]["id"] == "Hero"
    shuffled = small(seat_assignment="shuffle")
    assert seat_lineup(shuffled, 4) == seat_lineup(shuffled, 4)
    assert sorted(s["id"] for s in seat_lineup(shuffled, 4)) == sorted(s["id"] for s in shuffled.agent_specs())


def test_player_count_and_unique_ids():
    config = small(players=4)
    assert [s["type"] for s in config.agent_specs()] == ["Experimental", "Nit", "TAG", "LAG"]
    eight_tags = small(lineup=[{"type": "Experimental", "id": "Hero"}] + [{"type": "TAG"}] * 8)
    ids = [s["id"] for s in eight_tags.agent_specs()]
    assert ids[1:3] == ["TAG_1", "TAG_2"] and len(set(ids)) == 9
    with pytest.raises(ValueError):
        small(players=10)
    with pytest.raises(ValueError):
        load_config(None, {"not_a_key": 1})


@pytest.mark.parametrize("preset", [p.stem for p in list_presets()])
def test_every_preset_runs(preset):
    config = load_config(preset, {"hands": 30, "chunk_size": 30, "workers": 1, "strict": True})
    result = run_batch(config, progress=False)
    assert result.stats.table.hands == 30
    assert "POKER AI SIMULATION REPORT" in render_report(result)


def test_export_and_report(tmp_path):
    result = run_batch(small(), progress=False)
    out = export(result, tmp_path / "run")
    data = json.loads((out / "summary.json").read_text())
    assert data["table"]["hands_completed"] == 120
    assert (out / "agents.csv").exists() and (out / "positions.csv").exists()
    table = agent_table(result)
    assert set(table.agent) == {s["id"] for s in result.config.agent_specs()}
    report = render_report(result)
    assert "Net bb/100" in report and "Hands per second" in report


def test_cli_simulate(tmp_path, capsys):
    code = main(["simulate", "--hands", "40", "--players", "6", "--seed", "4", "--quiet",
                 "--chunk-size", "20", "--out", str(tmp_path / "cli")])
    assert code == 0
    out = capsys.readouterr().out
    assert "Hands completed:   40" in out
    assert (tmp_path / "cli" / "summary.json").exists()


def test_cli_strategy_override(capsys):
    assert main(["simulate", "--hands", "20", "--quiet", "--no-export", "--strategy", "archetype:Nit",
                 "--rake", "0"]) == 0
    assert "Rake:              none" in capsys.readouterr().out
