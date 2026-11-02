import pytest

from pokerlab.agents import AGENT_TYPES, create_agent
from pokerlab.agents.heuristics import PreflopSpot, analyze_hand, equity_vs_ranges
from pokerlab.agents.opponents import ARCHETYPES, TAG
from pokerlab.engine.cards import parse_cards as P
from pokerlab.simulation.batch import run_batch
from pokerlab.simulation.configurations import load_config
from pokerlab.strategies.base import AgentStrategy, Strategy, load_strategy
from pokerlab.strategies.heuristic import EquityStrategy


def test_eight_distinct_archetypes():
    assert set(ARCHETYPES) == {"Nit", "TAG", "LAG", "CallingStation", "Maniac", "LoosePassive", "FitOrFold", "Balanced"}
    postflop_methods = {cls.postflop for cls in ARCHETYPES.values()}
    assert len(postflop_methods) == 8  # each archetype has its own postflop logic


def test_parameters_are_configurable():
    agent = TAG(seed=1, aggression=0.9, open_range={"late": 0.5})
    assert agent.p["aggression"] == 0.9
    assert agent.p["open_range"]["late"] == 0.5 and agent.p["open_range"]["early"] == 0.11
    with pytest.raises(ValueError):
        TAG(not_a_param=1)


@pytest.mark.parametrize(
    "hole, board, tier",
    [
        ("AsKd", "Kc7h2s", "strong"),  # top pair top kicker
        ("QcQd", "Jh7h2s", "strong"),  # overpair
        ("5c5d", "5h9sKd", "monster"),  # set
        ("7c8c", "9cTd2h", "nothing"),  # draw only
        ("2c3d", "AhAdAs", "nothing"),  # board trips, nothing of our own
        ("6c6d", "KhQsJs", "weak"),  # underpair
    ],
)
def test_hand_features(hole, board, tier):
    assert analyze_hand(P(hole), P(board)).tier == tier


def test_draw_detection():
    f = analyze_hand(P("7c8c"), P("9cTd2h"))
    assert f.straight_draw == "oesd" and f.outs >= 8
    f = analyze_hand(P("AhKh"), P("Qh7h2c"))
    assert f.flush_draw and f.nut_flush_draw


def test_equity_vs_narrow_ranges_is_lower():
    import random

    rng = random.Random(0)
    hole, board = P("JcTc"), ()
    vs_any = equity_vs_ranges(hole, board, [1.0], 1500, rng)
    vs_tight = equity_vs_ranges(hole, board, [0.08], 1500, rng)
    assert vs_tight < vs_any - 0.05


def test_strategy_loading():
    assert isinstance(load_strategy("equity"), EquityStrategy)
    tag = load_strategy("archetype:TAG")
    assert isinstance(tag, AgentStrategy) and tag.agent.archetype == "TAG"
    assert isinstance(load_strategy("pokerlab.strategies.heuristic:EquityStrategy"), Strategy)
    with pytest.raises(ValueError):
        load_strategy("archetype:Nobody")
    with pytest.raises(TypeError):
        load_strategy("pokerlab.agents.opponents:TAG")


def test_create_agent_registry():
    assert set(AGENT_TYPES) == set(ARCHETYPES) | {"Experimental"}
    hero = create_agent("Experimental", "Hero", seed=1, params={"strategy": "archetype:Maniac"})
    assert hero.strategy.agent.archetype == "Maniac"
    with pytest.raises(ValueError):
        create_agent("Shark", "x")


def test_all_agents_only_make_legal_actions():
    """Strict mode raises on any illegal action or agent exception."""
    config = load_config("mixed_table", {"hands": 400, "chunk_size": 200, "workers": 1, "strict": True})
    result = run_batch(config, progress=False)
    assert result.stats.table.invalid_actions == 0


@pytest.mark.parametrize("players", [2, 3, 6])
def test_agents_at_short_tables(players):
    config = load_config("default", {"hands": 150, "chunk_size": 150, "workers": 1, "strict": True, "players": players})
    assert run_batch(config, progress=False).stats.table.hands == 150


def test_archetype_styles_are_distinct():
    """Observed tendencies line up with each archetype's description."""
    config = load_config("mixed_table", {"hands": 2500, "chunk_size": 500, "workers": 1})
    result = run_batch(config, progress=False)
    s = {a: st.summary(50) for a, st in result.stats.agents.items()}
    assert s["Nit"]["vpip"] < s["TAG"]["vpip"] < s["LAG"]["vpip"] < s["Maniac"]["vpip"]
    assert s["Maniac"]["pfr"] > 0.3 and s["Nit"]["pfr"] < 0.08
    assert s["CallingStation"]["vpip"] > 0.3 and s["CallingStation"]["pfr"] < 0.08
    assert s["LoosePassive"]["vpip"] > 0.25 and s["LoosePassive"]["aggression_frequency"] < 0.3
    assert s["Maniac"]["aggression_frequency"] > s["TAG"]["aggression_frequency"] > s["CallingStation"]["aggression_frequency"]
    assert s["CallingStation"]["wtsd"] > s["FitOrFold"]["wtsd"]
    assert s["LAG"]["threebet"] > s["Nit"]["threebet"]


def test_preflop_spot_parsing():
    from pokerlab.engine.actions import ActionType as T
    from pokerlab.engine.observations import ActionRecord, Street

    class Obs:
        seat = 5
        current_bet = 400
        history = (ActionRecord(2, Street.PREFLOP, T.CALL, 50), ActionRecord(3, Street.PREFLOP, T.RAISE, 150),
                   ActionRecord(4, Street.PREFLOP, T.RAISE, 400))

    spot = PreflopSpot.from_obs(Obs)
    assert (spot.raises, spot.limpers, spot.situation, spot.last_raiser) == (2, 1, "facing_3bet", 4)
