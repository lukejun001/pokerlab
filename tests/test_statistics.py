from conftest import Scripted, passive

from pokerlab.analysis.statistics import StatsCollector, action_flags
from pokerlab.engine.actions import ActionType as T
from pokerlab.engine.observations import ActionRecord, Street

P, F = Street.PREFLOP, Street.FLOP


def rec(seat, kind, street=P, amount=0):
    return ActionRecord(seat, street, kind, amount)


def test_vpip_pfr_threebet_and_fold_to_threebet():
    # Seats: 0 SB, 1 BB, 2 UTG, 3 BTN.
    actions = (
        rec(2, T.RAISE, amount=125),   # UTG opens
        rec(3, T.RAISE, amount=400),   # BTN 3-bets
        rec(0, T.FOLD),                # SB folds facing 2 raises: no 3-bet opportunity
        rec(1, T.FOLD),
        rec(2, T.FOLD),                # opener folds to the 3-bet
    )
    f = action_flags(actions, range(4))
    assert f[2].vpip and f[2].pfr and f[2].fold_to_3bet_opportunity and f[2].fold_to_3bet
    assert f[3].vpip and f[3].pfr and f[3].threebet_opportunity and f[3].threebet
    assert not f[0].threebet_opportunity and not f[1].threebet_opportunity
    assert not f[0].vpip and not f[1].vpip


def test_limp_and_big_blind_check_are_not_pfr():
    actions = (rec(2, T.CALL, amount=50), rec(0, T.CALL, amount=25), rec(1, T.CHECK),
               rec(0, T.CHECK, F), rec(1, T.BET, F, 100), rec(2, T.CALL, F, 100), rec(0, T.FOLD, F))
    f = action_flags(actions, range(3))
    assert f[2].vpip and not f[2].pfr
    assert f[0].vpip and not f[1].vpip
    assert (f[1].postflop_aggressive, f[2].postflop_calls, f[0].postflop_folds, f[0].postflop_checks) == (1, 1, 1, 1)


def test_threebet_opportunity_when_facing_single_raise():
    actions = (rec(2, T.RAISE, amount=125), rec(3, T.CALL, amount=125), rec(0, T.FOLD), rec(1, T.CALL, amount=75))
    f = action_flags(actions, range(4))
    assert f[3].threebet_opportunity and not f[3].threebet
    assert f[0].threebet_opportunity and f[1].threebet_opportunity
    assert not f[2].threebet_opportunity


def test_collector_net_rake_and_positions(run_hand):
    agents = {s: Scripted(passive) for s in range(3)}
    from pokerlab.engine.rake import RakeConfig

    r = run_hand(agents, button=2, rake=RakeConfig(percentage=0.04, cap=150))
    stats = StatsCollector(big_blind=50)
    stats.record(r, {s: (f"P{s}", "Test") for s in range(3)})
    stats.record(r, {s: (f"P{s}", "Test") for s in range(3)})
    total_net = sum(s.net for s in stats.agents.values())
    assert total_net == -2 * r.rake == -stats.table.rake
    winner = max(stats.agents.values(), key=lambda s: s.net)
    summary = winner.summary(50)
    assert summary["hands"] == 2 and summary["vpip"] == 1.0 or summary["vpip"] == 0.0
    assert summary["gross_bb100"] > summary["net_bb100"]
    assert sum(b.hands for b in winner.by_position.values()) == 2

    merged = StatsCollector(big_blind=50)
    merged.merge(stats)
    merged.merge(stats)
    assert merged.table.hands == 4
    assert merged.agents[winner.agent_id].net == 2 * winner.net
