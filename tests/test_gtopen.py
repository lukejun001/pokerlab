"""The asymmetric-stack GTOpen fork, checked against PokerKit.

GTOpen's exported trees (``preflop-cli`` with ``dump_terminals``) are replayed
node by node in real PokerKit states, and every terminal's pot structure is
compared with pokerlab's PokerKit-validated pot builder and with PokerKit's
actual payouts for concrete cards. Skipped when the fork is not built.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pokerlab.engine.cards import evaluate
from pokerlab.engine.rake import RakeConfig
from pokerlab.solver import gtopen
from pokerlab.solver.game import PreflopGame, action_amount, action_kind, build_pots, preflop_order_from_pokerkit
from test_solver_game import apply_pk, play_terminal, pokerkit_state

pytestmark = pytest.mark.skipif(not gtopen.BINARY.exists(), reason="GTOpen fork not built")

BB, SB = 5000, 2500
RAKE = RakeConfig(percentage=0.045, cap=300)
ASYM = (100, 150, 70, 115, 40, 200)


def game(stacks_bb, max_raises=3, rake=RAKE, **ab) -> PreflopGame:
    abstraction = gtopen.GTOpenAbstraction(max_raises=max_raises, **ab)
    return PreflopGame(tuple(int(s * BB) for s in stacks_bb), SB, BB, rake, abstraction)


def dump(g: PreflopGame, per_seat=True):
    return gtopen.run(g, iterations=0, dump_terminals=True, per_seat=per_seat, progress=False, threads=2)


class Node:
    """Minimal stand-in for ``play_terminal`` (it needs only the history)."""

    def __init__(self, history: str) -> None:
        self.history = tuple(history.split("-")) if history else ()


def check_action_nodes(g, out, limit, rng):
    nodes = out["nodes"] if len(out["nodes"]) <= limit else rng.sample(out["nodes"], limit)
    order = preflop_order_from_pokerkit(g.n)
    for nd in nodes:
        pk = pokerkit_state(g)
        for token in filter(None, nd["history"].split("-")):
            apply_pk(pk, token)
        assert pk.street_index == 0 and pk.actor_index is not None, nd["history"]
        assert order[pk.actor_index] == nd["actor"], nd["history"]
        owed = pk.checking_or_calling_amount
        tokens = nd["actions"]
        assert ("f" in tokens) == bool(owed) and ("x" in tokens) == (not owed), nd["history"]
        raises = [t for t in tokens if action_kind(t) in "ra"]
        if not pk.can_complete_bet_or_raise_to():
            assert not raises, (nd["history"], tokens)  # e.g. a short all-in did not reopen betting
        for t in raises:
            assert pk.can_complete_bet_or_raise_to(action_amount(t)), (nd["history"], t)
            if action_kind(t) == "a":
                assert action_amount(t) == pk.max_completion_betting_or_raising_to_amount
    return len(nodes)


def check_terminals(g, out, limit, rng):
    terms = out["terminals"] if len(out["terminals"]) <= limit else rng.sample(out["terminals"], limit)
    for t in terms:
        pk = pokerkit_state(g)
        for token in filter(None, t["history"].split("-")):
            apply_pk(pk, token)
        assert pk.actor_index is None or pk.street_index != 0, t["history"]
        bets = [int(round(x * BB)) for x in t["invested_bb"]]
        contrib, pots = build_pots(bets, t["live"], g.rake, flop_dealt=t["kind"] == "flop")
        if t["kind"] == "flop":
            assert [int(round(c * BB)) for c in t["contributions_bb"]] == contrib, t["history"]
            assert [tuple(p["eligible"]) for p in t["pots"]] == [p.eligible for p in pots], t["history"]
            for ours, theirs in zip(pots, t["pots"]):
                assert abs(ours.amount - theirs["amount_bb"] * BB) <= 1.0 + 1e-6, t["history"]  # per-pot rake rounding
        else:
            winner = t["live"].index(True)
            assert t["pots"][0]["eligible"] == [winner]
            assert abs(t["pots"][0]["amount_bb"] * BB - sum(bets)) < 1e-6  # no flop, no drop
    return terms


@pytest.fixture(scope="module")
def asym_dumps():
    return {r: dump(game(ASYM, max_raises=r)) for r in (3, 4)}


@pytest.mark.parametrize("max_raises", [3, 4])
def test_asymmetric_tree_matches_pokerkit(asym_dumps, max_raises):
    """Test B: every sampled node of the fork's tree is a legal PokerKit state."""
    g = game(ASYM, max_raises=max_raises)
    out = asym_dumps[max_raises]
    rng = random.Random(max_raises)
    assert check_action_nodes(g, out, 2500, rng) > 0
    check_terminals(g, out, 2500, rng)


def test_unequal_all_ins_pay_like_pokerkit(asym_dumps):
    """Test C: three or more players all-in for different amounts; payouts from
    the fork's pot structure equal PokerKit's for concrete cards."""
    g = game(ASYM, max_raises=4)
    out = asym_dumps[4]
    multi = [t for t in out["terminals"] if t["kind"] == "flop" and len(t["pots"]) >= 3]
    assert multi, "no three-pot terminals"
    rng = random.Random(3)
    for t in rng.sample(multi, min(40, len(multi))):
        for seed in range(3):
            net, ranks, result = play_terminal(g, Node(t["history"]), 100 * seed + 7)
            won = [0.0] * g.n
            for pot in t["pots"]:
                best = max(ranks[i] for i in pot["eligible"])
                winners = [i for i in pot["eligible"] if ranks[i] == best]
                for i in winners:
                    won[i] += pot["amount_bb"] * BB / len(winners)
            ours = [w - c * BB for w, c in zip(won, t["contributions_bb"])]
            tolerance = len(t["pots"]) + 1  # per-pot rake rounding and odd chips
            assert all(abs(a - b) <= tolerance for a, b in zip(ours, net)), (t["history"], ours, net)


def test_randomized_stack_vectors_match_pokerkit():
    """Property test: random six-max stack vectors (2-250 bb)."""
    rng = random.Random(17)
    for _ in range(3):
        stacks = [rng.randint(2, 250) for _ in range(6)]
        g = game(stacks, max_raises=4)
        out = dump(g)
        check_action_nodes(g, out, 400, rng)
        check_terminals(g, out, 400, rng)


def test_heads_up_matches_pokerkit():
    g = game((60, 90), max_raises=4, sb_open_raises=())
    out = dump(g)
    rng = random.Random(1)
    check_action_nodes(g, out, 10_000, rng)
    check_terminals(g, out, 10_000, rng)


def test_equal_stacks_reproduce_original_engine():
    """Test A: six equal stacks through the per-seat path vs GTOpen's original
    single-stack path: same tree, and strategies after a short solve agree
    within a measured tolerance."""
    g = game((100,) * 6)
    a, b = dump(g, per_seat=False), dump(g, per_seat=True)
    assert [n["history"] for n in a["nodes"]] == [n["history"] for n in b["nodes"]]
    assert [n["actions"] for n in a["nodes"]] == [n["actions"] for n in b["nodes"]]
    assert a["terminals"] == b["terminals"]
    sa = gtopen.run(g, iterations=3, per_seat=False, progress=False)
    sb = gtopen.run(g, iterations=3, per_seat=True, progress=False)
    diff = max(float(np.abs(np.asarray(x["strategy"]) - np.asarray(y["strategy"])).max())
               for x, y in zip(sa["nodes"], sb["nodes"]))
    conv = [r["metrics"]["checkpoints"][-1]["nash_conv_bb"] for r in (sa, sb)]
    assert diff <= 1e-5, diff  # measured: identical up to float noise from parallel summation
    assert abs(conv[0] - conv[1]) <= 1e-6


def test_rake_sensitivity_in_the_fork():
    """Test D: zero rake leaves the tree unchanged and every terminal's pots sum to the contributions."""
    raked, free = dump(game(ASYM)), dump(game(ASYM, rake=RakeConfig()))
    assert [t["history"] for t in raked["terminals"]] == [t["history"] for t in free["terminals"]]
    for r, f in zip(raked["terminals"], free["terminals"]):
        total = sum(f["contributions_bb"]) if f["kind"] == "flop" else sum(f["invested_bb"])
        assert sum(p["amount_bb"] for p in f["pots"]) == pytest.approx(total)
        rake = total - sum(p["amount_bb"] for p in r["pots"])
        assert -1e-9 <= rake <= 0.06 + 1e-9
        if r["kind"] == "fold":
            assert rake == pytest.approx(0)


def test_rotated_stacks_are_different_games():
    """Test E: the same stack vector in other seats gives other trees and keys."""
    from pokerlab.solver.store import strategy_key

    base = game(ASYM)
    keys, roots = set(), set()
    for k in range(6):
        g = base.rotated(k)
        keys.add(strategy_key(g))
        out = dump(g)
        roots.add(tuple(out["nodes"][0]["actions"]))
    assert len(keys) == 6 and len(roots) == 6
