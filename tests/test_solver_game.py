"""The asymmetric-stack preflop model, checked against PokerKit.

PokerKit is the independent oracle: every abstract node is replayed in a real
PokerKit state (legal actions, call amounts, raise bounds), and every terminal
payout is compared with what our simulator's PokerKit engine actually pays
for concrete cards, including side pots and rake.
"""

from __future__ import annotations

import itertools
import random

import pytest
from pokerkit import Mode

from pokerlab.engine.actions import Action
from pokerlab.engine.cards import PK_CARDS, evaluate
from pokerlab.engine.game import AUTOMATIONS, _HoldemGame, play_hand
from pokerlab.engine.observations import GameInfo, Street
from pokerlab.engine.rake import RakeConfig, make_rake_function
from pokerlab.solver.game import (
    BettingAbstraction,
    BettingState,
    PreflopGame,
    PreflopTree,
    action_amount,
    action_kind,
    build_pots,
    preflop_order_from_pokerkit,
)

SB, BB = 2500, 5000
RAKE = RakeConfig(percentage=0.045, cap=300)  # 4.5%, cap 0.06 bb
ASYM = (100, 150, 70, 115, 40, 200)
EQUAL = (100,) * 6
WIDE = BettingAbstraction(max_sized_raises=2, max_raises=4, max_calls_per_raise=1)


def game(stacks_bb, rake=RAKE, abstraction=WIDE, bb=BB, sb=SB) -> PreflopGame:
    return PreflopGame(tuple(int(s * bb) for s in stacks_bb), sb, bb, rake, abstraction)


# --- PokerKit replay helpers ------------------------------------------------------


def pk_index(n: int) -> list[int]:
    """Preflop order index -> PokerKit index."""
    order = preflop_order_from_pokerkit(n)
    inverse = [0] * n
    for pk, p in enumerate(order):
        inverse[p] = pk
    return inverse


def pokerkit_state(g: PreflopGame, rake: RakeConfig | None = None):
    n = g.n
    to_pk = pk_index(n)
    stacks = [0] * n
    for p, chips in enumerate(g.stacks):
        stacks[to_pk[p]] = chips
    state = _HoldemGame.create_state(AUTOMATIONS, True, 0, (g.small_blind, g.big_blind), g.big_blind,
                                     tuple(stacks), n, mode=Mode.CASH_GAME, rake=make_rake_function(rake or g.rake))
    for k in range(n):
        state.deal_hole((PK_CARDS[2 * k], PK_CARDS[2 * k + 1]))
    return state


def apply_pk(state, token: str) -> None:
    kind = action_kind(token)
    if kind == "f":
        state.fold()
    elif kind in "xc":
        state.check_or_call()
    else:
        state.complete_bet_or_raise_to(action_amount(token))


def assert_node_matches(g: PreflopGame, tree: PreflopTree, node, state) -> None:
    n = g.n
    order = preflop_order_from_pokerkit(n)
    if node.is_terminal:
        assert state.actor_index is None or state.street_index != 0, node.key
        return
    assert state.street_index == 0
    assert order[state.actor_index] == node.actor, node.key
    owed = state.checking_or_calling_amount
    assert ("f" in node.actions) == bool(owed), node.key
    assert ("x" in node.actions) == (not owed), node.key
    bstate = replay(g, node.history)
    assert bstate.to_call(node.actor) == owed, node.key
    bounds = bstate.raise_bounds(node.actor, g.big_blind)
    assert state.can_complete_bet_or_raise_to() == (bounds is not None), node.key
    if bounds:
        assert bounds == (state.min_completion_betting_or_raising_to_amount,
                          state.max_completion_betting_or_raising_to_amount), node.key
    for token in node.actions:
        if action_kind(token) in "ra":
            assert state.can_complete_bet_or_raise_to(action_amount(token)), (node.key, token)
        if action_kind(token) == "a":
            assert action_amount(token) == state.max_completion_betting_or_raising_to_amount


def replay(g: PreflopGame, history) -> BettingState:
    state = BettingState.initial(g)
    for token in history:
        state.apply(action_kind(token), action_amount(token))
    return state


def walk_against_pokerkit(g: PreflopGame, limit: int | None = None, rng: random.Random | None = None) -> int:
    """Replay (all or a sample of) the tree's nodes in PokerKit; returns nodes checked."""
    tree = PreflopTree(g)
    nodes = tree.nodes if limit is None or limit >= len(tree.nodes) else rng.sample(tree.nodes, limit)
    for node in nodes:
        state = pokerkit_state(g)
        for token in node.history:
            apply_pk(state, token)
        assert_node_matches(g, tree, node, state)
    return len(nodes)


# --- Milestone 2: the betting tree ---------------------------------------------------


@pytest.mark.parametrize("stacks", [EQUAL, ASYM, (40, 41, 300, 12, 9, 80), (3, 250)])
def test_every_node_is_a_legal_pokerkit_state(stacks):
    g = game(stacks, abstraction=BettingAbstraction(max_sized_raises=2, max_raises=4, max_calls_per_raise=1))
    assert walk_against_pokerkit(g, limit=1500, rng=random.Random(1)) > 0


def test_asymmetric_tree_legal_amounts():
    """Test B: raise sizes and all-ins respect each player's own stack."""
    g = game(ASYM)
    tree = PreflopTree(g)
    root = tree.nodes[0]
    assert root.actor == 0 and root.actions == ("f", "r12500", "a500000")  # UTG: 2.5bb or all-in for 100bb
    # UTG folds, HJ (150bb) opens, CO (70bb) can only jam for 70bb; the SB (40bb) jam is 40bb.
    hj_open = tree.node(("f", "r12500"))
    assert hj_open.actor == 2 and hj_open.actions == ("f", "c", "r37500", "a350000")
    sb = tree.node(("f", "r12500", "f", "f"))
    assert sb.actor == 4 and sb.actions == ("f", "c", "r37500", "a200000")
    # The short SB jam (40bb) over a 3-bet to 7.5bb: HJ may call or re-raise (full raise).
    node = tree.node(("f", "r12500", "f", "f", "r37500"))
    assert node.actor == 5
    for nd in tree.nodes:
        for token in nd.actions:
            if action_kind(token) in "ra":
                assert action_amount(token) <= g.stacks[nd.actor]


def test_short_all_in_does_not_reopen_betting():
    """A raises, B calls, C is all-in for less than a full raise: A and B may only call or fold."""
    g = game((100, 100, 4.5, 100, 100), abstraction=BettingAbstraction(max_calls_per_raise=None))
    state = BettingState.initial(g)
    for token in ("r15000", "c", "a22500"):  # 3bb open, call, all-in to 4.5bb (increment 1.5bb < 2bb)
        state.apply(action_kind(token), action_amount(token))
    assert state.raise_bounds(3, BB) is not None  # SB has not acted: may raise
    state.apply("f")
    state.apply("f")  # SB, BB fold
    assert state.actor == 0 and state.raise_bounds(0, BB) is None  # opener faces a short all-in
    pk = pokerkit_state(g)
    for token in ("r15000", "c", "a22500", "f", "f"):
        apply_pk(pk, token)
    assert not pk.can_complete_bet_or_raise_to()


def test_randomized_full_game_rules_match_pokerkit():
    """Property test: random stacks, random legal actions with any raise size."""
    rng = random.Random(7)
    for _ in range(400):
        n = rng.randint(2, 6)
        stacks = [rng.randint(BB + 1, 300 * BB) for _ in range(n)]
        g = PreflopGame(tuple(stacks), SB, BB, RAKE)
        ours = BettingState.initial(g)
        pk = pokerkit_state(g)
        order = preflop_order_from_pokerkit(n)
        while not ours.closed:
            i = ours.actor
            assert order[pk.actor_index] == i
            assert pk.checking_or_calling_amount == ours.to_call(i)
            bounds = ours.raise_bounds(i, BB)
            assert pk.can_complete_bet_or_raise_to() == (bounds is not None)
            if bounds:
                assert bounds == (pk.min_completion_betting_or_raising_to_amount,
                                  pk.max_completion_betting_or_raising_to_amount)
            x = rng.random()
            if bounds and x < 0.45:
                to = bounds[1] if rng.random() < 0.3 else rng.randint(*bounds)
                token = f"r{to}"
            elif x < 0.75 or not ours.to_call(i):
                token = "c" if ours.to_call(i) else "x"
            else:
                token = "f"
            ours.apply(action_kind(token), action_amount(token))
            apply_pk(pk, token)
        assert pk.actor_index is None or pk.street_index != 0


# --- Milestone 3: pots, side pots, rake and payouts ------------------------------------------


class PathAgent:
    """Plays a fixed preflop history, then checks/calls down."""

    def __init__(self, tokens_by_seat):
        self.tokens = tokens_by_seat

    def act(self, obs):
        if obs.street == Street.PREFLOP and self.tokens:
            token = self.tokens.pop(0)
            kind = action_kind(token)
            if kind == "f":
                return Action.fold()
            if kind in "xc":
                return Action.check() if obs.to_call == 0 else Action.call()
            return Action.raise_to(action_amount(token))
        return Action.check() if obs.to_call == 0 else Action.call()

    def observe_hand(self, record):
        pass


def play_terminal(g: PreflopGame, node, seed: int, rake: RakeConfig | None = None):
    """Play ``node``'s history in the PokerKit engine; returns (net by preflop index, ranks)."""
    rake = rake or g.rake
    n = g.n
    order = preflop_order_from_pokerkit(n)  # seat == PokerKit index here
    per_seat = {seat: [] for seat in range(n)}
    replay_state = BettingState.initial(g)
    for token in node.history:
        per_seat[order.index(replay_state.actor)].append(token)
        replay_state.apply(action_kind(token), action_amount(token))
    agents = {seat: PathAgent(per_seat[seat]) for seat in range(n)}
    info = GameInfo(n, g.small_blind, g.big_blind, 0, rake.percentage, rake.cap, rake.no_flop_no_drop, "independent")
    result = play_hand(agents=agents, seats=list(range(n)), button_seat=n - 1,
                       stacks={seat: g.stacks[order[seat]] for seat in range(n)}, game=info,
                       rake_function=make_rake_function(rake), deck_rng=random.Random(seed))
    net = [0] * n
    ranks = [0] * n
    for seat in range(n):
        net[order[seat]] = result.net[seat]
        if result.board:
            ranks[order[seat]] = evaluate(result.hole_cards[seat] + result.board)
    return net, ranks, result


def assert_terminal_payoffs(g: PreflopGame, node, seeds) -> None:
    for seed in seeds:
        net, ranks, result = play_terminal(g, node, seed)
        if node.terminal == "fold":
            assert net == node.fold_payoffs(), node.key
            continue
        ours = node.showdown_payoffs(ranks)
        tied_pots = sum(len({ranks[i] for i in p.eligible}) < len(p.eligible) and
                        sum(ranks[i] == max(ranks[j] for j in p.eligible) for i in p.eligible) > 1
                        for p in node.pots)
        # PokerKit gives odd chips of a split pot to one winner. At showdown it
        # also mucks losing hands and merges pots whose eligible players then
        # coincide before raking, so per-pot rounding can move up to
        # (pots - 1) chips of rake. Everything else is chip-exact.
        rounding = len(node.pots) - 1
        tolerance = rounding + (1 if tied_pots else 0)
        assert all(abs(a - b) <= tolerance for a, b in zip(ours, net)), (node.key, ours, net)
        assert node.rake <= result.rake <= node.rake + rounding, node.key
        assert sum(net) == -result.rake


@pytest.mark.parametrize("stacks", [ASYM, EQUAL])
def test_terminal_payoffs_match_pokerkit(stacks):
    g = game(stacks)
    tree = PreflopTree(g)
    rng = random.Random(3)
    terminals = rng.sample(tree.terminals, 120)
    for k, node in enumerate(terminals):
        assert_terminal_payoffs(g, node, seeds=[1000 * k + j for j in range(3)])


def test_multiple_unequal_all_ins_side_pots():
    """Test C: SB (40), CO (70), UTG (100) all-in and BB (200) calls: three side pots."""
    g = game(ASYM, abstraction=BettingAbstraction(max_sized_raises=0, max_raises=5, max_calls_per_raise=None))
    tree = PreflopTree(g)
    node = tree.node(("a500000", "f", "c", "f", "c", "c"))
    assert node is not None and node.terminal == "flop"
    assert node.contributions == (500_000, 0, 350_000, 0, 200_000, 500_000)
    amounts = [(p.amount + p.rake, p.eligible) for p in node.pots]
    assert amounts == [(800_000, (0, 2, 4, 5)), (450_000, (0, 2, 5)), (300_000, (0, 5))]
    # Whole-hand rake is capped at 300 and split pro rata; per-pot rounding down
    # leaves it one chip under the cap here.
    assert [p.rake for p in node.pots] == [154, 87, 58]  # floor(pot * 300 / 1,550,000)
    # The BB covered UTG; the BB's extra 100bb is never at risk.
    assert_terminal_payoffs(g, node, seeds=range(60))
    # Concrete hand: SB best, CO second, UTG third, BB worst.
    payoff = node.showdown_payoffs([2, 0, 3, 0, 4, 1])
    assert payoff[4] == pytest.approx(800_000 - node.pots[0].rake - 200_000)
    assert payoff[2] == pytest.approx(450_000 - node.pots[1].rake - 350_000)
    assert payoff[0] == pytest.approx(300_000 - node.pots[2].rake - 500_000)
    assert payoff[5] == -500_000
    assert sum(payoff) == pytest.approx(-node.rake)


def test_randomized_all_in_payouts_match_pokerkit():
    """Property test: random stack vectors, everyone all-in or calling, random cards."""
    rng = random.Random(11)
    jam_all = BettingAbstraction(max_sized_raises=0, max_raises=6, max_calls_per_raise=None)
    for trial in range(60):
        n = rng.randint(3, 6)
        stacks = [rng.randint(2, 250) for _ in range(n)]
        g = game(stacks, abstraction=jam_all)
        tree = PreflopTree(g)
        multi = [t for t in tree.terminals if t.terminal == "flop" and len(t.pots) >= 2]
        for node in rng.sample(multi, min(4, len(multi))):
            assert_terminal_payoffs(g, node, seeds=[trial * 100 + j for j in range(4)])


def test_rake_sensitivity_and_conservation():
    """Test D: same tree with and without rake; rake only reduces winners' payouts."""
    raked = PreflopTree(game(ASYM))
    free = PreflopTree(game(ASYM, rake=RakeConfig()))
    assert [nd.key for nd in raked.nodes] == [nd.key for nd in free.nodes]
    rng = random.Random(5)
    seen_cap = False
    for a, b in zip(raked.terminals, free.terminals):
        assert a.contributions == b.contributions
        assert b.rake == 0
        if a.terminal == "fold":
            assert a.rake == 0 and a.fold_payoffs() == b.fold_payoffs()  # no flop, no drop
            continue
        assert 0 < a.rake <= 300
        seen_cap |= a.rake == 300
        total = sum(a.contributions)
        assert a.rake == sum(int(p.amount + p.rake) * min(int(total * 0.045), 300) // total for p in a.pots)
        ranks = [rng.randrange(1000) for _ in range(6)]
        pa, pb = a.showdown_payoffs(ranks), b.showdown_payoffs(ranks)
        assert sum(pb) == pytest.approx(0)
        assert sum(pa) == pytest.approx(-a.rake)
        assert all(x <= y + 1e-9 for x, y in zip(pa, pb))
    assert seen_cap


def test_rake_cap_schedule_by_players_dealt():
    rake = RakeConfig(percentage=0.045, cap=300, cap_by_players={2: 100, 3: 200, 5: 300})
    for n, cap in ((2, 100), (3, 200), (4, 200), (6, 300)):
        _, pots = build_pots([500_000] * 2 + [0] * (n - 2), [True, True] + [False] * (n - 2), rake, True)
        assert sum(p.rake for p in pots) == cap


# --- Test A: equal stacks reproduce a scalar-stack model --------------------------------------


def scalar_reference(stack: int, n: int, ab: BettingAbstraction) -> dict[str, tuple]:
    """Independent equal-stack model in the style of a single-stack solver
    (GTOpen's documented v1 assumption): one stack value, calls always
    complete, a single pot. Returns terminal key -> (contributions, live)."""
    out: dict[str, tuple] = {}

    def rec(bets, live, queue, raises, calls, last_inc, history):
        if not queue or sum(live) == 1:
            out["-".join(history)] = (tuple(bets), tuple(live))
            return
        i = queue[0]
        current = max(bets)
        options = []
        if current > bets[i]:
            options.append("f")
            if raises and (ab.max_calls_per_raise is None or calls < ab.max_calls_per_raise):
                options.append("c")
        else:
            options.append("x")
        others_can_act = any(live[j] and bets[j] < stack for j in range(n) if j != i)
        if raises < ab.max_raises and bets[i] < stack and current < stack and others_can_act:
            min_to = min(stack, current + max(last_inc, BB))
            cands = []
            if raises < ab.max_sized_raises:
                if raises == 0:
                    sizes = ab.sb_open_sizes_bb if i == n - 2 else ab.open_sizes_bb
                    cands = [int(round(s * BB)) for s in sizes]
                else:
                    mults = ab.reraise_multipliers if raises == 1 else ab.later_multipliers
                    cands = [int(round(m * current)) for m in mults]
            seen = []
            for to in cands:
                to = max(to, min_to)
                if to >= ab.allin_threshold * stack:
                    to = stack
                if to not in seen:
                    seen.append(to)
                    options.append(("a" if to == stack else "r") + str(to))
            if stack not in seen:
                options.append(f"a{stack}")
        for token in options:
            b, lv, q = list(bets), list(live), queue[1:]
            r, c, inc = raises, calls, last_inc
            if token == "f":
                lv[i] = False
            elif token == "c":
                b[i] = current
                c += 1
            elif token != "x":
                to = int(token[1:])
                inc = max(inc, to - current)
                b[i] = to
                r += 1
                c = 0
                q = [(i + k) % n for k in range(1, n) if lv[(i + k) % n] and b[(i + k) % n] < stack]
            rec(b, lv, q, r, c, inc, history + [token])

    bets = [0] * n
    bets[n - 2], bets[n - 1] = SB, BB
    rec(bets, [True] * n, list(range(n)), 0, 0, 0, [])
    return out


def test_equal_stacks_reproduce_scalar_model():
    """Test A: with six equal stacks the vector model is the single-stack model, exactly.

    Measured tolerance: 0 differing terminals, 0 chips of payoff difference.
    """
    g = game(EQUAL)
    tree = PreflopTree(g)
    reference = scalar_reference(g.stacks[0], 6, g.abstraction)
    ours = {t.key: t for t in tree.terminals}
    assert set(ours) == set(reference)
    rng = random.Random(2)
    for key, (bets, live) in reference.items():
        node = ours[key]
        assert node.live == live
        pot = sum(bets)
        if node.terminal == "fold":
            expected = [(pot if live[i] else 0) - bets[i] for i in range(6)]
            assert node.fold_payoffs() == expected
        else:
            assert len(node.pots) == 1  # no side pots with equal stacks
            rake = min(int(pot * 0.045), 300)
            ranks = [rng.randrange(50) for _ in range(6)]
            best = max(ranks[i] for i in range(6) if live[i])
            winners = [i for i in range(6) if live[i] and ranks[i] == best]
            expected = [(pot - rake) / len(winners) * (i in winners) - bets[i] for i in range(6)]
            assert node.showdown_payoffs(ranks) == pytest.approx(expected, abs=0)


# --- Test E: position sensitivity -------------------------------------------------------------


def test_rotated_stacks_are_different_games():
    base = game(ASYM)
    keys = set()
    trees = []
    for k in range(6):
        g = base.rotated(k)
        assert sorted(g.stacks) == sorted(base.stacks)
        keys.add(g.fingerprint())
        trees.append(PreflopTree(g))
    assert len(keys) == 6
    # The same seat faces different legal all-in amounts after rotation.
    assert len({t.nodes[0].actions for t in trees}) == 6
    assert len({t.stats()["nodes"] for t in trees}) > 1


def test_heads_up_order_matches_pokerkit():
    g = game((60, 90), abstraction=BettingAbstraction())
    assert walk_against_pokerkit(g) == len(PreflopTree(g).nodes)
    tree = PreflopTree(g)
    assert tree.nodes[0].actor == 0  # button/small blind acts first preflop
    for node in itertools.islice(tree.terminals, 40):
        assert_terminal_payoffs(g, node, seeds=[1, 2])
