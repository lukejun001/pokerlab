"""Postflop decisions by real-time street search, for the solver-backed agent.

``PostflopSearch.decide`` handles heads-up postflop decisions (exactly two
players left, nobody all-in) whose preflop line was played from a strategy
profile (exact, approximate or network): that profile gives both players'
flop ranges. Anything else returns ``(None, reason)`` and the caller falls
back. Each street is solved once per hand (``search.solve_street``) and all
of that street's decisions read from the solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.observations import Observation, Street, position_labels
from pokerlab.postflop import search as S
from pokerlab.solver.game import action_kind

NodeTable = Callable[[str], tuple[tuple[str, ...], np.ndarray] | None]


@dataclass
class HandContext:
    hand_id: int
    preflop_seats: list[int]  # seats in preflop order
    preflop_tokens: list[str]  # the preflop line in tree tokens
    game: Any  # the modeled PreflopGame
    node_table: NodeTable  # preflop profile: history -> (actions, actions x 169)
    solves: dict[int, S.StreetSolve] = field(default_factory=dict)


def preflop_reach(ctx: HandContext) -> np.ndarray | None:
    """(n, 169) reach of every preflop seat along the line actually played, or None if off-profile."""
    from pokerlab.solver.game import BettingState, action_amount

    state = BettingState.initial(ctx.game)
    reach = np.ones((ctx.game.n, 169))
    history: list[str] = []
    for token in ctx.preflop_tokens:
        if state.closed:
            return None
        found = ctx.node_table("-".join(history))
        if found is None:  # the line left the profile's tree: later actions are uninformative
            break
        actions, table = found
        if token in actions:
            k = actions.index(token)
        else:  # an off-tree size: the action of the same kind (raise, all-in, passive)
            same = [i for i, t in enumerate(actions) if t[0] == token[0] or {t[0], token[0]} <= {"x", "c"}]
            k = same[0] if same else None  # e.g. a limp the tree lacks: treated as uninformative
        if k is not None:
            reach[state.actor] *= table[k]
        state.apply(action_kind(token), action_amount(token))
        history.append(token)
    return reach


class PostflopSearch:
    def __init__(self, settings: S.SearchSettings | None = None, rng=None) -> None:
        self.settings = settings or S.SearchSettings()
        self.rng = rng
        self.contexts: dict[int, HandContext] = {}
        self.solve_log: list[dict[str, Any]] = []

    def remember_preflop(self, ctx: HandContext) -> None:
        self.contexts[ctx.hand_id] = ctx
        while len(self.contexts) > 16:
            self.contexts.pop(next(iter(self.contexts)))

    def decide(self, obs: Observation) -> tuple[Action | None, str]:
        ctx = self.contexts.get(obs.hand_id)
        if ctx is None:
            return None, "no_preflop_profile"
        live = [s for s in ctx.preflop_seats if obs.in_hand[s]]
        if len(live) != 2:
            return None, "multiway"
        if any(obs.all_in[s] for s in live):
            return None, "all_in"
        labels = position_labels(obs.num_players)
        oop, ip = sorted(live, key=lambda s: labels.index(obs.seat_positions[s]))
        me = 0 if obs.seat == oop else 1
        bb = obs.game.big_blind
        street_no = int(obs.street)
        street = S.STREETS[street_no - 1]
        board = tuple(obs.board[: street_no + 2])
        solve = ctx.solves.get(street_no)
        if solve is None:
            ranges = self._root_ranges(ctx, obs, street_no, board, (oop, ip))
            if isinstance(ranges, str):
                return None, ranges
            hole = S.combo_key(*obs.hole_cards)
            mine = ranges[me]
            if mine.get(hole, 0.0) <= 0:  # keep our own hand in our range, at a negligible weight
                mine[hole] = 1e-3 * max(mine.values(), default=1.0)
            pot = (obs.pot - sum(obs.bets)) / bb
            stack = min(obs.stacks[s] + obs.bets[s] for s in (oop, ip)) / bb
            if stack <= 0:
                return None, "no_stack"
            cap = obs.game.rake_cap
            caps = dict(obs.game.rake_cap_by_players)
            if caps:
                eligible = [k for k in caps if k <= obs.num_players]
                cap = caps[max(eligible)] if eligible else cap
            solve = S.solve_street(street, board, ranges, pot, stack, obs.game.rake_percentage,
                                   (cap or 0) / bb, self.settings)
            ctx.solves[street_no] = solve
            m = solve.metrics
            self.solve_log.append({"street": street, "iterations": m["iterations"], "solve_s": round(m["solve_s"], 2),
                                   "exploit_pct_pot": round(m["exploit_pct_pot"], 3), "pot_bb": round(pot, 2),
                                   "stack_bb": round(stack, 2), "hands": m["hands"]})
        path = self._street_path(solve, obs, street_no, bb)
        if path is None:
            return None, "line_not_in_tree"
        view = solve.view(path)
        if view is None or view["node_type"] != "action" or view["player"] != me:
            return None, "line_not_in_tree"
        strategy = S.hand_strategy(view, me, obs.hole_cards)
        if strategy is None or strategy.sum() <= 0:
            return None, "hand_not_in_range"
        choice = self._sample(strategy / strategy.sum())
        return self._to_engine(view["actions"][choice], obs, bb), f"search:{street}"

    # --- helpers ---------------------------------------------------------------------------------

    def _sample(self, p: np.ndarray) -> int:
        x = self.rng.random() if self.rng else np.random.random()
        return int(min(np.searchsorted(np.cumsum(p), x, side="right"), len(p) - 1))

    def _root_ranges(self, ctx, obs, street_no, board, order):
        if street_no == 1:
            reach = preflop_reach(ctx)
            if reach is None:
                return "preflop_line_off_profile"
            idx = {s: k for k, s in enumerate(ctx.preflop_seats)}
            return tuple(S.range_from_classes(reach[idx[s]], board) for s in order)
        prev = ctx.solves.get(street_no - 1)
        if prev is None:
            return "previous_street_not_searched"
        path = self._street_path(prev, obs, street_no - 1, obs.game.big_blind)
        if path is None or prev.view(path) is None or prev.view(path)["node_type"] != "chance":
            return "previous_street_line_not_in_tree"
        new = board[-1]
        out = []
        for p in (0, 1):
            r = {k: w for k, w in prev.reach(path, p).items() if new not in k}
            if not r:
                return "empty_range"
            out.append(r)
        return tuple(out)

    @staticmethod
    def _street_path(solve, obs, street_no, bb) -> tuple[int, ...] | None:
        path: tuple[int, ...] = ()
        for a in obs.history:
            if int(a.street) != street_no:
                continue
            view = solve.view(path)
            if view is None or view["node_type"] != "action":
                return None
            kind = {ActionType.FOLD: "fold", ActionType.CHECK: "check", ActionType.CALL: "call",
                    ActionType.BET: "bet", ActionType.RAISE: "raise"}[a.type]
            to_bb = a.amount / bb if kind in ("bet", "raise") else None
            path = path + (S.match_action(view, kind, to_bb, a.all_in),)
        return path

    @staticmethod
    def _to_engine(action: dict, obs: Observation, bb: int) -> Action:
        legal = obs.legal_actions
        kind = action["kind"]
        if kind == "fold":
            return Action.fold() if ActionType.FOLD in legal and obs.to_call > 0 else Action.check()
        if kind in ("check", "call"):
            if ActionType.CHECK in legal:
                return Action.check()
            return Action.call()
        to = int(round(action["amount"] * bb))
        if obs.min_raise_to is None or obs.max_raise_to is None:
            return Action.call() if ActionType.CALL in legal else Action.check()
        to = max(obs.min_raise_to, min(obs.max_raise_to, to))
        if ActionType.BET in legal:
            return Action.bet(to)
        return Action.raise_to(to)
