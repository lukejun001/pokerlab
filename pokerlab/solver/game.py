"""Asymmetric-stack No-Limit Hold'em preflop game tree.

Every player has an independent starting stack. The betting rules mirror
PokerKit (our simulator's rules engine) so that every abstract node is a real,
legal PokerKit state:

* calls are capped at the caller's stack (all-in for less);
* a raise's legal range is ``[min_to, stack + bet]`` with PokerKit's minimum;
* a short all-in raise does not reopen betting for players who already acted
  (PokerKit's ``acted_player_indices`` / ``consecutive_short_all_in_amounts``);
* raising is impossible once every other live player is all-in or covered.

Terminal nodes are either a fold-out or "the flop": betting closed with two or
more live players. Flop terminals record the exact main/side pot structure
PokerKit builds (uncalled-bet refund, folded chips, merged layers) and the rake
each pot pays under the configured schedule, so the payout for any concrete
showdown is chip-exact. What happens *after* the flop is a separate
continuation model (see ``continuation.py``); this module is pure rules.

Players are indexed in preflop action order: 0 acts first, ``n-2`` is the
small blind and ``n-1`` the big blind (heads-up: 0 = button/small blind).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pokerlab.engine.rake import RakeConfig

GAME_VARIANT = "no_limit_texas_holdem"
TREE_VERSION = "preflop-tree-1"

_LABELS_6 = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
_MIDDLE = ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO")


def position_names(n: int) -> tuple[str, ...]:
    """Position labels in preflop action order."""
    if not 2 <= n <= 9:
        raise ValueError("between 2 and 9 players")
    if n == 2:
        return ("SB", "BB")
    if n == 6:
        return _LABELS_6
    middle = _MIDDLE[len(_MIDDLE) - (n - 3):] if n > 3 else ()
    return (*middle, "BTN", "SB", "BB")


def engine_label_aliases(n: int) -> dict[str, str]:
    """Simulator position label -> solver label (the simulator calls 6-max UTG "LJ")."""
    from pokerlab.engine.observations import position_labels

    engine = position_labels(n)
    order = preflop_order_from_pokerkit(n)
    names = position_names(n)
    return {engine[pk]: names[order[pk]] for pk in range(n)}


def preflop_order_from_pokerkit(n: int) -> tuple[int, ...]:
    """PokerKit player index (0 = left of the button) -> preflop order index."""
    if n == 2:
        return (1, 0)  # PokerKit heads-up: index 0 is the big blind
    return (n - 2, n - 1, *range(n - 2))


@dataclass(frozen=True)
class BettingAbstraction:
    """Which raise sizes the tree offers. Sizes are raise-to amounts."""

    name: str = "std-v1"
    open_sizes_bb: tuple[float, ...] = (2.5,)
    sb_open_sizes_bb: tuple[float, ...] = (3.0,)
    reraise_multipliers: tuple[float, ...] = (3.0,)  # 3-bet: multiple of the bet faced
    later_multipliers: tuple[float, ...] = (2.25,)  # 4-bet and later
    max_sized_raises: int = 3  # open, 3-bet, 4-bet; later raises are all-in only
    max_raises: int = 4  # including all-ins
    allow_allin: bool = True
    allin_threshold: float = 0.5  # a sized raise putting in >= this share of the stack becomes all-in
    allow_limp: bool = False  # open-limps and small-blind completes
    max_calls_per_raise: int | None = None  # flat calls allowed after each raise (None = no limit)

    def __post_init__(self) -> None:
        if not 0 < self.allin_threshold <= 1:
            raise ValueError("allin_threshold must be in (0, 1]")
        if self.max_sized_raises > self.max_raises:
            raise ValueError("max_sized_raises cannot exceed max_raises")
        for name in ("open_sizes_bb", "sb_open_sizes_bb", "reraise_multipliers", "later_multipliers"):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> BettingAbstraction:
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown abstraction keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class PreflopGame:
    """One concrete preflop game: stacks (chips) in preflop action order."""

    stacks: tuple[int, ...]
    small_blind: int
    big_blind: int
    rake: RakeConfig = field(default_factory=RakeConfig)
    abstraction: BettingAbstraction = field(default_factory=BettingAbstraction)
    ante: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stacks", tuple(int(s) for s in self.stacks))
        if not 2 <= len(self.stacks) <= 9:
            raise ValueError("between 2 and 9 players")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind <= big_blind")
        if self.ante:
            raise NotImplementedError("antes are not modeled yet (the tree is only validated without them)")
        if min(self.stacks) <= self.big_blind:
            raise ValueError("every stack must exceed the big blind (short-blind postings are not modeled)")

    @property
    def n(self) -> int:
        return len(self.stacks)

    @property
    def positions(self) -> tuple[str, ...]:
        return position_names(self.n)

    def stacks_bb(self) -> tuple[float, ...]:
        return tuple(s / self.big_blind for s in self.stacks)

    def describe(self) -> dict[str, Any]:
        """Canonical description: everything that changes the game's rules or payoffs."""
        r = self.rake
        return {
            "variant": GAME_VARIANT,
            "players": self.n,
            "positions": list(self.positions),
            "stacks": list(self.stacks),
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "ante": self.ante,
            "rake": {
                "percentage": r.percentage,
                "cap": r.cap,
                "cap_by_players": {str(k): v for k, v in sorted(r.cap_by_players.items())},
                "no_flop_no_drop": r.no_flop_no_drop,
            },
            "abstraction": self.abstraction.to_dict(),
            "tree_version": TREE_VERSION,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()[:16]

    def rotated(self, k: int) -> PreflopGame:
        """The same stack vector moved ``k`` seats (stack of position i goes to i+k)."""
        n = self.n
        stacks = tuple(self.stacks[(i - k) % n] for i in range(n))
        return PreflopGame(stacks, self.small_blind, self.big_blind, self.rake, self.abstraction, self.ante)


# ---------------------------------------------------------------------------
# Betting state (a port of PokerKit's preflop betting rules)
# ---------------------------------------------------------------------------


@dataclass
class BettingState:
    bets: list[int]  # chips put in this hand (preflop only, so == contributions)
    stacks: list[int]  # chips behind
    live: list[bool]
    queue: list[int]  # players still to act, in order
    acted: set[int]
    raise_amount: int  # PokerKit's completion_betting_or_raising_amount
    short_all_ins: list[int]
    raises: int  # completed raises so far (including all-in raises)
    calls: int = 0  # flat calls since the last raise

    @classmethod
    def initial(cls, game: PreflopGame) -> BettingState:
        n = game.n
        bets = [0] * n
        bets[n - 2] = game.small_blind
        bets[n - 1] = game.big_blind
        stacks = [s - b for s, b in zip(game.stacks, bets)]
        return cls(bets, stacks, [True] * n, list(range(n)), set(), 0, [], 0)

    def copy(self) -> BettingState:
        return BettingState(self.bets.copy(), self.stacks.copy(), self.live.copy(), self.queue.copy(),
                            set(self.acted), self.raise_amount, self.short_all_ins.copy(), self.raises, self.calls)

    @property
    def actor(self) -> int | None:
        return self.queue[0] if self.queue else None

    def to_call(self, i: int) -> int:
        return min(max(self.bets) - self.bets[i], self.stacks[i])

    def raise_bounds(self, i: int, big_blind: int) -> tuple[int, int] | None:
        """(min_to, max_to) if player ``i`` may raise, else None (PokerKit's rules)."""
        current = max(self.bets)
        owed = current - self.bets[i]
        if min(owed, self.stacks[i]) < self.raise_amount:
            return None  # "short all-in cannot be raised"
        if self.short_all_ins and sum(self.short_all_ins) < self.raise_amount and i in self.acted:
            return None  # already acted, facing only a non-full all-in raise
        if self.stacks[i] <= owed:
            return None  # covered: can only call
        if not any(j != i and self.live[j] and self.stacks[j] + self.bets[j] > current for j in range(len(self.bets))):
            return None  # everyone else folded or all-in
        max_to = self.stacks[i] + self.bets[i]
        min_to = min(max_to, max(self.raise_amount, big_blind) + current)
        return min_to, max_to

    def apply(self, kind: str, amount: int = 0) -> None:
        i = self.queue.pop(0)
        self.acted.add(i)
        n = len(self.bets)
        if kind == "f":
            self.live[i] = False
        elif kind in ("x", "c"):
            pay = self.to_call(i)
            self.calls += kind == "c"
            self.bets[i] += pay
            self.stacks[i] -= pay
        else:
            increment = amount - max(self.bets)
            delta = amount - self.bets[i]
            self.bets[i] = amount
            self.stacks[i] -= delta
            self.queue = [(i + k) % n for k in range(1, n)
                          if self.live[(i + k) % n] and self.stacks[(i + k) % n]]
            if increment >= self.raise_amount:
                self.acted = {i}
            self.raise_amount = max(self.raise_amount, increment)
            self.raises += 1
            self.calls = 0
            if self.stacks[i]:
                self.short_all_ins.clear()
            else:
                self.short_all_ins.append(increment)
            if sum(self.short_all_ins) >= self.raise_amount:
                self.short_all_ins.clear()
        if sum(self.live) <= 1:
            self.queue = []

    @property
    def closed(self) -> bool:
        return not self.queue


# ---------------------------------------------------------------------------
# Pots and rake (a port of PokerKit's bet collection and pot building)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pot:
    amount: int  # paid to the winners (after rake)
    rake: int
    eligible: tuple[int, ...]


def build_pots(bets: Sequence[int], live: Sequence[bool], rake: RakeConfig, flop_dealt: bool) -> tuple[list[int], list[Pot]]:
    """Return (contribution after uncalled-bet refund, pots) for a finished betting round.

    Follows PokerKit: the highest bet above the second-highest bet (folded
    players included) is refunded; pots are layered by every distinct
    contribution, eligible players are the live ones who reached the layer,
    adjacent layers with the same eligible players merge, and the rake
    callback is applied to each final pot.
    """
    n = len(bets)
    contrib = list(bets)
    if sum(live) == 1:
        winner = live.index(True)
        pending = list(bets)
        contrib[winner] = 0
    else:
        cutoff = sorted(bets)[-2]
        contrib = [min(b, cutoff) for b in bets]
        pending = contrib
    total = sum(contrib)
    hand_rake = rake.hand_rake(total, n, flop_dealt) if total else 0
    pots: list[tuple[int, tuple[int, ...]]] = []
    previous = 0
    for level in sorted(set(contrib)):
        amount = sum(min(c, level) - min(c, previous) for c in contrib)
        eligible = tuple(i for i in range(n) if pending[i] >= level and live[i])
        while pots and pots[-1][1] == eligible:
            amount += pots.pop()[0]
        if amount:
            pots.append((amount, eligible))
        previous = level
    result = []
    for amount, eligible in pots:
        taken = amount * hand_rake // total if hand_rake else 0
        result.append(Pot(amount - taken, taken, eligible))
    if sum(live) == 1:
        contrib[live.index(True)] = 0
    return contrib, result


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: int
    history: tuple[str, ...]
    actor: int = -1  # -1 for terminals
    actions: tuple[str, ...] = ()  # "f", "x", "c", "r<to>" or "a<to>" (all-in), amounts in chips
    children: tuple[int, ...] = ()
    terminal: str = ""  # "fold" or "flop"
    bets: tuple[int, ...] = ()  # each player's chips in when the node is reached
    live: tuple[bool, ...] = ()
    contributions: tuple[int, ...] = ()  # terminals: chips at risk after refunds
    pots: tuple[Pot, ...] = ()
    fold_winner: int = -1
    all_in: tuple[bool, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.actor < 0

    @property
    def key(self) -> str:
        return "-".join(self.history)

    @property
    def rake(self) -> int:
        return sum(p.rake for p in self.pots)

    def fold_payoffs(self) -> list[int]:
        """Net chips per player at a fold-out terminal."""
        total = sum(p.amount for p in self.pots)
        return [(total if i == self.fold_winner else 0) - c for i, c in enumerate(self.contributions)]

    def showdown_payoffs(self, ranks: Sequence[int]) -> list[float]:
        """Net chips per player at a flop terminal if the hand is checked down
        and the live players' final hands rank as ``ranks`` (higher wins)."""
        won = [0.0] * len(self.contributions)
        for pot in self.pots:
            best = max(ranks[i] for i in pot.eligible)
            winners = [i for i in pot.eligible if ranks[i] == best]
            for i in winners:
                won[i] += pot.amount / len(winners)
        return [w - c for w, c in zip(won, self.contributions)]


def action_kind(token: str) -> str:
    return token[0]


def action_amount(token: str) -> int:
    return int(token[1:]) if token[0] in "ra" else 0


def raise_targets(game: PreflopGame, state: BettingState, actor: int) -> list[str]:
    """Abstract raise actions for ``actor`` (``r<to>`` / ``a<to>``), deduplicated."""
    ab = game.abstraction
    if state.raises >= ab.max_raises:
        return []
    bounds = state.raise_bounds(actor, game.big_blind)
    if bounds is None:
        return []
    min_to, max_to = bounds
    candidates: list[int] = []
    if state.raises < ab.max_sized_raises:
        current = max(state.bets)
        if state.raises == 0:
            sizes = ab.sb_open_sizes_bb if actor == game.n - 2 and ab.sb_open_sizes_bb else ab.open_sizes_bb
            candidates = [int(round(s * game.big_blind)) for s in sizes]
        else:
            mults = ab.reraise_multipliers if state.raises == 1 else ab.later_multipliers
            candidates = [int(round(m * current)) for m in mults]
    tokens: list[str] = []
    seen: set[int] = set()
    for to in candidates:
        to = max(to, min_to)
        if to >= max_to or to >= ab.allin_threshold * max_to:
            to = max_to
        if to == max_to and not ab.allow_allin:
            continue
        if to not in seen:
            seen.add(to)
            tokens.append(("a" if to == max_to else "r") + str(to))
    if ab.allow_allin and max_to not in seen:
        tokens.append(f"a{max_to}")
    return tokens


def legal_tokens(game: PreflopGame, state: BettingState) -> list[str]:
    i = state.actor
    assert i is not None
    owed = max(state.bets) - state.bets[i]
    tokens = []
    if owed:
        tokens.append("f")
        limit = game.abstraction.max_calls_per_raise
        if (state.raises or game.abstraction.allow_limp) and (limit is None or state.calls < limit or not state.raises):
            tokens.append("c")
    else:
        tokens.append("x")
    tokens.extend(raise_targets(game, state, i))
    return tokens


class PreflopTree:
    """The full abstract preflop tree of one game, as a flat node list."""

    def __init__(self, game: PreflopGame, max_nodes: int = 2_000_000) -> None:
        self.game = game
        self.nodes: list[Node] = []
        self.by_key: dict[str, int] = {}
        self.max_nodes = max_nodes
        self._build(BettingState.initial(game), ())

    def _build(self, state: BettingState, history: tuple[str, ...]) -> int:
        if len(self.nodes) >= self.max_nodes:
            raise ValueError(f"tree exceeds {self.max_nodes} nodes; restrict the betting abstraction")
        node = Node(id=len(self.nodes), history=history, bets=tuple(state.bets), live=tuple(state.live),
                    all_in=tuple(state.live[i] and state.stacks[i] == 0 for i in range(self.game.n)))
        self.nodes.append(node)
        self.by_key[node.key] = node.id
        if state.closed:
            folded_out = sum(state.live) == 1
            node.terminal = "fold" if folded_out else "flop"
            contrib, pots = build_pots(state.bets, state.live, self.game.rake, flop_dealt=not folded_out)
            node.contributions = tuple(contrib)
            node.pots = tuple(pots)
            if folded_out:
                node.fold_winner = state.live.index(True)
            return node.id
        node.actor = state.actor
        node.actions = tuple(legal_tokens(self.game, state))
        children = []
        for token in node.actions:
            child = state.copy()
            child.apply(action_kind(token), action_amount(token))
            children.append(self._build(child, history + (token,)))
        node.children = tuple(children)
        return node.id

    # --- queries -------------------------------------------------------------

    @property
    def decision_nodes(self) -> list[Node]:
        return [nd for nd in self.nodes if not nd.is_terminal]

    @property
    def terminals(self) -> list[Node]:
        return [nd for nd in self.nodes if nd.is_terminal]

    def stats(self) -> dict[str, int]:
        terms = self.terminals
        return {
            "nodes": len(self.nodes),
            "decision_nodes": len(self.nodes) - len(terms),
            "fold_terminals": sum(t.terminal == "fold" for t in terms),
            "flop_terminals": sum(t.terminal == "flop" for t in terms),
            "multiway_flops": sum(t.terminal == "flop" and sum(t.live) >= 3 for t in terms),
            "side_pot_terminals": sum(len(t.pots) > 1 for t in terms),
            "infosets": 169 * (len(self.nodes) - len(terms)),
            "action_edges": sum(len(nd.actions) for nd in self.nodes),
        }

    def node(self, history: Sequence[str]) -> Node | None:
        idx = self.by_key.get("-".join(history))
        return None if idx is None else self.nodes[idx]
