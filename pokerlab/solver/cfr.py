"""Chance-sampled Discounted CFR over the asymmetric-stack preflop tree.

Each iteration deals a batch of real hands (exact card removal) and a board
per hand, walks the whole public betting tree once for the batch, and updates
every player's regrets at every node, vectorized over the deals. Information
sets are (node, own hand class): preflop the 169 classes are lossless.

Terminal values:

* fold-outs pay exactly what PokerKit pays (uncalled bets refunded, no rake
  under no-flop-no-drop);
* flop terminals use the **showdown continuation model** (version
  ``showdown-v1``): the hand is checked down on the dealt board and every
  main/side pot goes to its best eligible hand after that pot's rake. There is
  no postflop betting, so equity is fully realized; this is the main modeling
  approximation (see ``docs/phase3_report.md``).

The update rule is Discounted CFR (alpha=1.5, beta=0, gamma=2), applied to
sampled counterfactual values. Average strategies are accumulated per class
with each player's exact own reach (no sampling noise in the weights). For
games with three or more players CFR has no equilibrium guarantee; the
measured best-response gains (``evaluate``) are what to rely on.
"""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass

import numpy as np

from pokerlab.solver.game import PreflopTree
from pokerlab.solver.hands import NUM_CLASSES, deal_batch

SOLVER_VERSION = "pokerlab-cs-dcfr-1"
CONTINUATION_VERSION = "showdown-v1"

DCFR_ALPHA = 1.5
DCFR_BETA = 0.0
DCFR_GAMMA = 2.0
HOLDOUT_Z = 2.0


def class_sums(classes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum ``weights`` (B, A) by hand class -> (A, 169)."""
    return np.stack([np.bincount(classes, weights=weights[:, a], minlength=NUM_CLASSES)
                     for a in range(weights.shape[1])])


def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


@dataclass
class Deals:
    classes: np.ndarray  # (B, n) hand class per player
    ranks: np.ndarray  # (B, n) checked-down showdown strength

    @classmethod
    def sample(cls, rng: np.random.Generator, size: int, players: int) -> Deals:
        _, classes, ranks = deal_batch(rng, size, players)
        return cls(classes, ranks)

    def __len__(self) -> int:
        return len(self.classes)


class PreflopSolver:
    def __init__(self, tree: PreflopTree, batch_size: int = 4096, seed: int = 0) -> None:
        self.tree = tree
        self.game = tree.game
        self.n = tree.game.n
        self.batch_size = batch_size
        self.seed = seed
        self.iterations = 0
        self.elapsed = 0.0
        bb = self.game.big_blind

        # Decision nodes get a slice of the edge arrays.
        self.offset: dict[int, int] = {}
        edge_counts = []
        for nd in tree.nodes:
            if not nd.is_terminal:
                self.offset[nd.id] = sum(edge_counts)
                edge_counts.append(len(nd.actions))
        self.decision_ids = np.array(sorted(self.offset), dtype=np.int64)
        self.edges = int(sum(edge_counts))
        self.segment_starts = np.array([self.offset[k] for k in self.decision_ids], dtype=np.int64)
        self.edge_width = np.repeat(np.array(edge_counts, dtype=np.float64), edge_counts)

        # Tables are (edges, 169): a node's actions are contiguous rows.
        self.regret = np.zeros((self.edges, NUM_CLASSES))
        self.strategy_sum = np.zeros((self.edges, NUM_CLASSES))
        self.current = np.empty((self.edges, NUM_CLASSES))
        self.others = [np.array([j for j in range(self.n) if j != i]) for i in range(self.n)]
        self._update_current()

        # Terminal payoffs in big blinds.
        self.fold_value: dict[int, np.ndarray] = {}
        self.flop_pots: dict[int, tuple[np.ndarray, list[tuple[float, np.ndarray]]]] = {}
        for nd in tree.terminals:
            if nd.terminal == "fold":
                self.fold_value[nd.id] = np.array(nd.fold_payoffs(), dtype=np.float64) / bb
            else:
                pots = [(p.amount / bb, np.array(p.eligible)) for p in nd.pots]
                self.flop_pots[nd.id] = (np.array(nd.contributions, dtype=np.float64) / bb, pots)
        self.history: list[dict] = []
        self.br_z = 0.0  # significance threshold for best-response deviations (see _best_actions)

    # --- strategies ---------------------------------------------------------------

    def _normalize(self, table: np.ndarray) -> np.ndarray:
        positive = np.maximum(table, 0.0)
        sums = np.add.reduceat(positive, self.segment_starts, axis=0)
        widths = np.diff(np.append(self.segment_starts, self.edges))
        denom = np.repeat(sums, widths, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denom > 0, positive / denom, 1.0 / self.edge_width[:, None])
        return out

    def _update_current(self) -> None:
        self.current = self._normalize(self.regret)

    def average_strategy(self) -> np.ndarray:
        """(edges, 169) average strategy; uniform where a class never reached a node."""
        return self._normalize(self.strategy_sum)

    def node_strategy(self, node_id: int, table: np.ndarray | None = None) -> np.ndarray:
        table = self.average_strategy() if table is None else table
        off = self.offset[node_id]
        return table[off:off + len(self.tree.nodes[node_id].actions)].T  # (169, A)

    # --- terminal values -----------------------------------------------------------

    def _terminal(self, node_id: int, ranks: np.ndarray) -> np.ndarray:
        fold = self.fold_value.get(node_id)
        if fold is not None:
            return np.broadcast_to(fold, (len(ranks), self.n))
        contrib, pots = self.flop_pots[node_id]
        values = np.broadcast_to(-contrib, (len(ranks), self.n)).copy()
        for amount, eligible in pots:
            if len(eligible) == 1:
                values[:, eligible[0]] += amount
                continue
            r = ranks[:, eligible]
            win = r == r.max(axis=1, keepdims=True)
            values[:, eligible] += win * (amount / win.sum(axis=1, keepdims=True))
        return values

    # --- training --------------------------------------------------------------------

    def _walk(self, node_id: int, idx: np.ndarray, reach: np.ndarray, deals: Deals,
              regret_inc: np.ndarray) -> np.ndarray:
        nd = self.tree.nodes[node_id]
        if nd.is_terminal:
            return self._terminal(node_id, deals.ranks[idx])
        i = nd.actor
        off = self.offset[node_id]
        width = len(nd.actions)
        cls = deals.classes[idx, i]
        sigma = self.current[off:off + width][:, cls].T  # (B, A)
        values = np.zeros((len(idx), self.n))
        child_values = np.zeros((len(idx), width))
        for a, child in enumerate(nd.children):
            child_reach = reach.copy()
            child_reach[:, i] *= sigma[:, a]
            # A deal where two players can no longer reach the child contributes
            # nothing to anyone's counterfactual regret below it.
            alive = (child_reach > 0).sum(axis=1) >= self.n - 1
            if alive.all():
                v = self._walk(child, idx, child_reach, deals, regret_inc)
            else:
                v = np.zeros((len(idx), self.n))
                if alive.any():
                    v[alive] = self._walk(child, idx[alive], child_reach[alive], deals, regret_inc)
            values += sigma[:, a, None] * v
            child_values[:, a] = v[:, i]
        others = np.prod(reach[:, self.others[i]], axis=1)
        inc = others[:, None] * (child_values - values[:, i, None])
        regret_inc[off:off + width] += class_sums(cls, inc)
        return values

    def sampled_regrets(self, deals: Deals) -> np.ndarray:
        """Counterfactual regret increments summed over ``deals`` for the current strategy."""
        regret_inc = np.zeros_like(self.regret)
        self._walk(0, np.arange(len(deals)), np.ones((len(deals), self.n)), deals, regret_inc)
        return regret_inc

    def _accumulate_average(self, node_id: int, own: np.ndarray) -> None:
        """Add the current strategy, weighted by each class's own reach, to the average."""
        nd = self.tree.nodes[node_id]
        if nd.is_terminal:
            return
        i = nd.actor
        off = self.offset[node_id]
        strat = self.current[off:off + len(nd.actions)]  # (A, 169)
        self.strategy_sum[off:off + len(nd.actions)] += own[i] * strat
        for a, child in enumerate(nd.children):
            child_own = own.copy()
            child_own[i] = own[i] * strat[a]
            if child_own[i].any():
                self._accumulate_average(child, child_own)

    def iterate(self, deals: Deals | None = None, regret_inc: np.ndarray | None = None,
                deal_count: int | None = None) -> None:
        """One iteration. ``regret_inc``/``deal_count`` let a caller supply
        increments computed elsewhere (e.g. by worker processes)."""
        started = time.perf_counter()
        t = self.iterations + 1
        if regret_inc is None:
            if deals is None:
                deals = Deals.sample(np.random.default_rng([self.seed, t]), self.batch_size, self.n)
            regret_inc = self.sampled_regrets(deals)
            deal_count = len(deals)
        self._accumulate_average(0, np.ones((self.n, NUM_CLASSES)))
        # Sampled values are sums over the batch; scale to a per-deal average so
        # discounting behaves the same for any batch size.
        self.regret += regret_inc / deal_count
        pos = self.regret > 0
        self.regret[pos] *= t**DCFR_ALPHA / (t**DCFR_ALPHA + 1)
        self.regret[~pos] *= t**DCFR_BETA / (t**DCFR_BETA + 1)
        self.strategy_sum *= (t / (t + 1)) ** DCFR_GAMMA
        self._update_current()
        self.iterations = t
        self.elapsed += time.perf_counter() - started

    def solve(self, iterations: int, eval_every: int = 0, eval_deals: int = 20_000,
              time_limit_s: float | None = None, progress: bool = False) -> None:
        eval_set = None
        for _ in range(iterations):
            self.iterate()
            if eval_every and (self.iterations % eval_every == 0 or self.iterations == iterations):
                if eval_set is None:
                    eval_set = Deals.sample(np.random.default_rng([self.seed, 10**9]), eval_deals, self.n)
                report = self.evaluate(eval_set)
                report.update(iteration=self.iterations, elapsed_s=round(self.elapsed, 2))
                self.history.append(report)
                if progress:
                    print(f"  iter {self.iterations:>5}  {self.elapsed:7.1f}s  "
                          f"NashConv(in-sample) {report['nash_conv_bb100']:8.2f} bb/100", file=sys.stderr)
            if time_limit_s is not None and self.elapsed >= time_limit_s:
                break

    # --- evaluation --------------------------------------------------------------------

    def _values(self, node_id: int, idx: np.ndarray, deals: Deals, table: np.ndarray,
                br_player: int = -1, policy: dict | None = None, best: dict | None = None,
                weight: np.ndarray | None = None) -> np.ndarray:
        """Values (len(idx), n) when everyone plays ``table``, except that
        ``br_player`` plays a fixed ``policy`` (node -> best action per class)
        or, if ``best`` is given, computes a best response into it. ``weight``
        is the other players' probability of reaching this node in each deal."""
        if weight is None:
            weight = np.ones(len(idx))
        nd = self.tree.nodes[node_id]
        if nd.is_terminal:
            return self._terminal(node_id, deals.ranks[idx])
        i = nd.actor
        off = self.offset[node_id]
        width = len(nd.actions)
        if i == br_player and best is not None:
            cls = deals.classes[idx, i]
            child = np.stack([self._values(c, idx, deals, table, br_player, policy, best, weight)
                              for c in nd.children])  # (A, B, n)
            q = child[:, :, i].T  # (B, A) values for the responder
            sigma = table[off:off + width][:, cls].T
            choice = self._best_actions(cls, q, sigma, weight)
            best[node_id] = choice
            pick = choice[cls]
            out = np.einsum("ba,abn->bn", sigma, child)  # classes that keep the average strategy
            dev = pick >= 0
            out[dev] = child[pick[dev], np.flatnonzero(dev)]
            return out
        if i == br_player and policy is not None:
            pick = policy[node_id][deals.classes[idx, i]]
            sigma = table[off:off + width][:, deals.classes[idx, i]].T
            out = np.zeros((len(idx), self.n))
            for a, c in enumerate(nd.children):
                # Play action ``a`` where the policy says so, or with the average
                # strategy's probability where the policy keeps it (-1).
                share = np.where(pick == a, 1.0, np.where(pick < 0, sigma[:, a], 0.0))
                sel = share > 0
                if sel.any():
                    out[sel] += share[sel, None] * self._values(c, idx[sel], deals, table, br_player, policy, best,
                                                                weight[sel])
            return out
        sigma = table[off:off + width][:, deals.classes[idx, i]].T
        out = np.zeros((len(idx), self.n))
        for a, c in enumerate(nd.children):
            # Deals that never take this action contribute nothing below it.
            # (At the responder's own nodes every action is explored above.)
            sel = sigma[:, a] > 0
            if not sel.any():
                continue
            w = weight[sel] * sigma[sel, a] if br_player >= 0 else weight[sel]
            v = self._values(c, idx[sel], deals, table, br_player, policy, best, w)
            out[sel] += sigma[sel, a, None] * v
        return out

    def _best_actions(self, cls: np.ndarray, q: np.ndarray, sigma: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Best action per class, or -1 (keep the average strategy).

        With ``br_z == 0`` this is the plain empirical best response. With
        ``br_z > 0`` a class deviates only if its best action beats the average
        strategy by more than ``br_z`` standard errors on these deals, which
        keeps sampling noise from masquerading as exploitability."""
        total = class_sums(cls, q * weight[:, None])  # (A, 169)
        if self.br_z <= 0:
            return total.argmax(axis=0)
        follow = (sigma * q).sum(axis=1)
        d = q - follow[:, None]
        sw = np.bincount(cls, weights=weight, minlength=NUM_CLASSES)
        sw2 = np.bincount(cls, weights=weight**2, minlength=NUM_CLASSES)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = class_sums(cls, d * weight[:, None]) / sw
            m2 = class_sums(cls, d * d * weight[:, None]) / sw
            se = np.sqrt(np.maximum(m2 - m1**2, 0) * sw2 / sw**2)
        best = np.nan_to_num(m1, nan=-np.inf).argmax(axis=0)
        cols = np.arange(NUM_CLASSES)
        gain, err = m1[best, cols], se[best, cols]
        significant = (sw > 0) & (sw2 > 0) & (gain > self.br_z * err) & (gain > 0)
        return np.where(significant, best, -1)

    def evaluate(self, deals: Deals, holdout: Deals | None = None) -> dict:
        """Expected value of the average strategy and each player's sampled best-response gain.

        In-sample gains (plain best response chosen and scored on the same
        deals) are biased upward by sampling noise. With ``holdout``, a
        conservative response (deviating only where the gain is significant at
        ``HOLDOUT_Z`` standard errors) is chosen on ``deals`` and scored on the
        independent ``holdout`` deals: an unbiased value for a real deviation,
        hence an estimate of a lower bound on the true gain.
        """
        table = self.average_strategy()
        idx = np.arange(len(deals))
        base = self._values(0, idx, deals, table).mean(axis=0)
        gains, holdout_gains = [], []
        hold_base = None
        if holdout is not None:
            hidx = np.arange(len(holdout))
            hold_base = self._values(0, hidx, holdout, table).mean(axis=0)
        for p in range(self.n):
            self.br_z = 0.0
            br = self._values(0, idx, deals, table, p, None, {})[:, p].mean()
            gains.append(br - base[p])
            if holdout is not None:
                best: dict = {}
                self.br_z = HOLDOUT_Z
                self._values(0, idx, deals, table, p, None, best)
                self.br_z = 0.0
                hv = self._values(0, hidx, holdout, table, p, best, None)[:, p].mean()
                holdout_gains.append(hv - hold_base[p])
        out = {
            "ev_bb": [round(float(x), 5) for x in base],
            "br_gain_bb100": [round(float(100 * g), 3) for g in gains],
            "nash_conv_bb100": round(float(100 * sum(gains)), 3),
            "eval_deals": len(deals),
        }
        if holdout is not None:
            out["holdout_br_gain_bb100"] = [round(float(100 * g), 3) for g in holdout_gains]
            out["holdout_nash_conv_bb100"] = round(float(100 * sum(holdout_gains)), 3)
        return out
