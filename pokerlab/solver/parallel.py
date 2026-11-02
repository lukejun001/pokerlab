"""Multi-process sampling for the preflop solver.

Each iteration every worker deals its own batch (seeded by ``(seed, iteration,
worker)``), computes counterfactual regret increments against the current
strategy, and returns them; the parent sums them and performs the update.
The current strategy is shared through ``multiprocessing.shared_memory``.
Results depend on ``(seed, workers, batch_size)`` and nothing else.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import shared_memory

import numpy as np

from pokerlab.solver.cfr import HOLDOUT_Z, Deals, PreflopSolver
from pokerlab.solver.game import PreflopGame, PreflopTree

_STATE: dict = {}


def _init(game: PreflopGame, batch_size: int, seed: int, shm_name: str, shape: tuple[int, int]) -> None:
    solver = PreflopSolver(PreflopTree(game), batch_size=batch_size, seed=seed)
    shm = shared_memory.SharedMemory(name=shm_name)
    solver.current = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
    _STATE.update(solver=solver, shm=shm)


def _work(args: tuple[int, int]) -> np.ndarray:
    t, w = args
    solver: PreflopSolver = _STATE["solver"]
    deals = Deals.sample(np.random.default_rng([solver.seed, t, w]), solver.batch_size, solver.n)
    return solver.sampled_regrets(deals)


class ParallelSampler:
    """``with ParallelSampler(solver, workers) as ps: ps.iterate()``."""

    def __init__(self, solver: PreflopSolver, workers: int) -> None:
        self.solver = solver
        self.workers = workers
        shape = solver.current.shape
        self.shm = shared_memory.SharedMemory(create=True, size=solver.current.nbytes)
        self.shared = np.ndarray(shape, dtype=np.float64, buffer=self.shm.buf)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init,
                             initargs=(solver.game, solver.batch_size, solver.seed, self.shm.name, shape))

    def iterate(self) -> None:
        s = self.solver
        started, elapsed = time.perf_counter(), s.elapsed
        self.shared[:] = s.current
        t = s.iterations + 1
        total = None
        for inc in self.pool.imap_unordered(_work, [(t, w) for w in range(self.workers)]):
            total = inc if total is None else total + inc
        s.iterate(regret_inc=total, deal_count=self.workers * s.batch_size)
        s.elapsed = elapsed + time.perf_counter() - started  # wall time, workers included

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
        self.shm.close()
        self.shm.unlink()

    def __enter__(self) -> ParallelSampler:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _evaluate_player(args) -> dict:
    game, table, deals, seed, player = args
    solver = PreflopSolver(PreflopTree(game), batch_size=1, seed=0)
    sample = Deals.sample(np.random.default_rng([seed, 10**9]), deals, game.n)
    holdout = Deals.sample(np.random.default_rng([seed, 10**9 + 1]), deals, game.n)
    idx, hidx = np.arange(deals), np.arange(deals)
    base = solver._values(0, idx, sample, table).mean(axis=0)
    hold_base = solver._values(0, hidx, holdout, table).mean(axis=0)
    br = solver._values(0, idx, sample, table, player, None, {})[:, player].mean()
    best: dict = {}
    solver.br_z = HOLDOUT_Z
    solver._values(0, idx, sample, table, player, None, best)
    solver.br_z = 0.0
    hv = solver._values(0, hidx, holdout, table, player, best, None)[:, player].mean()
    return {"player": player, "base": base, "gain": br - base[player], "holdout_gain": hv - hold_base[player]}


def evaluate_parallel(solver: PreflopSolver, deals: int, seed: int, workers: int) -> dict:
    """``PreflopSolver.evaluate`` with one process per player (same deals for every player)."""
    table = solver.average_strategy()
    ctx = mp.get_context("spawn")
    jobs = [(solver.game, table, deals, seed, p) for p in range(solver.n)]
    with ctx.Pool(min(workers, solver.n)) as pool:
        results = sorted(pool.map(_evaluate_player, jobs), key=lambda r: r["player"])
    gains = [r["gain"] for r in results]
    holdout = [r["holdout_gain"] for r in results]
    return {
        "ev_bb": [round(float(x), 5) for x in results[0]["base"]],
        "br_gain_bb100": [round(float(100 * g), 3) for g in gains],
        "nash_conv_bb100": round(float(100 * sum(gains)), 3),
        "holdout_br_gain_bb100": [round(float(100 * g), 3) for g in holdout],
        "holdout_nash_conv_bb100": round(float(100 * sum(holdout)), 3),
        "eval_deals": deals,
    }
