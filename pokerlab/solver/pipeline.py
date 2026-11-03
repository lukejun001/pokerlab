"""Generate, store and inspect solver strategies; build simulations from solver configs."""

from __future__ import annotations

import copy
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pokerlab.simulation.configurations import MIXED_LINEUP, SimulationConfig
from pokerlab.solver.cfr import CONTINUATION_VERSION, SOLVER_VERSION, Deals, PreflopSolver, peak_memory_mb
from pokerlab.solver.config import SolverConfig
from pokerlab.solver.game import PreflopTree, action_kind
from pokerlab.solver.hands import CLASS_COMBOS, CLASS_LABELS
from pokerlab.solver.parallel import ParallelSampler, evaluate_parallel
from pokerlab.solver.store import StrategyStore, strategy_key


def _children_peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def generate(config: SolverConfig, store: StrategyStore, iterations: int | None = None,
             workers: int | None = None, progress: bool = True) -> dict[str, Any]:
    """Solve ``config``'s game, store the average strategy, return measured metrics."""
    from pokerlab.solver.gtopen import GTOpenAbstraction

    if isinstance(config.game.abstraction, GTOpenAbstraction):
        from pokerlab.solver import gtopen

        return gtopen.generate(config, store, iterations=iterations, threads=workers, progress=progress)
    s = config.solve
    iterations = iterations or s.iterations
    workers = s.workers if workers is None else workers
    wall0, cpu0 = time.perf_counter(), os.times()
    tree = PreflopTree(config.game)
    build_s = time.perf_counter() - wall0
    solver = PreflopSolver(tree, batch_size=s.batch_size, seed=s.seed)
    checkpoints = sorted({c for c in range(s.eval_every, iterations + 1, s.eval_every)} | {iterations}) if s.eval_every else [iterations]
    history = []
    previous = None
    sampler = ParallelSampler(solver, workers) if workers > 1 else None
    stopped_early = False
    try:
        for target in checkpoints:
            while solver.iterations < target:
                sampler.iterate() if sampler else solver.iterate()
                if s.time_limit_s and solver.elapsed >= s.time_limit_s:
                    stopped_early = True
                    break
            t0 = time.perf_counter()
            if workers > 1:
                report = evaluate_parallel(solver, s.eval_deals, s.seed, workers)
            else:
                eval_set = Deals.sample(np.random.default_rng([s.seed, 10**9]), s.eval_deals, config.game.n)
                holdout = Deals.sample(np.random.default_rng([s.seed, 10**9 + 1]), s.eval_deals, config.game.n)
                report = solver.evaluate(eval_set, holdout)
            avg = solver.average_strategy()
            report.update(iteration=solver.iterations, solve_s=round(solver.elapsed, 2),
                          eval_s=round(time.perf_counter() - t0, 2),
                          max_strategy_change=None if previous is None else round(float(np.abs(avg - previous).max()), 4),
                          mean_strategy_change=None if previous is None else round(float(np.abs(avg - previous).mean()), 5))
            previous = avg
            history.append(report)
            if progress:
                print(f"  iter {solver.iterations:>5}  solve {solver.elapsed:7.1f}s  NashConv in-sample "
                      f"{report['nash_conv_bb100']:8.2f}  holdout {report['holdout_nash_conv_bb100']:8.2f} bb/100",
                      file=sys.stderr)
            if stopped_early:
                break
    finally:
        if sampler:
            sampler.close()
    wall = time.perf_counter() - wall0
    cpu1 = os.times()
    cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system) + (cpu1.children_user - cpu0.children_user) \
        + (cpu1.children_system - cpu0.children_system)
    stats = tree.stats()
    metrics = {
        "config": config.name,
        "players": config.game.n,
        "positions": list(config.game.positions),
        "stacks_bb": list(config.game.stacks_bb()),
        "abstraction": config.game.abstraction.to_dict(),
        "tree": stats,
        "tree_build_s": round(build_s, 3),
        "iterations": solver.iterations,
        "deals_sampled": solver.iterations * s.batch_size * max(1, workers),
        "batch_size": s.batch_size,
        "workers": workers,
        "seed": s.seed,
        "stopped_by_time_limit": stopped_early,
        "solve_s": round(solver.elapsed, 2),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "cpu_utilization": round(cpu / wall / (os.cpu_count() or 1), 3),
        "cores": os.cpu_count(),
        "gpu": "not used (no GPU code path; CPU NumPy only)",
        "peak_memory_parent_mb": round(peak_memory_mb(), 1),
        "peak_memory_worker_mb": round(_children_peak_mb(), 1) if workers > 1 else None,
        "convergence": history,
        "solver_version": SOLVER_VERSION,
        "continuation_version": CONTINUATION_VERSION,
    }
    key = store.save(solver, metrics)
    row = next(r for r in store.list() if r["key"] == key)
    metrics["key"] = key
    metrics["strategy_bytes"] = row["size_bytes"]
    store.db.execute("UPDATE strategies SET metrics = ? WHERE key = ?", (json.dumps(metrics), key))
    store.db.commit()
    return metrics


def first_in_summary(strategy, game) -> dict[str, dict[str, float]]:
    """Combo-weighted action frequencies when everyone before the actor folded."""
    out = {}
    names = game.positions
    for depth in range(game.n - 1):
        found = strategy.node_table("-".join(["f"] * depth))
        if found is None:
            continue
        actions, table = found
        weights = CLASS_COMBOS / CLASS_COMBOS.sum()
        freqs = {}
        for token, row in zip(actions, table):
            label = {"f": "fold", "c": "limp", "x": "check"}.get(action_kind(token), token)
            freqs[label] = round(float((row * weights).sum()), 4)
        out[names[depth]] = freqs
    return out


def class_table(strategy, history: str) -> list[dict[str, Any]]:
    found = strategy.node_table(history)
    if found is None:
        raise KeyError(f"history {history!r} is not a decision node of this strategy")
    actions, table = found
    return [{"hand": CLASS_LABELS[c], **{a: round(float(table[k, c]), 4) for k, a in enumerate(actions)}}
            for c in range(len(CLASS_LABELS))]


def inspect(config: SolverConfig, store: StrategyStore) -> dict[str, Any] | None:
    key = strategy_key(config.game)
    t0 = time.perf_counter()
    strategy = store.load(key)
    load_s = time.perf_counter() - t0
    if strategy is None:
        return None
    t0 = time.perf_counter()
    lookups = 20_000
    for k in range(lookups):
        store.lookup(config.game).strategy.action_probabilities("", k % 169)
    lookup_us = (time.perf_counter() - t0) / lookups * 1e6
    return {
        "key": key,
        "stacks_bb": dict(zip(config.game.positions, config.game.stacks_bb())),
        "decision_nodes": len(strategy.nodes),
        "file_bytes": strategy.meta["size_bytes"],
        "first_load_ms": round(1000 * load_s, 2),
        "cached_lookup_us": round(lookup_us, 2),
        "first_in": first_in_summary(strategy, config.game),
        "metrics": strategy.meta["metrics"],
    }


def simulation_config(config: SolverConfig, overrides: dict[str, Any] | None = None) -> SimulationConfig:
    """An offline simulation of the solved game: same blinds, per-position stacks and rake."""
    sim = copy.deepcopy(config.simulation)
    lineup = sim.pop("lineup", None) or copy.deepcopy(MIXED_LINEUP[: config.game.n])
    for entry in lineup:
        if entry["type"] == "Experimental":
            params = entry.setdefault("params", {})
            params.setdefault("strategy", "solver")
            sp = params.setdefault("strategy_params", {})
            sp.setdefault("config", str(config.path) if config.path else config.name)
    rake = config.game.rake
    data = {
        "name": config.name,
        "players": config.game.n,
        "small_blind": config.game.small_blind,
        "big_blind": config.game.big_blind,
        "starting_stack_bb": max(config.game.stacks_bb()),
        "stacks_by_position_bb": config.stacks_by_position_bb,
        "rake": {"percentage": rake.percentage, "cap": rake.cap,
                 "cap_by_players": dict(rake.cap_by_players), "no_flop_no_drop": rake.no_flop_no_drop},
        "lineup": lineup,
        **sim,
    }
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value
    return SimulationConfig(**data)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
