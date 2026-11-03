"""Phase 3 experiments: solve the six-max configs, compare strategies, simulate.

    python tools/phase3_experiments.py solve        # generate every config (bounded budgets)
    python tools/phase3_experiments.py compare      # strategy differences between solved games
    python tools/phase3_experiments.py simulate     # offline simulations with cached strategies

Everything printed is measured; outputs land in reports/phase3/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pokerlab.analysis.reporting import agent_table, export, render_report
from pokerlab.simulation.batch import run_batch
from pokerlab.solver.config import load_solver_config
from pokerlab.solver.game import action_kind
from pokerlab.solver.hands import CLASS_COMBOS, CLASS_INDEX
from pokerlab.solver.pipeline import first_in_summary, generate, inspect, simulation_config, write_json
from pokerlab.solver.store import StrategyStore, strategy_key

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase3"
SOLVE = ["6max_gtopen_equal", "6max_gtopen_asymmetric", "6max_gtopen_asymmetric_allin"]
SOLVE_POKERLAB = ["6max_equal", "6max_asymmetric", "6max_asymmetric_zero_rake", "6max_asymmetric_allin"]
SIMULATE = ["6max_gtopen_asymmetric", "6max_gtopen_asymmetric_selfplay", "6max_gtopen_equal"]
COMPARE_EXTRA = ["6max_gtopen_asymmetric_rot1"]


def cmd_solve(args) -> None:
    store = StrategyStore()
    for name in args.configs or SOLVE:
        config = load_solver_config(name)
        print(f"\n== {name}: stacks {config.stacks_by_position_bb}", file=sys.stderr)
        metrics = generate(config, store, iterations=args.iterations, workers=args.workers)
        write_json(OUT / f"solve_{name}.json", metrics)
        summary = ("key", "iterations", "solve_s", "wall_s", "cpu_utilization", "peak_memory_mb",
                   "peak_memory_parent_mb", "peak_memory_worker_mb", "final_nash_conv_bb100", "strategy_bytes")
        print(json.dumps({k: metrics[k] for k in summary if k in metrics}, indent=1))


def weighted(row: np.ndarray, mask: np.ndarray | None = None) -> float:
    w = CLASS_COMBOS if mask is None else CLASS_COMBOS * mask
    return float((row * w).sum() / w.sum())


def node_summary(strategy, history: str) -> dict[str, float] | None:
    found = strategy.node_table(history)
    if found is None:
        return None
    actions, table = found
    return {token: round(weighted(table[k]), 4) for k, token in enumerate(actions)}


def hand_rows(strategy, history: str, hands: list[str]) -> dict[str, dict[str, float]] | None:
    found = strategy.node_table(history)
    if found is None:
        return None
    actions, table = found
    return {h: {a: round(float(table[k, CLASS_INDEX[h]]), 3) for k, a in enumerate(actions)} for h in hands}


def cmd_compare(args) -> None:
    store = StrategyStore()
    solved = {}
    for name in SOLVE + COMPARE_EXTRA:
        config = load_solver_config(name)
        strategy = store.load(strategy_key(config.game))
        if strategy is not None:
            solved[name] = (config, strategy)
    report: dict = {"first_in": {}, "spots": {}, "hands": {}, "inspect": {}}
    for name, (config, strategy) in solved.items():
        report["first_in"][name] = first_in_summary(strategy, config.game)
        info = inspect(config, store)
        report["inspect"][name] = {k: info[k] for k in ("key", "decision_nodes", "file_bytes", "first_load_ms",
                                                         "cached_lookup_us")}
    # Spots that exist (with different all-in amounts) in every six-max game: a jam is
    # identified by its kind, not its amount, so equal and asymmetric games line up.
    spots = {
        "UTG first in": [],
        "BTN first in": ["f", "f", "f"],
        "SB first in": ["f", "f", "f", "f"],
        "BB vs SB open": ["f", "f", "f", "f", "r15000"],
        "BB vs SB all-in": ["f", "f", "f", "f", "a*"],
        "SB vs BTN open": ["f", "f", "f", "r12500"],
        "BB vs BTN open, SB folds": ["f", "f", "f", "r12500", "f"],
        "UTG vs HJ 3-bet": ["r12500", "r37500", "f", "f", "f", "f"],
    }
    probe = ["AA", "AKs", "AQo", "TT", "77", "A5s", "KJo", "QTs", "98s", "55", "K9o", "J8o", "72o"]
    for name, (config, strategy) in solved.items():
        for label, pattern in spots.items():
            history = resolve(strategy, pattern)
            if history is None:
                continue
            report["spots"].setdefault(label, {})[name] = {"history": history, "freq": node_summary(strategy, history)}
            report["hands"].setdefault(label, {})[name] = hand_rows(strategy, history, probe)
    # Distance between solutions of the equal and asymmetric games at shared spots.
    pairs = {
        "equal_vs_asymmetric_distance": ("6max_gtopen_equal", "6max_gtopen_asymmetric"),
        "asymmetric_vs_rotated_distance": ("6max_gtopen_asymmetric", "6max_gtopen_asymmetric_rot1"),
        "asymmetric_vs_rejam_tree_distance": ("6max_gtopen_asymmetric", "6max_gtopen_asymmetric_allin"),
    }
    for field, (a, b) in pairs.items():
        if a in solved and b in solved:
            report[field] = {label: strategy_distance(solved, label, report, a=a, b=b)
                             for label, by_game in report["spots"].items() if {a, b} <= set(by_game)}
    write_json(OUT / "comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("hands", "spots")}, indent=1))


def resolve(strategy, pattern: list[str]) -> str | None:
    """Follow a history through a stored strategy; ``a*`` means "the all-in
    action, whatever its size", so games with different stacks line up."""
    history: list[str] = []
    for token in pattern:
        node = strategy.nodes.get("-".join(history))
        if node is None:
            return None
        actions = node[1]
        match = [t for t in actions if (action_kind(t) == "a" if token == "a*" else t == token)]
        if not match:
            return None
        history.append(match[0])
    return "-".join(history) if "-".join(history) in strategy.nodes else None


def strategy_distance(solved, label, report, a="6max_gtopen_equal", b="6max_gtopen_asymmetric") -> dict:
    """Combo-weighted total variation distance between two games' strategies at one spot,
    comparing actions by kind (fold / call / sized raise / all-in)."""
    out = {}
    tables = {}
    for name in (a, b):
        history = report["spots"][label][name]["history"]
        actions, table = solved[name][1].node_table(history)
        kinds: dict[str, np.ndarray] = {}
        for k, token in enumerate(actions):
            kind = {"f": "fold", "x": "check", "c": "call", "r": "raise", "a": "allin"}[action_kind(token)]
            kinds[kind] = kinds.get(kind, 0) + table[k]
        tables[name] = kinds
    kinds = set(tables[a]) | set(tables[b])
    tv = 0.5 * sum(np.abs(tables[a].get(k, 0) - tables[b].get(k, 0)) for k in kinds)
    out["mean_tv"] = round(weighted(tv), 4)
    out["max_tv_over_classes"] = round(float(np.max(tv)), 4)
    out["share_of_combos_changing_over_10pct"] = round(weighted((tv > 0.10).astype(float)), 4)
    return out


def cmd_simulate(args) -> None:
    rows = {}
    for name in args.configs or SIMULATE:
        config = load_solver_config(name)
        sim = simulation_config(config, {"hands": args.hands, "workers": args.workers, "chunk_size": 1000})
        started = time.perf_counter()
        result = run_batch(sim, progress=False)
        text = render_report(result)
        (OUT / f"simulate_{name}.txt").parent.mkdir(parents=True, exist_ok=True)
        (OUT / f"simulate_{name}.txt").write_text(text)
        export(result, OUT / f"simulate_{name}")
        table = agent_table(result)
        sources = {s.agent_id: s.decision_sources for s in result.stats.agents.values() if s.decision_sources}
        rows[name] = {
            "hands": result.stats.table.hands,
            "seconds": round(time.perf_counter() - started, 1),
            "rake_chips": result.stats.table.rake,
            "agents": table[["agent", "archetype", "net_bb100", "bb100_ci95_low", "bb100_ci95_high",
                             "gross_bb100", "vpip", "pfr"]].to_dict(orient="records"),
            "decision_sources": sources,
        }
        print(text)
    write_json(OUT / "simulations.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("solve")
    p.add_argument("configs", nargs="*")
    p.add_argument("--iterations", type=int)
    p.add_argument("--workers", type=int)
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("compare")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("simulate")
    p.add_argument("configs", nargs="*")
    p.add_argument("--hands", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_simulate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
