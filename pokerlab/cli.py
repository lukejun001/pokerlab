"""Command-line interface: ``python -m pokerlab <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pokerlab.analysis.reporting import agent_table, default_output_dir, export, render_report
from pokerlab.simulation.batch import run_batch
from pokerlab.simulation.configurations import list_presets, load_config

SUITE = [
    ("A", "zero_rake"),
    ("B", "rake_comparison"),
    ("C", "tight_table"),
    ("D", "aggressive_table"),
    ("E", "mixed_table"),
    ("F", "position_benchmark"),
]


def _overrides(args: argparse.Namespace) -> dict:
    rake = {}
    if args.rake is not None:
        rake["percentage"] = args.rake / 100 if args.rake >= 1 else args.rake
    if args.rake_cap is not None:
        rake["cap"] = None if args.rake_cap < 0 else args.rake_cap
    overrides = {
        "hands": args.hands,
        "players": args.players,
        "seed": args.seed,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "mode": args.mode,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "starting_stack_bb": args.stack_bb,
        "seat_assignment": args.seats,
        "rake": rake or None,
        "hand_log": True if args.hand_log else None,
        "strict": True if args.strict else None,
    }
    if args.strategy:
        overrides["_strategy"] = args.strategy
    return overrides


def _apply_strategy(config, strategy: str | None):
    if strategy:
        for entry in config.lineup:
            if entry["type"] == "Experimental":
                entry.setdefault("params", {})["strategy"] = strategy
    return config


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hands", type=int)
    p.add_argument("--players", type=int, help="2-9; trims or extends the lineup")
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, help="parallel processes (0 = all cores)")
    p.add_argument("--chunk-size", type=int, help="hands per independent chunk")
    p.add_argument("--mode", choices=["independent", "continuous"])
    p.add_argument("--small-blind", type=int)
    p.add_argument("--big-blind", type=int)
    p.add_argument("--stack-bb", type=float, help="starting stack in big blinds")
    p.add_argument("--seats", choices=["fixed", "rotate", "shuffle"], help="seat assignment per chunk")
    p.add_argument("--rake", type=float, help="rake percentage, e.g. 4 or 0.04")
    p.add_argument("--rake-cap", type=int, help="rake cap in chips (-1 = uncapped)")
    p.add_argument("--strategy", help="experimental agent strategy: equity | archetype:TAG | module:Class")
    p.add_argument("--out", help="output directory (default results/<name>_<timestamp>)")
    p.add_argument("--no-export", action="store_true", help="skip writing CSV/JSON results")
    p.add_argument("--hand-log", action="store_true", help="write every hand to JSONL (large)")
    p.add_argument("--strict", action="store_true", help="fail on any agent or engine error")
    p.add_argument("--quiet", action="store_true", help="no progress output")


def _load_simulation(args: argparse.Namespace, overrides: dict):
    source = args.config or args.preset
    if source:
        from pokerlab.solver.config import is_solver_config, load_solver_config

        if is_solver_config(source):
            from pokerlab.solver.pipeline import simulation_config

            solver_config = load_solver_config(source)
            if overrides.get("rake"):
                overrides["rake"] = {**simulation_config(solver_config).rake, **overrides["rake"]}
            return simulation_config(solver_config, overrides)
    return load_config(source, overrides)


def cmd_simulate(args: argparse.Namespace) -> int:
    overrides = _overrides(args)
    strategy = overrides.pop("_strategy", None)
    config = _apply_strategy(_load_simulation(args, overrides), strategy)
    out = Path(args.out) if args.out else default_output_dir(config.name, config.output_dir)
    print(f"Running '{config.name}': {config.hands:,} hands, {config.players} players, "
          f"seed {config.seed}, workers {config.workers or 'all'}", file=sys.stderr)
    result = run_batch(config, progress=not args.quiet, log_dir=out / "hands" if config.hand_log else None)
    print(render_report(result))
    if not args.no_export:
        export(result, out)
        print(f"\nResults written to {out}/ (summary.json, agents.csv, positions.csv, seats.csv)")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    overrides = _overrides(args)
    strategy = overrides.pop("_strategy", None)
    base_out = Path(args.out) if args.out else default_output_dir("suite")
    rows = []
    for letter, preset in SUITE:
        config = _apply_strategy(load_config(preset, overrides), strategy)
        print(f"\n[{letter}] {config.name}: {config.hands:,} hands", file=sys.stderr)
        result = run_batch(config, progress=not args.quiet)
        if args.verbose:
            print(render_report(result))
        if not args.no_export:
            export(result, base_out / f"{letter}_{config.name}")
        table = agent_table(result)
        hero = table[table.archetype == "Experimental"].iloc[0]
        rows.append((letter, config.name, result, hero))

    print("\nEXPERIMENT SUITE: experimental agent results")
    header = f"{'':<3}{'experiment':<22}{'hands':>9}{'net bb/100':>12}{'95% CI':>20}{'gross':>9}{'rake':>7}{'h/s':>7}"
    print(header)
    print("-" * len(header))
    for letter, name, result, hero in rows:
        ci = f"[{hero.bb100_ci95_low:+.1f}, {hero.bb100_ci95_high:+.1f}]"
        print(f"{letter:<3}{name:<22}{hero.hands:>9,}{hero.net_bb100:>+12.2f}{ci:>20}{hero.gross_bb100:>+9.2f}"
              f"{hero.rake_attributed_bb100:>7.1f}{result.hands / result.elapsed:>7.0f}")
    print("\nA and B use the same seed, so cards and (in independent mode) every decision match hand for hand;")
    print("their difference isolates the cost of rake. Intervals that include 0 are not evidence of an edge.")
    if not args.no_export:
        print(f"Results written to {base_out}/")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    for path in list_presets():
        config = load_config(str(path))
        kinds = ", ".join(e["type"] for e in config.lineup)
        print(f"{path.stem:<22} {config.hands:>8,} hands  rake {config.rake.get('percentage', 0):.0%}  [{kinds}]")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    import json

    from pokerlab.solver.config import load_solver_config
    from pokerlab.solver.pipeline import class_table, generate, inspect, write_json
    from pokerlab.solver.store import DEFAULT_STORE, StrategyStore

    store = StrategyStore(args.store or DEFAULT_STORE)
    if args.action == "list":
        rows = store.list()
        print(f"{len(rows)} strategies in {store.root} ({store.database_bytes() / 1e6:.2f} MB)")
        for r in rows:
            m = json.loads(r["metrics"])
            print(f"  {r['key'][:12]}  {r['players']}p  stacks {json.loads(r['stacks_bb'])}  {r['abstraction_name']}  "
                  f"{r['decision_nodes']:,} decision nodes  {m.get('iterations', '?')} iters  "
                  f"{r['size_bytes'] / 1e6:.2f} MB  [{m.get('config', '')}]")
        return 0
    if not args.config:
        print("--config is required", file=sys.stderr)
        return 2
    config = load_solver_config(args.config)
    if args.action == "generate":
        tree_note = f"{config.game.n} players, stacks {config.stacks_by_position_bb}"
        print(f"Solving '{config.name}': {tree_note}", file=sys.stderr)
        metrics = generate(config, store, iterations=args.iterations, workers=args.workers, progress=not args.quiet)
        final = metrics["convergence"][-1]
        print(f"Stored strategy {metrics['key'][:12]}: {metrics['tree']['decision_nodes']:,} decision nodes, "
              f"{metrics['iterations']} iterations, {metrics['deals_sampled']:,} sampled deals")
        print(f"Solve {metrics['solve_s']}s (wall {metrics['wall_s']}s, CPU utilization {metrics['cpu_utilization']:.0%} "
              f"of {metrics['cores']} cores), peak memory {metrics['peak_memory_parent_mb']} MB parent / "
              f"{metrics['peak_memory_worker_mb']} MB per worker, file {metrics['strategy_bytes'] / 1e6:.2f} MB")
        print(f"Sampled NashConv: {final['nash_conv_bb100']:.2f} bb/100 in-sample (biased up), "
              f"{final['holdout_nash_conv_bb100']:.2f} bb/100 holdout (lower bound)")
        if args.out:
            print(f"Metrics written to {write_json(args.out, metrics)}")
        return 0
    report = inspect(config, store)
    if report is None:
        print(f"No stored strategy for '{config.name}' (run `strategies generate --config {args.config}`)")
        return 1
    if args.history is not None:
        strategy = store.load(report["key"])
        rows = class_table(strategy, args.history)
        print(json.dumps(rows, indent=1) if args.json else "\n".join(
            f"{r['hand']:<5}" + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "hand") for r in rows))
        return 0
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    m = report["metrics"]
    print(f"Strategy {report['key'][:12]} for '{config.name}'")
    print(f"Stacks: {report['stacks_bb']}")
    print(f"Decision nodes: {report['decision_nodes']:,}; file {report['file_bytes'] / 1e6:.2f} MB; "
          f"first load {report['first_load_ms']} ms; cached lookup {report['cached_lookup_us']} us")
    print(f"Solved with {m.get('iterations')} iterations in {m.get('solve_s')}s")
    print("First-in action frequencies (everyone before folded):")
    for pos, freqs in report["first_in"].items():
        print(f"  {pos:<4} " + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in freqs.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerlab", description="Offline No-Limit Hold'em AI simulation lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run one experiment")
    source = sim.add_mutually_exclusive_group()
    source.add_argument("--config", help="path to a YAML/JSON config")
    source.add_argument("--preset", help="name of a config in configs/ (see `presets`)")
    _add_run_options(sim)
    sim.set_defaults(func=cmd_simulate)

    suite = sub.add_parser("suite", help="run experiments A-F and compare")
    _add_run_options(suite)
    suite.add_argument("--verbose", action="store_true", help="print the full report for each experiment")
    suite.set_defaults(func=cmd_suite)

    presets = sub.add_parser("presets", help="list ready-made experiment configs")
    presets.set_defaults(func=cmd_presets)

    strat = sub.add_parser("strategies", help="generate, inspect or list precomputed solver strategies")
    strat.add_argument("action", choices=["generate", "inspect", "list"])
    strat.add_argument("--config", help="solver config (YAML with a `game:` section)")
    strat.add_argument("--store", help="strategy store directory (default strategies/)")
    strat.add_argument("--iterations", type=int, help="override solve.iterations")
    strat.add_argument("--workers", type=int, help="override solve.workers (1 = single process)")
    strat.add_argument("--history", help="inspect: per-hand-class strategy at this betting history ('' = root)")
    strat.add_argument("--json", action="store_true", help="inspect: print JSON")
    strat.add_argument("--out", help="generate: also write the metrics JSON here")
    strat.add_argument("--quiet", action="store_true")
    strat.set_defaults(func=cmd_strategies)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
