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


def cmd_simulate(args: argparse.Namespace) -> int:
    overrides = _overrides(args)
    strategy = overrides.pop("_strategy", None)
    config = _apply_strategy(load_config(args.config or args.preset, overrides), strategy)
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
