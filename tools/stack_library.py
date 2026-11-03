"""Build and measure the stack-vector library for approximate strategies.

    python tools/stack_library.py build --count 16 --iterations 200      # solve random stack vectors
    python tools/stack_library.py measure --tests 30                      # score approximations exactly
    python tools/stack_library.py curve --tests 10                        # error versus library size
    python tools/stack_library.py calibrate --threshold 1.0               # write strategies/approximation.json

Stacks are drawn log-uniformly per seat from [20, 250] bb (integers), with a
fixed seed. Everything else (blinds, rake, abstraction) comes from
configs/6max_gtopen_asymmetric.yaml. Results go to reports/phase3/library/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path

from pokerlab.solver import gtopen
from pokerlab.solver.config import load_solver_config
from pokerlab.solver.generalize import Library, approximate_profile, calibrate
from pokerlab.solver.pipeline import write_json
from pokerlab.solver.store import StrategyStore, strategy_key

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase3" / "library"
BASE = "6max_gtopen_asymmetric"
LOW, HIGH = 20, 250


def random_stacks(rng: random.Random, n: int = 6) -> list[int]:
    return [int(round(math.exp(rng.uniform(math.log(LOW), math.log(HIGH))))) for _ in range(n)]


def game_for(stacks_bb):
    base = load_solver_config(BASE)
    g = base.game
    return base, dataclasses.replace(g, stacks=tuple(int(s * g.big_blind) for s in stacks_bb))


def cmd_build(args) -> None:
    rng = random.Random(args.seed)
    vectors = [random_stacks(rng) for _ in range(args.count)]
    store = StrategyStore()
    started = time.perf_counter()
    for i, stacks in enumerate(vectors):
        base, game = game_for(stacks)
        if store.lookup(game):
            print(f"[{i}] {stacks} already solved", file=sys.stderr)
            continue
        if args.time_limit_h and time.perf_counter() - started > args.time_limit_h * 3600:
            print("time limit reached", file=sys.stderr)
            break
        print(f"[{i}] solving {stacks}", file=sys.stderr, flush=True)
        config = dataclasses.replace(base, name=f"library_{args.seed}_{i:03d}", game=game,
                                     solve=dataclasses.replace(base.solve, iterations=args.iterations, eval_every=args.iterations))
        metrics = gtopen.generate(config, store, iterations=args.iterations, threads=args.threads, progress=False)
        write_json(OUT / f"solve_{config.name}.json", metrics)
        print(f"    NashConv {metrics['final_nash_conv_bb100']} bb/100 in {metrics['solve_s']:.0f}s", file=sys.stderr, flush=True)


def cmd_measure(args) -> None:
    store = StrategyStore()
    _, template = game_for([100] * 6)
    library = Library(store, template)
    rng = random.Random(args.seed)
    tests = [random_stacks(rng) for _ in range(args.tests)]
    rows = []
    jobs = [("holdout", s, None) for s in tests]
    if args.leave_one_out:
        jobs += [("leave_one_out", s, key) for key, s in library.entries]
    for kind, stacks, exclude in jobs:
        _, target = game_for(stacks)
        tree = gtopen.tree(target)
        for k in args.k:
            t0 = time.perf_counter()
            profile, info = approximate_profile(library, target, tree, k=k, exclude=exclude)
            ev = gtopen.evaluate(target, profile, threads=args.threads)
            row = {"set": kind, "stacks_bb": stacks, "k": k, "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3),
                   "max_gap_bb100": round(100 * max(ev["br_gaps_bb"]), 3),
                   "br_gap_bb100": [round(100 * g, 3) for g in ev["br_gaps_bb"]],
                   "uniform_nodes": ev["uniform_nodes"], "nodes": len(tree), **info,
                   "seconds": round(time.perf_counter() - t0, 1)}
            rows.append(row)
            print(json.dumps({k2: row[k2] for k2 in ("set", "stacks_bb", "k", "nash_conv_bb100", "max_gap_bb100",
                                                     "distance", "node_distance_p90", "uniform_nodes")}), flush=True)
    write_json(OUT / "measurements.json", {"library": library.entries, "rows": rows})


def cmd_curve(args) -> None:
    """Error versus library size: random sub-libraries scored on the same
    held-out vectors, plus the all-uniform profile as a floor-of-skill baseline."""
    store = StrategyStore()
    _, template = game_for([100] * 6)
    library = Library(store, template)
    everything = list(library.entries)
    rng = random.Random(args.seed)
    tests = [random_stacks(rng) for _ in range(args.tests)]
    plan = [(n, draw) for n in args.sizes if n < len(everything) for draw in range(args.draws)]
    plan.append((len(everything), 0))
    rows = []
    for stacks in tests:
        _, target = game_for(stacks)
        tree = gtopen.tree(target)
        ev = gtopen.evaluate(target, {}, threads=args.threads)
        rows.append({"stacks_bb": stacks, "library_size": 0, "draw": 0, "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3)})
        for n, draw in plan:
            library.entries = random.Random(1000 * n + draw).sample(everything, n)
            profile, info = approximate_profile(library, target, tree, k=1)
            ev = gtopen.evaluate(target, profile, threads=args.threads)
            rows.append({"stacks_bb": stacks, "library_size": n, "draw": draw,
                         "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3), **info})
            print(json.dumps({k: rows[-1][k] for k in ("stacks_bb", "library_size", "draw", "nash_conv_bb100")}), flush=True)
    library.entries = everything
    write_json(OUT / "curve.json", {"library": everything, "rows": rows})


def cmd_calibrate(args) -> None:
    data = json.loads((OUT / "measurements.json").read_text())
    rows = [r for r in data["rows"] if r["set"] == "holdout"]
    store = StrategyStore()
    _, template = game_for([100] * 6)
    library = Library(store, template)
    cal = calibrate(rows, library.family, args.k, args.threshold, stat=args.stat)
    cal.save(store.root / "approximation.json")
    print(json.dumps(dataclasses.asdict(cal), indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--count", type=int, default=16)
    b.add_argument("--iterations", type=int, default=200)
    b.add_argument("--threads", type=int, default=10)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--time-limit-h", type=float)
    b.set_defaults(func=cmd_build)
    m = sub.add_parser("measure")
    m.add_argument("--tests", type=int, default=30)
    m.add_argument("--k", type=int, nargs="+", default=[1, 3])
    m.add_argument("--threads", type=int, default=10)
    m.add_argument("--seed", type=int, default=11)
    m.add_argument("--leave-one-out", action="store_true")
    m.set_defaults(func=cmd_measure)
    v = sub.add_parser("curve")
    v.add_argument("--tests", type=int, default=10)
    v.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    v.add_argument("--draws", type=int, default=2)
    v.add_argument("--threads", type=int, default=10)
    v.add_argument("--seed", type=int, default=11)
    v.set_defaults(func=cmd_curve)
    c = sub.add_parser("calibrate")
    c.add_argument("--threshold", type=float, default=1.0)
    c.add_argument("--k", type=int, default=1)
    c.add_argument("--stat", default="node_distance_p90")
    c.set_defaults(func=cmd_calibrate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
