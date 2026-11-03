"""Does warm-starting from the nearest library solve speed up a new stack vector?

    python tools/warm_start.py compare --tests 3 --iterations 200 --check-every 20 --threads 4
    python tools/warm_start.py compare --tests 1 --iterations 60 --check-every 10 \\
        --variants cold r1s1t0 r1s1t20 r4s4t50      # quick tuning of the warm-start knobs

A variant name ``r<regret_scale>s<sum_scale>t<warm_iterations>`` is a warm start
from the k=1 per-decision approximation (``solver/generalize.py``); ``cold``
starts from uniform play. Every variant solves the same held-out vector
(drawn like ``stack_library.py measure``, same seed) and records GTOpen's exact
NashConv at each checkpoint. Nothing is written to the strategy store.
Results go to reports/phase3/warm/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack_library import game_for, random_stacks  # noqa: E402

from pokerlab.solver import gtopen  # noqa: E402
from pokerlab.solver.generalize import Library, approximate_profile  # noqa: E402
from pokerlab.solver.pipeline import write_json  # noqa: E402
from pokerlab.solver.store import StrategyStore  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase3" / "warm"
VARIANT = re.compile(r"r([\d.]+)s([\d.]+)t(\d+)$")


def parse_variant(name: str, profile) -> gtopen.WarmStart | None:
    if name == "cold":
        return None
    m = VARIANT.match(name)
    if not m:
        raise SystemExit(f"bad variant {name!r}: use cold or r<regret>s<sum>t<iterations>")
    return gtopen.WarmStart(profile, float(m.group(1)), float(m.group(2)), int(m.group(3)))


def cmd_compare(args) -> None:
    store = StrategyStore()
    _, template = game_for([100] * 6)
    library = Library(store, template)
    rng = random.Random(args.seed)
    tests = [random_stacks(rng) for _ in range(args.skip + args.tests)][args.skip:]
    path = OUT / f"compare_{args.label}.json"
    results = json.loads(path.read_text())["runs"] if path.exists() else []
    done = {(tuple(r["stacks_bb"]), r["variant"]) for r in results}
    for stacks in tests:
        _, target = game_for(stacks)
        tree = gtopen.tree(target)
        t0 = time.perf_counter()
        profile, info = approximate_profile(library, target, tree, k=1)
        approx_s = time.perf_counter() - t0
        for name in args.variants:
            if (tuple(stacks), name) in done:
                continue
            warm = parse_variant(name, profile)
            print(f"{stacks} {name}", file=sys.stderr, flush=True)
            out = gtopen.run(target, args.iterations, check_every=args.check_every, threads=args.threads,
                             progress=False, warm_start=warm)
            m = out["metrics"]
            curve = [{"iteration": c["iteration"], "solve_s": round(c["solve_s"], 1),
                      "nash_conv_bb100": round(100 * c["nash_conv_bb"], 3)} for c in m["checkpoints"]]
            results.append({"stacks_bb": stacks, "variant": name, "approximation": info,
                            "approximation_s": round(approx_s, 1), "warm_start": m.get("warm_start"),
                            "threads": m["threads"], "wall_s": m["wall_s"], "curve": curve})
            print(json.dumps({"stacks": stacks, "variant": name,
                              "curve": [(c["iteration"], c["nash_conv_bb100"]) for c in curve]}), flush=True)
            write_json(path, {"library_size": len(library), "runs": results})


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--tests", type=int, default=3)
    c.add_argument("--skip", type=int, default=0, help="start at this held-out vector")
    c.add_argument("--iterations", type=int, default=200)
    c.add_argument("--check-every", type=int, default=20)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--seed", type=int, default=11)
    c.add_argument("--variants", nargs="+", default=["cold", "r1s1t20"])
    c.add_argument("--label", default="main")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
