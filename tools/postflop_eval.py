"""Stage 2 evaluation: heads-up postflop search against the heuristic fallback.

    python tools/postflop_eval.py --hands 200 --label main

Plays the same seeded 6-max table twice with the hero on the preflop
network: once with ``postflop="search"`` and once with the heuristic postflop
fallback. Reports hero bb/100 (with its standard error: a few hundred hands
are far too few to separate the two, so this is a sanity check, not a
verdict), decision sources, and every street solve's exploitability within
its own tree and time (the quality measure of the search itself).
Results go to reports/phase4/postflop_<label>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from pokerlab.simulation.batch import run_batch
from pokerlab.simulation.configurations import SimulationConfig
from pokerlab.solver.pipeline import write_json

ROOT = Path(__file__).resolve().parents[1]
AB = {"engine": "gtopen", "name": "gtopen-6max-v1", "open_raises": [2.5], "sb_open_raises": [3.0],
      "raise_mults": [3.0], "max_raises": 3, "add_allin": True, "allin_threshold": 0.5, "limp": False,
      "realization": "balanced", "multiway_equity_model": "coupled_deck_v1"}
STACKS = {"UTG": 63, "HJ": 82, "CO": 206, "BTN": 65, "SB": 72, "BB": 88}  # a held-out vector (seed 11, #0)


def run(args, postflop: str, log_path: Path | None) -> dict:
    params = {"strategy": "solver", "strategy_params": {
        "abstraction": AB, "approximate": "off", "network": args.network, "postflop": postflop,
        "search_budget": {"flop": args.flop_s, "turn": args.turn_s, "river": args.river_s},
        "search_threads": args.threads}}
    sim = SimulationConfig(players=6, hands=args.hands, chunk_size=args.hands, workers=1, strict=True,
                           small_blind=2500, big_blind=5000, stacks_by_position_bb=STACKS, seed=args.seed,
                           rake={"percentage": 0.045, "cap": 300, "no_flop_no_drop": True},
                           lineup=[{"type": "Experimental", "id": "Hero", "params": params}, {"type": "TAG"},
                                   {"type": "LAG"}, {"type": "CallingStation"}, {"type": "Nit"}, {"type": "Balanced"}])
    if log_path:
        os.environ["POKERLAB_SEARCH_LOG"] = str(log_path)
    t0 = time.perf_counter()
    result = run_batch(sim, progress=False)
    os.environ.pop("POKERLAB_SEARCH_LOG", None)
    a = result.stats.agents["Hero"]
    bb100 = 100 * a.net / 5000 / a.hands
    var = a.net_sq_bb / a.hands - (a.net / 5000 / a.hands) ** 2
    return {"postflop": postflop, "hands": a.hands, "bb_per_100": round(bb100, 1),
            "se_bb_per_100": round(100 * math.sqrt(max(var, 0) / a.hands), 1),
            "decision_sources": dict(a.decision_sources), "invalid_actions": result.stats.table.invalid_actions,
            "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hands", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--network", default="models/lib16.pt")
    p.add_argument("--flop-s", type=float, default=45)
    p.add_argument("--turn-s", type=float, default=15)
    p.add_argument("--river-s", type=float, default=5)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--label", default="main")
    args = p.parse_args()
    out = ROOT / "reports" / "phase4" / f"postflop_{args.label}.json"
    log = ROOT / "reports" / "phase4" / f"postflop_{args.label}_solves.jsonl"
    log.unlink(missing_ok=True)
    rows = [run(args, "search", log)]
    print(json.dumps(rows[-1]), flush=True)
    rows.append(run(args, "fallback", None))
    print(json.dumps(rows[-1]), flush=True)
    solves = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    summary = {}
    for street in ("flop", "turn", "river"):
        xs = [s for s in solves if s["street"] == street]
        if xs:
            summary[street] = {"solves": len(xs),
                               "median_exploit_pct_pot": float(np.median([s["exploit_pct_pot"] for s in xs])),
                               "p90_exploit_pct_pot": float(np.quantile([s["exploit_pct_pot"] for s in xs], 0.9)),
                               "median_solve_s": float(np.median([s["solve_s"] for s in xs])),
                               "median_iterations": float(np.median([s["iterations"] for s in xs]))}
    print(json.dumps(summary, indent=1))
    write_json(out, {"args": vars(args), "runs": rows, "solve_summary": summary})


if __name__ == "__main__":
    sys.exit(main())
