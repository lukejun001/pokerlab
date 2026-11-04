"""Training data for the preflop strategy network: GTOpen solves of random stack vectors.

    python tools/preflop_data.py generate --count 200 --iterations 100 --threads 4 [--push-every 1]
    python tools/preflop_data.py summary

Each solve of a random 6-max stack vector (log-uniform 20-250 bb per seat,
seed 101; the library used seed 7, held-out tests use seed 11) is saved as
``data/preflop/solve_<seed>_<index>.npz``, small enough to commit:

- ``histories``/``actions``/``actors``: the decision nodes kept (zlib JSON);
- ``probs``: (edges, 169) uint8, the average strategy times 255 (pokerlab class order);
- ``actor_reach``: (nodes, 169) float16, the actor's own reach of the node per hand class;
- ``opp_reach``: (nodes,) float32, product over the other seats of their mean reach;
- ``meta``: stacks, solve metrics and final exact NashConv (JSON).

Nodes are kept when the actor reaches them with some hand class (largest
own reach at least ``--min-reach``): a node only opponents' deviations reach
still matters to exploitability, since best responses deviate into it. The base game (blinds, rake, abstraction) is
configs/6max_gtopen_asymmetric.yaml; only the stacks vary.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack_library import game_for, random_stacks  # noqa: E402

from pokerlab.solver import gtopen  # noqa: E402
from pokerlab.solver.hands import CLASS_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "preflop"


def class_weights() -> np.ndarray:
    """Combos per class (pairs 6, suited 4, offsuit 12), normalized."""
    w = np.array([6.0 if len(c) == 2 else (4.0 if c.endswith("s") else 12.0) for c in CLASS_LABELS])
    return w / w.sum()


def compact(out: dict, n: int, min_reach: float) -> dict[str, np.ndarray]:
    order = [out["class_labels"].index(label) for label in CLASS_LABELS]
    nodes = {node["history"]: node for node in out["nodes"]}
    w0 = class_weights()
    reach = {"": np.tile(w0, (n, 1))}  # per seat, per class (combo-weighted)
    keep_h, keep_a, keep_actor, probs, actor_reach, opp_reach = [], [], [], [], [], []
    for history, node in nodes.items():  # parents precede children in the export (depth first)
        r = reach.pop(history)
        actor = node["actor"]
        table = np.asarray(node["strategy"], dtype=np.float64)[:, order]
        opp = float(np.prod([r[i].sum() for i in range(n) if i != actor]))
        if (r[actor] / w0).max() >= min_reach:
            keep_h.append(history)
            keep_a.append(node["actions"])
            keep_actor.append(actor)
            probs.append(np.round(table * 255).astype(np.uint8))
            actor_reach.append((r[actor] / w0).astype(np.float16))
            opp_reach.append(opp)
        for a, token in enumerate(node["actions"]):
            child = f"{history}-{token}" if history else token
            if child in nodes:
                rc = r.copy()
                rc[actor] = r[actor] * table[a]
                reach[child] = rc
    return {
        "histories": np.frombuffer(zlib.compress(json.dumps(keep_h).encode()), dtype=np.uint8),
        "actions": np.frombuffer(zlib.compress(json.dumps(keep_a).encode()), dtype=np.uint8),
        "actors": np.asarray(keep_actor, dtype=np.uint8),
        "probs": np.concatenate(probs, axis=0),
        "actor_reach": np.stack(actor_reach),
        "opp_reach": np.asarray(opp_reach, dtype=np.float32),
    }


def load(path: Path) -> dict:
    with np.load(path) as d:
        return {
            "histories": json.loads(zlib.decompress(d["histories"].tobytes())),
            "actions": json.loads(zlib.decompress(d["actions"].tobytes())),
            "actors": d["actors"], "probs": d["probs"].astype(np.float32) / 255.0,
            "actor_reach": d["actor_reach"].astype(np.float32), "opp_reach": d["opp_reach"],
            "meta": json.loads(zlib.decompress(d["meta"].tobytes())),
        }


def push(paths: list[Path], message: str) -> None:
    """Commit just these files and push; failures are reported, never fatal."""
    rel = [str(p.relative_to(ROOT)) for p in paths]
    for _ in range(3):
        try:
            subprocess.run(["git", "-C", str(ROOT), "add", "--", *rel], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(ROOT), "commit", "-q", "-m", message, "--", *rel],
                           check=True, capture_output=True)
            branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"], check=True,
                                    capture_output=True, text=True).stdout.strip()
            subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", branch], check=True,
                           capture_output=True, timeout=300)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"push failed ({e}); retrying", file=sys.stderr, flush=True)
            time.sleep(10)


def cmd_generate(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    vectors = [random_stacks(rng) for _ in range(args.start + args.count)]
    pending: list[Path] = []
    for i in range(args.start, args.start + args.count):
        path = OUT / f"solve_{args.seed}_{i:04d}.npz"
        if path.exists():
            continue
        stacks = vectors[i]
        _, game = game_for(stacks)
        t0 = time.perf_counter()
        out = gtopen.run(game, args.iterations, check_every=0, threads=args.threads, progress=False)
        m = out["metrics"]
        final = m["checkpoints"][-1]
        data = compact(out, game.n, args.min_reach)
        meta = {"stacks_bb": stacks, "seed": args.seed, "index": i, "iterations": m["iterations"],
                "nash_conv_bb100": round(100 * final["nash_conv_bb"], 3), "evs_bb": final["evs_bb"],
                "solve_s": round(m["solve_s"], 1), "wall_s": round(time.perf_counter() - t0, 1),
                "threads": m["threads"], "action_nodes": m["action_nodes"], "kept_nodes": len(data["actors"]),
                "min_reach": args.min_reach, "fork_commit": m["fork_commit"],
                **gtopen.GTOpenAbstraction().versions}
        data["meta"] = np.frombuffer(zlib.compress(json.dumps(meta).encode()), dtype=np.uint8)
        buf = io.BytesIO()
        np.savez_compressed(buf, **data)
        path.write_bytes(buf.getvalue())
        print(json.dumps({k: meta[k] for k in ("index", "stacks_bb", "nash_conv_bb100", "wall_s", "kept_nodes")}
                         | {"kb": path.stat().st_size // 1024}), flush=True)
        pending.append(path)
        if args.push_every and len(pending) >= args.push_every:
            push(pending, f"Preflop training data: {len(pending)} solve(s) through #{i}")
            pending = []
    if pending and args.push_every:
        push(pending, f"Preflop training data: last {len(pending)} solve(s)")


def cmd_summary(args) -> None:
    files = sorted(OUT.glob("solve_*.npz"))
    rows = [load(p)["meta"] for p in files]
    if not rows:
        print("no data")
        return
    nc = np.array([r["nash_conv_bb100"] for r in rows])
    print(f"{len(rows)} solves, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB; NashConv median "
          f"{np.median(nc):.2f}, max {nc.max():.2f} bb/100; median wall {np.median([r['wall_s'] for r in rows]):.0f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--count", type=int, default=200)
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--iterations", type=int, default=100)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--seed", type=int, default=101)
    g.add_argument("--min-reach", type=float, default=1e-3)
    g.add_argument("--push-every", type=int, default=0, help="commit and push data every N solves (0 = never)")
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("summary")
    s.set_defaults(func=cmd_summary)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
