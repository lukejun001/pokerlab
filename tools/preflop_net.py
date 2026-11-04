"""Train and score the preflop strategy network (pokerlab/solver/network.py).

    python tools/preflop_net.py train --solves 50 --epochs 30 --out models/preflop_50.pt
    python tools/preflop_net.py evaluate --model models/preflop_50.pt --tests 10
    python tools/preflop_net.py curve --sizes 25 50 100 200 --epochs 30 --tests 10

``evaluate`` scores the network's full profile in each held-out game (seed 11,
the vectors the library pilot used) with GTOpen's exact NashConv and puts the
k=1 library approximation's score from reports/phase3/library/measurements.json
next to it. Results go to reports/phase4/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflop_data import OUT as DATA  # noqa: E402
from preflop_data import class_weights, load  # noqa: E402
from stack_library import game_for, random_stacks  # noqa: E402

from pokerlab.solver import gtopen  # noqa: E402
from pokerlab.solver import network as net  # noqa: E402
from pokerlab.solver.generalize import python_tree  # noqa: E402
from pokerlab.solver.pipeline import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "phase4"
CACHE = ROOT / "data" / "cache"


def encoded(path: Path, opp_power: float) -> dict[str, np.ndarray]:
    """Features, masks, slot targets and loss weights of one solve (cached on disk)."""
    cache = CACHE / (path.stem + ".npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache) as d:
            arrays = dict(d)
    else:
        d = load(path)
        _, game = game_for(d["meta"]["stacks_bb"])
        feats, masks = net.encode_nodes(game, d["histories"], d["actions"])
        targets = np.zeros((len(d["histories"]), net.NUM_CLASSES, len(net.SLOTS)), dtype=np.uint8)
        off = 0
        for k, acts in enumerate(d["actions"]):
            block = d["probs"][off:off + len(acts)]
            for j, t in enumerate(acts):
                targets[k, :, net.slot(t)] = np.round(block[j] * 255)
            off += len(acts)
        arrays = {"feats": feats, "masks": masks, "targets": targets,
                  "actor_reach": d["actor_reach"].astype(np.float16), "opp_reach": d["opp_reach"]}
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **arrays)
    w = arrays["actor_reach"].astype(np.float32) * class_weights()[None, :].astype(np.float32)
    w *= (arrays["opp_reach"][:, None] + 1e-3) ** opp_power
    arrays["weights"] = w
    return arrays


_LIBRARY = None


def library():
    """The 16-game library (k=1 per-decision borrowing supplies the residual model's anchors)."""
    global _LIBRARY
    if _LIBRARY is None:
        from pokerlab.solver.generalize import Library
        from pokerlab.solver.store import StrategyStore

        _, template = game_for([100] * 6)
        _LIBRARY = Library(StrategyStore(), template)
    return _LIBRARY


def anchor_tables(game, tree: dict[str, list[str]], exclude: str | None = None) -> dict[str, np.ndarray]:
    from pokerlab.solver.generalize import approximate_profile

    profile, _ = approximate_profile(library(), game, tree, k=1, exclude=exclude)
    return profile


def anchors(path: Path) -> np.ndarray:
    """(nodes, 169, slots) borrowed strategy at each kept node of one solve (leave-one-out for library solves)."""
    cache = CACHE / (path.stem + "_anchor.npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache) as d:
            return d["anchor"]
    d = load(path)
    _, game = game_for(d["meta"]["stacks_bb"])
    tree = dict(zip(d["histories"], d["actions"]))
    profile = anchor_tables(game, tree, exclude=d["meta"].get("key"))
    out = np.full((len(d["histories"]), net.NUM_CLASSES, len(net.SLOTS)), 0.25, dtype=np.float16)
    for k, h in enumerate(d["histories"]):
        if h in profile:
            out[k] = net.slot_table(tree[h], profile[h])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, anchor=out)
    return out


def dataset(files: list[Path], opp_power: float, residual: bool = False):
    import torch

    parts = [encoded(p, opp_power) for p in files]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("feats", "masks", "targets", "weights")}
    out = [torch.from_numpy(cat["feats"]), torch.from_numpy(cat["masks"]), torch.from_numpy(cat["targets"]),
           torch.from_numpy(cat["weights"])]
    out.append(torch.from_numpy(np.concatenate([anchors(p) for p in files])) if residual else None)
    return tuple(out)


def train(files: list[Path], epochs: int, args, log=print):
    import torch

    # More threads than free cores collapses throughput (20x measured) while the solver runs.
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, len(files) // 10) if len(files) >= 10 else 0
    val_files, train_files = files[:n_val], files[n_val:]
    X, M, T, W, A = dataset(train_files, args.opp_power, args.residual)
    W = W / W.mean()
    val = dataset(val_files, args.opp_power, args.residual) if val_files else None
    model = build(args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = epochs * ((len(X) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def loss_of(x, m, t, w, a=None):
        logp = torch.log_softmax(model(x, m, None if a is None else a.float()), dim=-1)
        target = t.float() / 255.0
        ce = -(target * logp.clamp_min(-30)).sum(-1)  # (B, 169)
        return (ce * w).sum() / w.sum()

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for s in range(0, len(X), args.batch):
            idx = perm[s:s + args.batch]
            loss = loss_of(X[idx], M[idx], T[idx], W[idx], None if A is None else A[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        msg = f"epoch {epoch + 1}/{epochs} train {total / len(X):.4f}"
        if val is not None and (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                msg += f" val {loss_of(*val).item():.4f}"
        log(f"{msg} ({time.perf_counter() - t0:.0f}s)")
    model.eval()
    return model, {"train_solves": [p.name for p in train_files], "val_solves": [p.name for p in val_files],
                   "nodes": len(X), "epochs": epochs, "train_s": round(time.perf_counter() - t0, 1)}


def build(args):
    return net.build_model(hidden=args.hidden, dim=args.dim, layers=args.layers, residual=args.residual)


def knn_reference() -> dict[tuple, float]:
    path = ROOT / "reports" / "phase3" / "library" / "measurements.json"
    rows = json.loads(path.read_text())["rows"]
    return {tuple(r["stacks_bb"]): r["nash_conv_bb100"] for r in rows if r["set"] == "holdout" and r["k"] == 1}


def evaluate(model, tests: int, threads: int, seed: int = 11, log=print) -> list[dict]:
    rng = random.Random(seed)
    vectors = [random_stacks(rng) for _ in range(tests)]
    ref = knn_reference()
    rows = []
    for stacks in vectors:
        _, game = game_for(stacks)
        t0 = time.perf_counter()
        tree = python_tree(game)
        anchor = anchor_tables(game, tree) if getattr(model, "residual", False) else None
        profile = net.profile_for(model, game, tree, anchors=anchor)
        infer_s = time.perf_counter() - t0
        ev = gtopen.evaluate(game, profile, threads=threads)
        row = {"stacks_bb": stacks, "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3),
               "br_gap_bb100": [round(100 * g, 2) for g in ev["br_gaps_bb"]], "uniform_nodes": ev["uniform_nodes"],
               "knn_bb100": ref.get(tuple(stacks)), "inference_s": round(infer_s, 2), "nodes": len(tree)}
        rows.append(row)
        log(json.dumps({k: row[k] for k in ("stacks_bb", "nash_conv_bb100", "knn_bb100", "inference_s")}))
    return rows


def solve_files(limit: int | None) -> list[Path]:
    files = sorted(DATA.glob("solve_lib_*.npz")) + sorted(DATA.glob("solve_101_*.npz"))
    return files[:limit] if limit else files


def cmd_train(args) -> None:
    files = solve_files(args.solves)
    model, info = train(files, args.epochs, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_model(model, out, {**info, "opp_power": args.opp_power})
    print(f"saved {out} ({len(files)} solves)")


def cmd_evaluate(args) -> None:
    model, extra = net.load_model(args.model)
    rows = evaluate(model, args.tests, args.threads)
    nc = [r["nash_conv_bb100"] for r in rows]
    knn = [r["knn_bb100"] for r in rows if r["knn_bb100"] is not None]
    print(f"network median {np.median(nc):.2f} bb/100 (min {min(nc):.2f}, max {max(nc):.2f}); "
          f"library k=1 median {np.median(knn):.2f}")
    write_json(REPORTS / f"eval_{Path(args.model).stem}.json", {"model": args.model, "extra": extra, "rows": rows})


def cmd_curve(args) -> None:
    files = solve_files(None)
    out_path = REPORTS / f"curve_{args.label}.json"
    results = json.loads(out_path.read_text())["results"] if out_path.exists() else []
    done = {r["solves"] for r in results}
    for size in args.sizes:
        if size > len(files) or size in done:
            continue
        print(f"== {size} solves", flush=True)
        model, info = train(files[:size], args.epochs, args, log=lambda m: print("   " + m, flush=True))
        rows = evaluate(model, args.tests, args.threads, log=lambda m: print("   " + m, flush=True))
        nc = [r["nash_conv_bb100"] for r in rows]
        results.append({"solves": size, "median_bb100": float(np.median(nc)), "mean_bb100": float(np.mean(nc)),
                        "max_bb100": float(max(nc)), "info": info, "rows": rows})
        print(f"   median {np.median(nc):.2f} bb/100", flush=True)
        write_json(out_path, {"args": vars(args) | {"func": None}, "results": results})
        net.save_model(model, ROOT / "models" / f"preflop_{args.label}_{size}.pt", info)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--epochs", type=int, default=30)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--lr", type=float, default=2e-3)
        q.add_argument("--hidden", type=int, default=256)
        q.add_argument("--dim", type=int, default=64)
        q.add_argument("--layers", type=int, default=3)
        q.add_argument("--opp-power", type=float, default=0.5, help="loss weight exponent on opponents' reach")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--tests", type=int, default=10)
        q.add_argument("--torch-threads", type=int, default=2)
        q.add_argument("--residual", action="store_true",
                       help="learn a correction to the nearest library game's strategy (anchor input)")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--solves", type=int)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)
    e = sub.add_parser("evaluate")
    e.add_argument("--model", required=True)
    e.add_argument("--tests", type=int, default=10)
    e.add_argument("--threads", type=int, default=4)
    e.set_defaults(func=cmd_evaluate)
    c = sub.add_parser("curve")
    common(c)
    c.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200])
    c.add_argument("--label", default="main")
    c.set_defaults(func=cmd_curve)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
