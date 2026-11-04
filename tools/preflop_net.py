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
from preflop_data import class_weights, game_from_meta, load, random_spec  # noqa: E402
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
        game = game_from_meta(d["meta"])
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
    game = game_from_meta(d["meta"])
    tree = dict(zip(d["histories"], d["actions"]))
    # Borrowing needs the library's family (6-max, base rake): other games get a uniform anchor.
    same_family = len(d["meta"]["stacks_bb"]) == 6 and "rake_pct" not in d["meta"]
    profile = anchor_tables(game, tree, exclude=d["meta"].get("key")) if same_family else {}
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

    def loss_of(x, m, t, w, a=None, sample: int = 0):
        """Reach-weighted cross-entropy. With ``sample`` > 0, each row's hands are drawn in
        proportion to their weights (an unbiased estimate of the same loss at sample/169 the cost)."""
        if sample:
            # Fixed batch shapes (rows with no weight just contribute nothing): filtering rows gave
            # every batch a new shape, and the allocator's fragmentation grew the process to 13 GB.
            wsum = w.sum(-1)
            hands = torch.multinomial(w + 1e-12, sample, replacement=True)  # (B, K)
            rows = torch.arange(len(x))[:, None]
            logp = torch.log_softmax(model.forward_hands(x, m, hands, None if a is None else a[rows, hands].float()),
                                     dim=-1)
            target = t[rows, hands].float() / 255.0
            ce = -(target * logp.clamp_min(-30)).sum(-1).mean(-1)  # (B,)
            return (ce * wsum).sum() / wsum.sum()
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
            loss = loss_of(X[idx], M[idx], T[idx], W[idx], None if A is None else A[idx], sample=args.hand_samples)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        msg = f"epoch {epoch + 1}/{epochs} train {total / len(X):.4f}"
        if val is not None and (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                vx, vm, vt, vw, va = val
                num = den = 0.0
                for s in range(0, len(vx), args.batch):  # batched: one pass over everything needs ~13 GB
                    sl = slice(s, s + args.batch)
                    w_sum = float(vw[sl].sum())
                    if w_sum > 0:
                        num += loss_of(vx[sl], vm[sl], vt[sl], vw[sl], None if va is None else va[sl]).item() * w_sum
                        den += w_sum
                msg += f" val {num / max(den, 1e-12):.4f}"
        log(f"{msg} ({time.perf_counter() - t0:.0f}s)")
    model.eval()
    return model, {"train_solves": [p.name for p in train_files], "val_solves": [p.name for p in val_files],
                   "nodes": len(X), "epochs": epochs, "train_s": round(time.perf_counter() - t0, 1)}


def build(args):
    return net.build_model(hidden=args.hidden, dim=args.dim, layers=args.layers, residual=args.residual,
                             arch=args.arch)


def knn_reference() -> dict[tuple, float]:
    path = ROOT / "reports" / "phase3" / "library" / "measurements.json"
    rows = json.loads(path.read_text())["rows"]
    return {tuple(r["stacks_bb"]): r["nash_conv_bb100"] for r in rows if r["set"] == "holdout" and r["k"] == 1}


def evaluate(model, tests: int, threads: int, seed: int = 11, log=print, vary: bool = False) -> list[dict]:
    """Held-out games: 6-max stack vectors (seed 11), or with ``vary`` games that also vary
    players dealt and rake (seed 12; no library baseline there)."""
    rng = random.Random(12 if vary else seed)
    specs = [random_spec(rng, True) if vary else {"stacks_bb": random_stacks(rng)} for _ in range(tests)]
    ref = {} if vary else knn_reference()
    rows = []
    for spec in specs:
        stacks = spec["stacks_bb"]
        game = game_from_meta(spec)
        t0 = time.perf_counter()
        tree = python_tree(game)
        anchor = anchor_tables(game, tree) if getattr(model, "residual", False) else None
        profile = net.profile_for(model, game, tree, anchors=anchor)
        infer_s = time.perf_counter() - t0
        ev = gtopen.evaluate(game, profile, threads=threads)
        row = {**spec, "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3),
               "br_gap_bb100": [round(100 * g, 2) for g in ev["br_gaps_bb"]], "uniform_nodes": ev["uniform_nodes"],
               "knn_bb100": ref.get(tuple(stacks)), "inference_s": round(infer_s, 2), "nodes": len(tree)}
        rows.append(row)
        log(json.dumps({k: row[k] for k in ("stacks_bb", "nash_conv_bb100", "knn_bb100", "inference_s")}))
    return rows


def solve_files(limit: int | None) -> list[Path]:
    files = (sorted(DATA.glob("solve_lib_*.npz")) + sorted(DATA.glob("solve_101_*.npz"))
             + sorted(DATA.glob("solve_102_*.npz")))
    return files[:limit] if limit else files


def cmd_train(args) -> None:
    files = solve_files(args.solves)
    model, info = train(files, args.epochs, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_model(model, out, {**info, "opp_power": args.opp_power})
    print(f"saved {out} ({len(files)} solves)")


def evaluate_train(model, names: list[str], threads: int, log=print) -> list[dict]:
    """Exact NashConv of the network in games it was trained on (fit, not generalization)."""
    rows = []
    for name in names:
        d = load(DATA / name)
        game = game_from_meta(d["meta"])
        tree = python_tree(game)
        anchor = anchor_tables(game, tree, exclude=d["meta"].get("key")) if getattr(model, "residual", False) else None
        ev = gtopen.evaluate(game, net.profile_for(model, game, tree, anchors=anchor), threads=threads)
        rows.append({"solve": name, "stacks_bb": d["meta"]["stacks_bb"], "nash_conv_bb100": round(100 * ev["nash_conv_bb"], 3),
                     "solve_bb100": d["meta"]["nash_conv_bb100"]})
        log(json.dumps(rows[-1]))
    return rows


def cmd_evaluate(args) -> None:
    model, extra = net.load_model(args.model)
    train_rows = evaluate_train(model, extra.get("train_solves", [])[: args.on_train], args.threads) if args.on_train else []
    rows = evaluate(model, args.tests, args.threads, vary=args.vary)
    nc = [r["nash_conv_bb100"] for r in rows]
    knn = [r["knn_bb100"] for r in rows if r["knn_bb100"] is not None]
    print(f"network median {np.median(nc):.2f} bb/100 (min {min(nc):.2f}, max {max(nc):.2f})"
          + (f"; library k=1 median {np.median(knn):.2f}" if knn else ""))
    if train_rows:
        print(f"on training games: median {np.median([r['nash_conv_bb100'] for r in train_rows]):.2f} bb/100")
    write_json(REPORTS / f"eval_{Path(args.model).stem}{'_vary' if args.vary else ''}.json",
               {"model": args.model, "extra": extra, "rows": rows, "train_rows": train_rows})


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
        q.add_argument("--arch", default="bilinear", choices=["bilinear", "mlp"])
        q.add_argument("--hand-samples", type=int, default=0,
                       help="hands drawn per node by loss weight in training (0 = all 169)")
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
    e.add_argument("--on-train", type=int, default=0, help="also score this many training games")
    e.add_argument("--vary", action="store_true", help="held-out games that also vary players dealt and rake")
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
