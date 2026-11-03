"""GTOpen as the preflop solver (asymmetric-stack fork, run as a subprocess).

GTOpen (https://github.com/MatthewPDingle/GTOpen) is a Rust CFR solver. The
fork in ``../gtopen-asymmetric`` (private) adds per-seat stacks, PokerKit's
all-in rules and side-pot terminals to its Preflop Lab engine, plus a
``preflop-cli`` binary. This module turns a pokerlab solver config into a
GTOpen ``PreflopConfig``, runs the binary, and stores the exported average
strategy in the local strategy cache under a GTOpen-specific key.

Set ``GTOPEN_DIR`` to point elsewhere; build with
``cargo build --release -p solver --bin preflop-cli`` in that directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokerlab.solver.game import PreflopGame
from pokerlab.solver.hands import CLASS_LABELS

GTOPEN_DIR = Path(os.environ.get("GTOPEN_DIR", Path(__file__).resolve().parents[3] / "gtopen-asymmetric"))
BINARY = GTOPEN_DIR / "target" / "release" / "preflop-cli"
UPSTREAM_COMMIT = "6c70502"
# Bump when the fork's game rules or terminal values change: it is part of every strategy key.
FORK_VERSION = "asym-2"


@dataclass(frozen=True)
class GTOpenAbstraction:
    """GTOpen Preflop Lab sizing and continuation settings (sizes in bb)."""

    name: str = "gtopen-6max-v1"
    open_raises: tuple[float, ...] = (2.5,)
    sb_open_raises: tuple[float, ...] = (3.0,)
    raise_mults: tuple[float, ...] = (3.0,)
    fourbet_mults: tuple[float, ...] | None = None
    max_raises: int = 3
    add_allin: bool = True
    allin_threshold: float = 0.5
    limp: bool = False
    realization: str = "balanced"
    multiway_equity_model: str = "coupled_deck_v1"

    def __post_init__(self) -> None:
        for name in ("open_raises", "sb_open_raises", "raise_mults", "fourbet_mults"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(float(x) for x in value))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GTOpenAbstraction:
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown GTOpen abstraction keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        out = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        out["engine"] = "gtopen"
        return out

    @property
    def versions(self) -> dict[str, str]:
        return {
            "solver_version": f"gtopen-{UPSTREAM_COMMIT}+{FORK_VERSION}",
            "continuation_version": f"gtopen-{self.realization}+{self.multiway_equity_model}+raw-side-pots",
            "tree_version": f"gtopen-preflop+{FORK_VERSION}",
        }


def preflop_config(game: PreflopGame, per_seat: bool = True) -> dict[str, Any]:
    """GTOpen ``PreflopConfig`` JSON for ``game`` (bb units).

    ``per_seat=False`` (equal stacks only) uses GTOpen's original single-stack
    path, for reproducing the unmodified engine.
    """
    ab: GTOpenAbstraction = game.abstraction
    bb = game.big_blind
    n = game.n
    stacks = [s / bb for s in game.stacks]
    posts = [0.0] * n
    posts[n - 2], posts[n - 1] = game.small_blind / bb, 1.0
    cap = game.rake.cap_for(n)
    cfg: dict[str, Any] = {
        "positions": list(game.positions),
        "stack": max(stacks),
        "posts": posts,
        "ante": game.ante / bb,
        "limp": ab.limp,
        "open_raises": list(ab.open_raises),
        "raise_mults": list(ab.raise_mults),
        "max_raises": ab.max_raises,
        "add_allin": ab.add_allin,
        "allin_threshold": ab.allin_threshold,
        "rake_pct": 100 * game.rake.percentage,
        "rake_cap": 0.0 if cap is None else cap / bb,  # GTOpen: 0 = uncapped
        "no_flop_no_drop": game.rake.no_flop_no_drop,
        "realization": ab.realization,
    }
    if ab.sb_open_raises:
        cfg["open_raises_by_seat"] = [list(ab.sb_open_raises) if i == n - 2 else [] for i in range(n)]
    if ab.fourbet_mults is not None:
        cfg["fourbet_mults"] = list(ab.fourbet_mults)
    if per_seat:
        cfg["stacks"] = stacks
    elif len(set(stacks)) != 1:
        raise ValueError("the single-stack path needs equal stacks")
    return cfg


def fork_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(GTOPEN_DIR), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(game: PreflopGame, iterations: int, check_every: int = 0, threads: int | None = None,
        dump_terminals: bool = False, per_seat: bool = True, progress: bool = True) -> dict[str, Any]:
    """Run ``preflop-cli`` on ``game``; returns its JSON output plus process metrics."""
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} not found: build it with `cargo build --release -p solver "
                                f"--bin preflop-cli` in {GTOPEN_DIR}")
    request = {
        "config": preflop_config(game, per_seat=per_seat),
        "iterations": iterations,
        "check_every": check_every,
        "multiway_equity_model": game.abstraction.multiway_equity_model,
        "chips_per_bb": game.big_blind,
        "dump_terminals": dump_terminals,
    }
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(request))
        started = time.perf_counter()
        proc = subprocess.Popen(["/usr/bin/time", "-l", str(BINARY), str(src), str(dst)], cwd=GTOPEN_DIR,
                                env=env, stderr=subprocess.PIPE, text=True)
        log = []
        for line in proc.stderr:
            log.append(line)
            if progress and ("iter" in line or "built" in line or "rejected" in line):
                print(line.rstrip(), file=sys.stderr)
        proc.wait()
        wall = time.perf_counter() - started
        text = "".join(log)
        if proc.returncode != 0:
            raise RuntimeError(f"preflop-cli failed ({proc.returncode}):\n{text[-2000:]}")
        out = json.loads(dst.read_text())
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    user = re.search(r"([\d.]+) user", text)
    system = re.search(r"([\d.]+) sys", text)
    cpu = (float(user.group(1)) if user else 0.0) + (float(system.group(1)) if system else 0.0)
    out["metrics"].update(
        wall_s=round(wall, 2),
        cpu_s=round(cpu, 2),
        cpu_utilization=round(cpu / wall / (os.cpu_count() or 1), 3) if wall else None,
        cores=os.cpu_count(),
        peak_memory_mb=round(int(rss.group(1)) / 2**20, 1) if rss else None,
        fork_commit=fork_commit(),
        gpu="not used (CPU build)",
        request=request,
    )
    return out


def strategy_rows(out: dict[str, Any]) -> tuple[list, np.ndarray]:
    """GTOpen export -> store rows ``[history, offset, actions, actor]`` and probabilities (pokerlab class order)."""
    order = [out["class_labels"].index(label) for label in CLASS_LABELS]
    rows, blocks, offset = [], [], 0
    for node in out["nodes"]:
        table = np.asarray(node["strategy"], dtype=np.float64)[:, order]
        rows.append([node["history"], offset, node["actions"], node["actor"]])
        blocks.append(table)
        offset += len(node["actions"])
    return rows, np.concatenate(blocks, axis=0)


def generate(config, store, iterations: int | None = None, threads: int | None = None,
             progress: bool = True) -> dict[str, Any]:
    """Solve a solver config with GTOpen and store the strategy; returns metrics."""
    s = config.solve
    iterations = iterations or s.iterations
    out = run(config.game, iterations, check_every=s.eval_every, threads=threads or s.workers or None,
              progress=progress)
    m = out["metrics"]
    final = m["checkpoints"][-1] if m["checkpoints"] else {}
    metrics = {
        "config": config.name,
        "engine": "gtopen",
        "players": config.game.n,
        "positions": list(config.game.positions),
        "stacks_bb": list(config.game.stacks_bb()),
        "abstraction": config.game.abstraction.to_dict(),
        "tree": {"nodes": m["nodes"], "decision_nodes": m["action_nodes"], "infosets": 169 * m["action_nodes"]},
        "arena_mb": m["arena_mb"],
        "iterations": m["iterations"],
        "threads": m["threads"],
        "solve_s": round(m["solve_s"], 2),
        "eval_s": round(m["eval_s"], 2),
        "wall_s": m["wall_s"],
        "cpu_s": m["cpu_s"],
        "cpu_utilization": m["cpu_utilization"],
        "cores": m["cores"],
        "peak_memory_mb": m["peak_memory_mb"],
        "gpu": m["gpu"],
        "fork_commit": m["fork_commit"],
        "realization_note": m["realization_note"],
        "convergence": [{
            "iteration": c["iteration"], "solve_s": round(c["solve_s"], 2),
            "br_gap_bb100": [round(100 * g, 3) for g in c["br_gaps_bb"]],
            "nash_conv_bb100": round(100 * c["nash_conv_bb"], 3),
            "ev_bb": [round(e, 5) for e in c["evs_bb"]],
        } for c in m["checkpoints"]],
        "final_nash_conv_bb100": round(100 * final.get("nash_conv_bb", float("nan")), 3),
        **config.game.abstraction.versions,
    }
    rows, probs = strategy_rows(out)
    key = store.save_strategy(config.game, rows, probs, m["nodes"], metrics)
    row = next(r for r in store.list() if r["key"] == key)
    metrics.update(key=key, strategy_bytes=row["size_bytes"])
    store.db.execute("UPDATE strategies SET metrics = ? WHERE key = ?", (json.dumps(metrics), key))
    store.db.commit()
    return metrics
