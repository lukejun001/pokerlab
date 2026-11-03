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
FORK_VERSION = "asym-3"


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
        dump_terminals: bool = False, per_seat: bool = True, progress: bool = True,
        warm_start: WarmStart | None = None) -> dict[str, Any]:
    """Run ``preflop-cli`` on ``game``; returns its JSON output plus process metrics.

    ``warm_start`` starts the solve from a given profile (e.g. one borrowed
    from the nearest solved stack vector) instead of from uniform play.
    """
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
    if warm_start is not None:
        request.update(warm_start.request())
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(request))
        started = time.perf_counter()
        proc = subprocess.Popen([str(BINARY), str(src), str(dst)], cwd=GTOPEN_DIR,
                                env=env, stderr=subprocess.PIPE, text=True)
        log = []
        for line in proc.stderr:
            log.append(line)
            if progress and ("iter" in line or "built" in line or "rejected" in line):
                print(line.rstrip(), file=sys.stderr)
        # wait4 reports the child's own CPU time and peak RSS on macOS and Linux alike.
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = time.perf_counter() - started
        text = "".join(log)
        if proc.returncode != 0:
            raise RuntimeError(f"preflop-cli failed ({proc.returncode}):\n{text[-2000:]}")
        out = json.loads(dst.read_text())
    cpu = usage.ru_utime + usage.ru_stime
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss_mb = usage.ru_maxrss / 2**20 if sys.platform == "darwin" else usage.ru_maxrss / 2**10
    out["metrics"].update(
        wall_s=round(wall, 2),
        cpu_s=round(cpu, 2),
        cpu_utilization=round(cpu / wall / (os.cpu_count() or 1), 3) if wall else None,
        cores=os.cpu_count(),
        peak_memory_mb=round(rss_mb, 1),
        fork_commit=fork_commit(),
        gpu="not used (CPU build)",
        request=request,
    )
    return out


_RANKS = "23456789TJQKA"


def gtopen_class_labels() -> list[str]:
    """GTOpen's class order: index ``r*13 + c`` is a pair (r == c), suited (r > c) or offsuit (r < c)."""
    labels = []
    for idx in range(169):
        r, c = divmod(idx, 13)
        if r == c:
            labels.append(_RANKS[r] * 2)
        elif r > c:
            labels.append(_RANKS[r] + _RANKS[c] + "s")
        else:
            labels.append(_RANKS[c] + _RANKS[r] + "o")
    return labels


TO_GTOPEN = [CLASS_LABELS.index(label) for label in gtopen_class_labels()]


def to_gtopen_profile(strategy: dict[str, np.ndarray]) -> dict[str, list]:
    """Profile in pokerlab class order -> the CLI's JSON form (GTOpen class order)."""
    return {h: np.asarray(t, dtype=np.float64)[:, TO_GTOPEN].round(6).tolist() for h, t in strategy.items()}


@dataclass
class WarmStart:
    """A profile to start a solve from (pokerlab class order), and how hard to trust it.

    ``sum_scale``: how many iterations' worth of average strategy the prior
    counts as (each node's strategy sums become the prior times the actor's
    reach under it, times this). ``regret_scale``: the prior's regret mass
    relative to one iteration's (rescaled to the prior's opponent reach), so
    regret matching plays the prior until real regrets outweigh it.
    ``iterations``: DCFR's iteration counter after seeding, i.e. how slowly the
    discounting forgets the prior. See ``preflop-cli``.
    """

    profile: dict[str, np.ndarray]
    regret_scale: float = 1.0
    sum_scale: float = 1.0
    iterations: int = 0

    def request(self) -> dict[str, Any]:
        return {"warm_start": to_gtopen_profile(self.profile), "warm_regret_scale": self.regret_scale,
                "warm_sum_scale": self.sum_scale, "warm_iterations": self.iterations}


def evaluate(game: PreflopGame, strategy: dict[str, np.ndarray], threads: int | None = None) -> dict[str, Any]:
    """Exact best-response gaps of a fixed strategy profile in ``game``.

    ``strategy`` maps betting histories of ``game``'s tree to (actions x 169)
    probability tables in pokerlab class order. Nodes not listed play
    uniformly (reported as ``uniform_nodes``).
    """
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} not found")
    request = {
        "config": preflop_config(game),
        "iterations": 0,
        "multiway_equity_model": game.abstraction.multiway_equity_model,
        "chips_per_bb": game.big_blind,
        "evaluate": to_gtopen_profile(strategy),
    }
    env = dict(os.environ)
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps(request))
        proc = subprocess.run([str(BINARY), str(src), str(dst)], cwd=GTOPEN_DIR, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"preflop-cli evaluate failed:\n{proc.stderr[-2000:]}")
        return json.loads(dst.read_text())["evaluation"]


def tree(game: PreflopGame) -> dict[str, list[str]]:
    """GTOpen's abstract tree for ``game``: history -> action tokens (no solving)."""
    out = run(game, iterations=0, progress=False, threads=2)
    return {node["history"]: node["actions"] for node in out["nodes"]}


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
             progress: bool = True, warm_start: WarmStart | None = None,
             extra_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Solve a solver config with GTOpen and store the strategy; returns metrics."""
    s = config.solve
    iterations = iterations or s.iterations
    out = run(config.game, iterations, check_every=s.eval_every, threads=threads or s.workers or None,
              progress=progress, warm_start=warm_start)
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
        "warm_start": m.get("warm_start"),
        **(extra_metrics or {}),
        **config.game.abstraction.versions,
    }
    rows, probs = strategy_rows(out)
    key = store.save_strategy(config.game, rows, probs, m["nodes"], metrics)
    row = next(r for r in store.list() if r["key"] == key)
    metrics.update(key=key, strategy_bytes=row["size_bytes"])
    store.db.execute("UPDATE strategies SET metrics = ? WHERE key = ?", (json.dumps(metrics), key))
    store.db.commit()
    return metrics
