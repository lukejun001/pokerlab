"""Local strategy cache: metadata in SQLite, strategies in compressed files.

A strategy's **key** hashes everything that changes the modeled game or the
way it was solved: variant, players dealt in, positions and button, every
starting stack (chips), blinds and ante, rake (percentage, cap schedule,
no-flop-no-drop), the betting abstraction, and the solver, continuation-model
and strategy-model versions. A lookup with any difference is a miss: a
configuration with other stacks or rake never silently reuses a strategy.

Inside a strategy, each decision node is addressed by its betting history
(``"f-r12500-c"``: tokens with chip amounts). Each node stores mixed action
probabilities for all 169 hand classes (float32, renormalized on load).

``nearest`` lists solved games that differ only in stacks, with their stack
distance, as the foundation for interpolation/generalization experiments.
Such matches are always reported as approximate, never as exact.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pokerlab.solver.cfr import CONTINUATION_VERSION, SOLVER_VERSION
from pokerlab.solver.config import STRATEGY_MODEL_VERSION
from pokerlab.solver.game import PreflopGame

DEFAULT_STORE = Path(__file__).resolve().parents[2] / "strategies"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    key TEXT PRIMARY KEY,
    variant TEXT NOT NULL,
    players INTEGER NOT NULL,
    positions TEXT NOT NULL,
    button_index INTEGER NOT NULL,
    stacks TEXT NOT NULL,
    stacks_bb TEXT NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    ante INTEGER NOT NULL,
    rake TEXT NOT NULL,
    abstraction TEXT NOT NULL,
    abstraction_name TEXT NOT NULL,
    tree_version TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    continuation_version TEXT NOT NULL,
    strategy_model_version TEXT NOT NULL,
    family TEXT NOT NULL,
    file TEXT NOT NULL,
    nodes INTEGER NOT NULL,
    decision_nodes INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    created REAL NOT NULL,
    metrics TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS strategies_family ON strategies (family);
"""


def button_index(players: int) -> int:
    """Preflop-order index of the button (heads-up: the small blind)."""
    return 0 if players == 2 else players - 3


def model_description(game: PreflopGame, solver_version: str | None = None,
                      continuation_version: str | None = None,
                      strategy_model_version: str = STRATEGY_MODEL_VERSION) -> dict[str, Any]:
    """Every key field. The abstraction may carry its own engine versions
    (``versions``: e.g. GTOpen's), which then replace this solver's defaults."""
    versions = dict(getattr(game.abstraction, "versions", {}) or {})
    desc = game.describe()
    desc.update(button_index=button_index(game.n),
                solver_version=solver_version or versions.get("solver_version", SOLVER_VERSION),
                continuation_version=continuation_version or versions.get("continuation_version", CONTINUATION_VERSION),
                strategy_model_version=strategy_model_version)
    if "tree_version" in versions:
        desc["tree_version"] = versions["tree_version"]
    return desc


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def strategy_key(game: PreflopGame, **versions: str) -> str:
    return _hash(model_description(game, **versions))


def family_key(game: PreflopGame, **versions: str) -> str:
    """Everything in the key except the stacks: games one could interpolate between."""
    desc = model_description(game, **versions)
    desc.pop("stacks")
    return _hash(desc)


@dataclass
class StoredStrategy:
    key: str
    meta: dict[str, Any]
    nodes: dict[str, tuple[int, tuple[str, ...], int]]  # history -> (offset, actions, actor)
    probs: np.ndarray  # (edges, 169)

    def action_probabilities(self, history: str, hand_class: int) -> tuple[tuple[str, ...], np.ndarray] | None:
        node = self.nodes.get(history)
        if node is None:
            return None
        offset, actions, _ = node
        return actions, self.probs[offset:offset + len(actions), hand_class]

    def node_table(self, history: str) -> tuple[tuple[str, ...], np.ndarray] | None:
        node = self.nodes.get(history)
        if node is None:
            return None
        offset, actions, _ = node
        return actions, self.probs[offset:offset + len(actions)]


@dataclass
class LookupResult:
    kind: str  # "exact" (approximate candidates come from ``nearest``)
    strategy: StoredStrategy


class StrategyStore:
    def __init__(self, root: str | Path = DEFAULT_STORE, cache_size: int = 16) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "files").mkdir(exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self.db = sqlite3.connect(self.db_path)
        self.db.executescript(_SCHEMA)
        self._cache: OrderedDict[str, StoredStrategy] = OrderedDict()
        self.cache_size = cache_size

    # --- writing ----------------------------------------------------------------------

    def save(self, solver, metrics: dict[str, Any] | None = None) -> str:
        """Store a ``PreflopSolver``'s average strategy; returns its key."""
        nodes = [[nd.key, solver.offset[nd.id], list(nd.actions), nd.actor] for nd in solver.tree.decision_nodes]
        return self.save_strategy(solver.game, nodes, solver.average_strategy(), solver.tree.stats()["nodes"], metrics)

    def save_strategy(self, game: PreflopGame, nodes: list, probs: np.ndarray, total_nodes: int,
                      metrics: dict[str, Any] | None = None) -> str:
        """Store any solver's output: ``nodes`` are ``[history, offset, actions,
        actor]`` rows into ``probs`` (edges x 169, pokerlab class order)."""
        key = strategy_key(game)
        probs = np.asarray(probs, dtype=np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, probs=probs,
                            nodes=np.frombuffer(zlib.compress(json.dumps(nodes).encode()), dtype=np.uint8))
        rel = f"files/{key[:24]}.npz"
        (self.root / rel).write_bytes(buf.getvalue())
        desc = model_description(game)
        self.db.execute(
            "INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, desc["variant"], game.n, json.dumps(desc["positions"]), desc["button_index"],
             json.dumps(desc["stacks"]), json.dumps(list(game.stacks_bb())), game.small_blind, game.big_blind,
             game.ante, json.dumps(desc["rake"], sort_keys=True), json.dumps(desc["abstraction"], sort_keys=True),
             game.abstraction.name, desc["tree_version"], desc["solver_version"], desc["continuation_version"],
             desc["strategy_model_version"], family_key(game), rel, total_nodes, len(nodes),
             len(buf.getvalue()), time.time(), json.dumps(metrics or {})),
        )
        self.db.commit()
        self._cache.pop(key, None)
        return key

    # --- reading ----------------------------------------------------------------------

    def _row(self, key: str) -> dict[str, Any] | None:
        cur = self.db.execute("SELECT * FROM strategies WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([c[0] for c in cur.description], row))

    def load(self, key: str) -> StoredStrategy | None:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        meta = self._row(key)
        if meta is None:
            return None
        with np.load(self.root / meta["file"]) as data:
            probs = data["probs"].astype(np.float64)
            nodes_raw = json.loads(zlib.decompress(data["nodes"].tobytes()))
        nodes = {}
        for history, offset, actions, actor in nodes_raw:
            block = probs[offset:offset + len(actions)]
            total = block.sum(axis=0)
            block /= np.where(total > 0, total, 1.0)
            nodes[history] = (offset, tuple(actions), actor)
        meta["metrics"] = json.loads(meta["metrics"])
        strategy = StoredStrategy(key, meta, nodes, probs)
        self._cache[key] = strategy
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return strategy

    def lookup(self, game: PreflopGame) -> LookupResult | None:
        """Exact match only: every key field identical."""
        strategy = self.load(strategy_key(game))
        return None if strategy is None else LookupResult("exact", strategy)

    def nearest(self, game: PreflopGame, limit: int = 5) -> list[tuple[float, str]]:
        """Solved games identical except for stacks, as (max stack difference in bb, key)."""
        rows = self.db.execute("SELECT key, stacks_bb FROM strategies WHERE family = ?", (family_key(game),))
        target = np.array(game.stacks_bb())
        found = [(float(np.abs(np.array(json.loads(s)) - target).max()), k) for k, s in rows]
        return sorted(found)[:limit]

    def list(self) -> list[dict[str, Any]]:
        cur = self.db.execute("SELECT * FROM strategies ORDER BY created")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def database_bytes(self) -> int:
        files = sum((self.root / r["file"]).stat().st_size for r in self.list() if (self.root / r["file"]).exists())
        return files + self.db_path.stat().st_size

    def close(self) -> None:
        self.db.close()
