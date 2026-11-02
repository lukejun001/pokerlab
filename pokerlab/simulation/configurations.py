"""Simulation configuration: YAML/JSON files, presets and CLI overrides."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pokerlab.engine.observations import GameInfo
from pokerlab.engine.rake import RakeConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

MIXED_LINEUP = [
    {"type": "Experimental", "id": "Hero"},
    {"type": "Nit"},
    {"type": "TAG"},
    {"type": "LAG"},
    {"type": "CallingStation"},
    {"type": "Maniac"},
    {"type": "LoosePassive"},
    {"type": "FitOrFold"},
    {"type": "Balanced"},
]


@dataclass
class SimulationConfig:
    name: str = "default"
    hands: int = 10_000
    seed: int = 42
    players: int | None = None  # None = size of the lineup
    small_blind: int = 25
    big_blind: int = 50
    starting_stack_bb: float = 100
    # Independent mode: give each position its own starting stack every hand,
    # e.g. {UTG: 100, HJ: 150, CO: 70, BTN: 115, SB: 40, BB: 200} (solver labels).
    stacks_by_position_bb: dict[str, float] | None = None
    mode: str = "independent"  # independent | continuous
    rebuy_enabled: bool = True
    rebuy_below_bb: float = 40  # continuous mode: top up when a stack drops below this
    rebuy_to_bb: float = 100
    rake: dict[str, Any] = field(default_factory=lambda: {
        "percentage": 0.04, "cap": 150, "no_flop_no_drop": True,
    })
    seat_assignment: str = "fixed"  # fixed | rotate | shuffle (applied per chunk)
    chunk_size: int = 1000
    workers: int = 1  # 0 = one per CPU core
    max_actions_per_hand: int = 500
    chunk_timeout_s: float = 600
    strict: bool = False  # raise on agent errors instead of substituting check/fold
    history_window: int = 10  # previous public hand records passed to agents
    hand_log: bool = False  # write every hand to JSONL
    output_dir: str = "results"
    lineup: list[dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(MIXED_LINEUP))

    def __post_init__(self) -> None:
        if self.mode not in ("independent", "continuous"):
            raise ValueError(f"mode must be independent or continuous, not {self.mode!r}")
        if self.seat_assignment not in ("fixed", "rotate", "shuffle"):
            raise ValueError(f"seat_assignment must be fixed, rotate or shuffle, not {self.seat_assignment!r}")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind <= big_blind")
        if self.hands < 1 or self.chunk_size < 1:
            raise ValueError("hands and chunk_size must be positive")
        self.lineup = resize_lineup(self.lineup, self.players or len(self.lineup))
        self.players = len(self.lineup)
        RakeConfig.from_dict(self.rake)  # validate early
        if self.stacks_by_position_bb:
            from pokerlab.solver.game import position_names

            if self.mode != "independent":
                raise ValueError("stacks_by_position_bb requires independent mode")
            if set(self.stacks_by_position_bb) != set(position_names(self.players)):
                raise ValueError(f"stacks_by_position_bb must name {position_names(self.players)}")

    @property
    def starting_stack(self) -> int:
        return int(round(self.starting_stack_bb * self.big_blind))

    def rake_config(self) -> RakeConfig:
        return RakeConfig.from_dict(self.rake)

    def game_info(self) -> GameInfo:
        rake = self.rake_config()
        return GameInfo(
            num_seats=self.players,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            starting_stack=self.starting_stack,
            rake_percentage=rake.percentage,
            rake_cap=rake.cap,
            no_flop_no_drop=rake.no_flop_no_drop,
            mode=self.mode,
            rake_cap_by_players=tuple(sorted(rake.cap_by_players.items())),
        )

    def agent_specs(self) -> list[dict[str, Any]]:
        """Lineup entries with unique ids (``TAG_1``, ``TAG_2``...)."""
        counts: dict[str, int] = {}
        for entry in self.lineup:
            counts[entry["type"]] = counts.get(entry["type"], 0) + 1
        seen: dict[str, int] = {}
        specs = []
        for entry in self.lineup:
            kind = entry["type"]
            seen[kind] = seen.get(kind, 0) + 1
            default_id = kind if counts[kind] == 1 else f"{kind}_{seen[kind]}"
            specs.append({"type": kind, "id": entry.get("id", default_id), "params": entry.get("params", {})})
        ids = [s["id"] for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"agent ids must be unique: {ids}")
        return specs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resize_lineup(lineup: list[dict[str, Any]], players: int) -> list[dict[str, Any]]:
    if not 2 <= players <= 9:
        raise ValueError("players must be between 2 and 9")
    if players <= len(lineup):
        return [dict(e) for e in lineup[:players]]
    # Extend by cycling through the opponents (everyone but the first entry).
    pool = lineup[1:] or lineup
    extra = [dict(pool[i % len(pool)]) for i in range(players - len(lineup))]
    for e in extra:
        e.pop("id", None)
    return [dict(e) for e in lineup] + extra


def resolve_config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    for candidate in (CONFIG_DIR / name_or_path, CONFIG_DIR / f"{name_or_path}.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no config file or preset named {name_or_path!r} (presets live in {CONFIG_DIR})")


def load_config(name_or_path: str | None = None, overrides: dict[str, Any] | None = None) -> SimulationConfig:
    data: dict[str, Any] = {}
    if name_or_path:
        path = resolve_config_path(name_or_path)
        text = path.read_text()
        data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})
        data.setdefault("name", path.stem)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key == "rake" and isinstance(value, dict):
            data["rake"] = {**data.get("rake", SimulationConfig().rake), **value}
        else:
            data[key] = value
    unknown = set(data) - set(SimulationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return SimulationConfig(**data)


def list_presets() -> list[Path]:
    """Simulation presets (solver configs, which have a ``game:`` section, are listed separately)."""
    return [p for p in sorted(CONFIG_DIR.glob("*.yaml")) if "game" not in (yaml.safe_load(p.read_text()) or {})]


def list_solver_configs() -> list[Path]:
    return [p for p in sorted(CONFIG_DIR.glob("*.yaml")) if "game" in (yaml.safe_load(p.read_text()) or {})]
