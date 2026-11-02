"""Runs a contiguous block ("chunk") of hands at one table.

Each chunk is self-contained and seeded only from ``(seed, chunk index)``
and the global hand number, so a run gives identical results whether its
chunks execute on one worker or many.

* Cards for global hand ``h`` come from ``Random(f"{seed}-deck-{h}")``, so two
  configs with the same seed see the same cards hand for hand (matched
  comparisons, e.g. with and without rake).
* The button sits on seat ``h % players`` (advancing to the next occupied seat
  in continuous mode if that seat is sitting out).
* Agents are rebuilt at the start of every chunk with chunk-specific seeds.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from pokerlab.agents import create_agent
from pokerlab.analysis.statistics import StatsCollector, action_flags
from pokerlab.engine.cards import cards_str
from pokerlab.engine.game import ChipConservationError, HandResult, StalledHandError, play_hand, seat_order
from pokerlab.engine.observations import PlayerPublicStats
from pokerlab.engine.rake import make_rake_function
from pokerlab.simulation.configurations import SimulationConfig


@dataclass
class ChunkResult:
    chunk_index: int
    hands: int
    stats: StatsCollector
    elapsed: float = 0.0
    seat_agents: dict[int, str] = field(default_factory=dict)


class PublicStatsTracker:
    """Opponent tendencies built from public actions only."""

    def __init__(self) -> None:
        self.counts: dict[int, dict[str, int]] = {}
        self.snapshot: MappingProxyType = MappingProxyType({})

    def update(self, result: HandResult) -> None:
        flags = action_flags(result.actions, result.seats)
        flop = set(result.flop_seats)
        showdown = set(result.showdown_seats)
        for seat in result.seats:
            c = self.counts.setdefault(seat, dict.fromkeys(PlayerPublicStats.__dataclass_fields__, 0))
            f = flags[seat]
            c["hands"] += 1
            c["vpip_hands"] += f.vpip
            c["pfr_hands"] += f.pfr
            c["postflop_aggressive"] += f.postflop_aggressive
            c["postflop_calls"] += f.postflop_calls
            c["postflop_folds"] += f.postflop_folds
            c["postflop_checks"] += f.postflop_checks
            c["saw_flop"] += seat in flop
            c["showdowns"] += seat in showdown
        self.snapshot = MappingProxyType({s: PlayerPublicStats(**c) for s, c in self.counts.items()})

    def reset_seat(self, seat: int) -> None:
        self.counts.pop(seat, None)


def seat_lineup(config: SimulationConfig, chunk_index: int) -> list[dict]:
    """Agent spec for each seat in this chunk."""
    specs = config.agent_specs()
    n = len(specs)
    if config.seat_assignment == "rotate":
        shift = chunk_index % n
        return [specs[(seat - shift) % n] for seat in range(n)]
    if config.seat_assignment == "shuffle":
        order = list(range(n))
        random.Random(f"{config.seed}-seats-{chunk_index}").shuffle(order)
        return [specs[i] for i in order]
    return specs


def _position_stack_fn(config: SimulationConfig):
    """button seat -> {seat: chips} for per-position starting stacks."""
    from pokerlab.solver.game import position_names, preflop_order_from_pokerkit

    n = config.players
    names = position_names(n)
    order = preflop_order_from_pokerkit(n)
    chips = {p: int(round(v * config.big_blind)) for p, v in config.stacks_by_position_bb.items()}

    def stacks(button: int) -> dict[int, int]:
        seats = seat_order(range(n), button)
        return {seat: chips[names[order[pk]]] for pk, seat in enumerate(seats)}

    return stacks


def _hand_log_entry(result: HandResult, names: dict[int, str]) -> dict:
    return {
        "hand": result.hand_id,
        "button": result.button_seat,
        "players": {str(s): names[s] for s in result.seats},
        "positions": {str(s): p for s, p in result.positions.items()},
        "hole_cards": {str(s): cards_str(c) for s, c in result.hole_cards.items()},
        "board": cards_str(result.board),
        "actions": [[a.seat, int(a.street), a.type.value, a.amount] for a in result.actions],
        "net": {str(s): v for s, v in result.net.items()},
        "rake": result.rake,
        "shown": {str(s): cards_str(c) for s, c in result.shown_cards.items()},
    }


def run_chunk(config: SimulationConfig, chunk_index: int, hand_start: int, hand_count: int,
              log_dir: str | None = None) -> ChunkResult:
    started = time.perf_counter()
    game = config.game_info()
    rake_function = make_rake_function(config.rake_config())
    n = config.players
    bb = config.big_blind
    starting_stack = config.starting_stack

    lineup = seat_lineup(config, chunk_index)
    agents = {}
    seat_agents: dict[int, tuple[str, str]] = {}
    for seat, spec in enumerate(lineup):
        agent = create_agent(spec["type"], spec["id"], f"{config.seed}-{chunk_index}-{spec['id']}", spec["params"])
        agents[seat] = agent
        seat_agents[seat] = (agent.agent_id, agent.archetype)

    stats = StatsCollector(big_blind=bb)
    for agent_id, archetype in seat_agents.values():
        stats.ensure(agent_id, archetype)
    tracker = PublicStatsTracker()
    history: deque = deque(maxlen=config.history_window)

    continuous = config.mode == "continuous"
    position_stacks = _position_stack_fn(config) if config.stacks_by_position_bb else None
    stacks = dict.fromkeys(range(n), starting_stack)
    table_chips = starting_stack * n  # chips on the table, for the session conservation check
    rebuy_below = int(config.rebuy_below_bb * bb)
    rebuy_to = int(config.rebuy_to_bb * bb)

    log_file = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(log_dir) / f"hands_chunk_{chunk_index:05d}.jsonl", "w")
    names = {seat: aid for seat, (aid, _) in seat_agents.items()}

    try:
        for hand_id in range(hand_start, hand_start + hand_count):
            if continuous:
                if config.rebuy_enabled:
                    for seat in range(n):
                        if stacks[seat] < rebuy_below:
                            added = rebuy_to - stacks[seat]
                            stacks[seat] += added
                            table_chips += added
                            stats.record_rebuy(*seat_agents[seat], added)
                active = [s for s in range(n) if stacks[s] > 0]
                if len(active) < 2:  # session over: everyone restarts
                    stacks = dict.fromkeys(range(n), starting_stack)
                    table_chips = starting_stack * n
                    stats.table.session_resets += 1
                    active = list(range(n))
            else:
                stacks = dict.fromkeys(range(n), starting_stack)
                active = list(range(n))

            button = hand_id % n
            while button not in active:
                button = (button + 1) % n
            if position_stacks:
                stacks = position_stacks(button)

            try:
                result = play_hand(
                    agents=agents,
                    seats=active,
                    button_seat=button,
                    stacks=stacks,
                    game=game,
                    rake_function=rake_function,
                    deck_rng=random.Random(f"{config.seed}-deck-{hand_id}"),
                    hand_id=hand_id,
                    previous_hands=tuple(history),
                    player_stats=tracker.snapshot,
                    max_actions=config.max_actions_per_hand,
                    strict=config.strict,
                )
            except StalledHandError as exc:
                if config.strict:
                    raise
                stats.table.stalled_hands += 1
                if len(stats.table.error_samples) < 10:
                    stats.table.error_samples.append(str(exc))
                continue
            except ChipConservationError:
                raise  # accounting bug: never continue silently
            except Exception as exc:  # unexpected engine failure: record, skip the hand
                if config.strict:
                    raise
                stats.table.engine_errors += 1
                if len(stats.table.error_samples) < 10:
                    stats.table.error_samples.append(f"hand {hand_id}: {type(exc).__name__}: {exc}")
                continue

            stats.record(result, seat_agents)
            tracker.update(result)
            record = result.public_record()
            history.append(record)
            for agent in agents.values():
                agent.observe_hand(record)

            if continuous:
                stacks.update(result.final_stacks)
                table_chips -= result.rake
                if sum(stacks.values()) != table_chips:
                    raise ChipConservationError(
                        f"session chips {sum(stacks.values())} != expected {table_chips} after hand {hand_id}"
                    )
            if log_file:
                log_file.write(json.dumps(_hand_log_entry(result, names)) + "\n")
    finally:
        if log_file:
            log_file.close()

    for seat, agent in agents.items():
        sources = getattr(getattr(agent, "strategy", None), "decision_sources", None)
        if sources:
            counts = stats.agents[seat_agents[seat][0]].decision_sources
            for label, count in sources.items():
                counts[label] = counts.get(label, 0) + count

    return ChunkResult(
        chunk_index=chunk_index,
        hands=stats.table.hands,
        stats=stats,
        elapsed=time.perf_counter() - started,
        seat_agents=names,
    )
