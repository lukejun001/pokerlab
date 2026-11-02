"""Splits a run into fixed-size chunks and executes them on one or more workers.

Chunk boundaries depend only on ``hands`` and ``chunk_size``, and results are
merged in chunk order, so the worker count never changes the numbers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from pokerlab.analysis.statistics import StatsCollector
from pokerlab.simulation.configurations import SimulationConfig
from pokerlab.simulation.controller import ChunkResult, run_chunk


class StalledWorkerError(RuntimeError):
    pass


@dataclass
class BatchResult:
    config: SimulationConfig
    stats: StatsCollector
    elapsed: float
    workers: int
    chunks: int
    chunk_lineups: list[dict[int, str]]

    @property
    def hands(self) -> int:
        return self.stats.table.hands


def chunk_plan(hands: int, chunk_size: int) -> list[tuple[int, int, int]]:
    """``(chunk_index, first_hand, hand_count)`` for every chunk."""
    return [(i, start, min(chunk_size, hands - start)) for i, start in enumerate(range(0, hands, chunk_size))]


def _job(args: tuple[SimulationConfig, int, int, int, str | None]) -> ChunkResult:
    config, index, start, count, log_dir = args
    return run_chunk(config, index, start, count, log_dir)


def run_batch(config: SimulationConfig, progress: bool = True, log_dir: str | Path | None = None) -> BatchResult:
    plan = chunk_plan(config.hands, config.chunk_size)
    workers = config.workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(plan)))
    jobs = [(config, i, start, count, str(log_dir) if log_dir else None) for i, start, count in plan]

    stats = StatsCollector(big_blind=config.big_blind)
    lineups: list[dict[int, str]] = []
    started = time.perf_counter()
    done_hands = 0
    last_report = 0.0

    def absorb(result: ChunkResult) -> None:
        nonlocal done_hands, last_report
        stats.merge(result.stats)
        lineups.append(result.seat_agents)
        done_hands += result.hands
        now = time.perf_counter()
        if progress and (now - last_report > 2 or done_hands >= config.hands):
            last_report = now
            rate = done_hands / max(now - started, 1e-9)
            eta = (config.hands - done_hands) / rate if rate else 0
            print(
                f"\r  {done_hands:,}/{config.hands:,} hands  {rate:,.0f} hands/s  ETA {eta:,.0f}s   ",
                end="", file=sys.stderr, flush=True,
            )

    if workers == 1:
        for job in jobs:
            absorb(_job(job))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.imap(_job, jobs, chunksize=1)
            for _ in jobs:
                try:
                    absorb(results.next(timeout=config.chunk_timeout_s))
                except mp.TimeoutError:
                    pool.terminate()
                    raise StalledWorkerError(
                        f"a chunk produced no result within {config.chunk_timeout_s}s; worker appears stalled"
                    ) from None
    if progress:
        print(file=sys.stderr)

    return BatchResult(
        config=config,
        stats=stats,
        elapsed=time.perf_counter() - started,
        workers=workers,
        chunks=len(plan),
        chunk_lineups=lineups,
    )
