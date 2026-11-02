"""Terminal reports and CSV/JSON export. Every number comes from recorded results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pokerlab.simulation.batch import BatchResult

POSITION_ORDER = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def agent_table(result: BatchResult) -> pd.DataFrame:
    bb = result.config.big_blind
    rows = [s.summary(bb) for s in result.stats.agents.values()]
    return pd.DataFrame(rows).sort_values("net_bb100", ascending=False, ignore_index=True)


def bucket_table(result: BatchResult, attribute: str, key_name: str) -> pd.DataFrame:
    bb = result.config.big_blind
    rows = []
    for stats in result.stats.agents.values():
        for key, bucket in getattr(stats, attribute).items():
            n = bucket.hands
            mean = bucket.net / bb / n
            var = max(0.0, bucket.net_sq_bb / n - mean**2) * n / (n - 1) if n > 1 else 0.0
            rows.append({
                "agent": stats.agent_id,
                key_name: key,
                "hands": n,
                "net_bb100": 100 * mean,
                "bb100_std_error": 100 * (var / n) ** 0.5,
                "gross_bb100": 100 * (bucket.net + bucket.rake) / bb / n,
            })
    return pd.DataFrame(rows)


def table_summary(result: BatchResult) -> dict[str, Any]:
    t = result.stats.table
    cfg = result.config
    decisions = t.decisions
    return {
        "name": cfg.name,
        "hands_completed": t.hands,
        "players": cfg.players,
        "blinds": [cfg.small_blind, cfg.big_blind],
        "starting_stack": cfg.starting_stack,
        "mode": cfg.mode,
        "rake": cfg.rake,
        "seed": cfg.seed,
        "total_rake": t.rake,
        "rake_bb100_table": 100 * t.rake / cfg.big_blind / t.hands if t.hands else None,
        "raked_hand_share": t.raked_hands / t.hands if t.hands else None,
        "avg_pot": t.pot_total / t.hands if t.hands else None,
        "avg_pot_bb": t.pot_total / t.hands / cfg.big_blind if t.hands else None,
        "flop_share": t.flops / t.hands if t.hands else None,
        "showdown_share": t.showdowns / t.hands if t.hands else None,
        "decisions": decisions,
        "invalid_actions": t.invalid_actions,
        "stalled_hands": t.stalled_hands,
        "engine_errors": t.engine_errors,
        "error_samples": t.error_samples,
        "rebuy_chips": t.rebuy_chips,
        "session_resets": t.session_resets,
        "runtime_s": result.elapsed,
        "hands_per_second": t.hands / result.elapsed if result.elapsed else None,
        "decisions_per_second": decisions / result.elapsed if result.elapsed else None,
        "workers": result.workers,
        "chunks": result.chunks,
    }


def _fmt(value: float | None, spec: str = ".2f", suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{spec}}{suffix}"


def _pct(value: float | None) -> str:
    return _fmt(None if value is None else 100 * value, ".1f", "%")


def describe_rake(rake: dict[str, Any], bb: int) -> str:
    pct = rake.get("percentage", 0) or 0
    if not pct:
        return "none"
    parts = [f"{100 * pct:g}%"]
    if rake.get("cap_by_players"):
        parts.append("cap by players " + ", ".join(f"{k}+: {v}" for k, v in sorted(rake["cap_by_players"].items())))
    elif rake.get("cap") is not None:
        parts.append(f"capped at {rake['cap']} chips ({rake['cap'] / bb:g}bb)")
    else:
        parts.append("uncapped")
    if rake.get("no_flop_no_drop", True):
        parts.append("no flop no drop")
    return ", ".join(parts)


def render_report(result: BatchResult, hero: str | None = None) -> str:
    cfg = result.config
    summary = table_summary(result)
    agents = agent_table(result)
    hero = hero or next((s.agent_id for s in result.stats.agents.values() if s.archetype == "Experimental"), None)
    lines = [
        "",
        "POKER AI SIMULATION REPORT",
        "=" * 26,
        f"Experiment:        {cfg.name}",
        f"Hands completed:   {summary['hands_completed']:,}",
        f"Players:           {cfg.players}",
        f"Blinds:            {cfg.small_blind}/{cfg.big_blind} chips, stacks {cfg.starting_stack_bb:g}bb ({cfg.mode})",
        f"Rake:              {describe_rake(cfg.rake, cfg.big_blind)}",
        f"Seed:              {cfg.seed}",
    ]
    if hero and hero in result.stats.agents:
        h = result.stats.agents[hero].summary(cfg.big_blind)
        lines += [
            "",
            f"{hero} ({h['archetype']})",
            "-" * (len(hero) + len(str(h['archetype'])) + 3),
            f"Net profit:        {h['net_chips']:+,} chips ({h['net_bb']:+,.1f} bb)",
            f"Net bb/100:        {_fmt(h['net_bb100'], '+.2f')}  "
            f"(95% CI {_fmt(h['bb100_ci95_low'], '+.2f')} to {_fmt(h['bb100_ci95_high'], '+.2f')}, "
            f"SE {_fmt(h['bb100_std_error'])})",
            f"Gross bb/100:      {_fmt(h['gross_bb100'], '+.2f')}  (before attributed rake)",
            f"Rake attributed:   {h['rake_attributed_chips']:,.0f} chips ({_fmt(h['rake_attributed_bb100'])} bb/100)",
            f"Hands won:         {h['hands_won']:,} ({_pct(h['hands_won'] / h['hands'] if h['hands'] else None)})",
            f"VPIP / PFR:        {_pct(h['vpip'])} / {_pct(h['pfr'])}",
            f"3-bet:             {_pct(h['threebet'])} of {h['threebet_opportunities']:,} opportunities",
            f"Fold to 3-bet:     {_pct(h['fold_to_3bet'])} of {h['fold_to_3bet_opportunities']:,} opportunities",
            f"Aggression:        AFq {_pct(h['aggression_frequency'])}, AF {_fmt(h['aggression_factor'])}",
            f"WTSD / W$SD:       {_pct(h['wtsd'])} / {_pct(h['wsd'])}",
        ]
        ci_low, ci_high = h["bb100_ci95_low"], h["bb100_ci95_high"]
        if ci_low is not None and ci_low <= 0 <= ci_high:
            lines.append("Note: the 95% interval includes 0 bb/100; this win rate is not distinguishable from break-even.")

    lines += ["", "ALL AGENTS (sorted by net bb/100)", ""]
    header = f"{'agent':<16}{'hands':>9}{'net bb/100':>12}{'±95%':>8}{'gross':>9}{'rake':>7}{'VPIP':>7}{'PFR':>7}{'3B':>7}{'F3B':>7}{'AFq':>7}{'WTSD':>7}{'inval':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in agents.iterrows():
        se = r["bb100_std_error"]
        lines.append(
            f"{r['agent']:<16}{r['hands']:>9,}{_fmt(r['net_bb100'], '+.2f'):>12}{_fmt(None if se is None else 1.96 * se, '.1f'):>8}"
            f"{_fmt(r['gross_bb100'], '+.1f'):>9}{_fmt(r['rake_attributed_bb100'], '.1f'):>7}"
            f"{_pct(r['vpip']):>7}{_pct(r['pfr']):>7}{_pct(r['threebet']):>7}{_pct(r['fold_to_3bet']):>7}"
            f"{_pct(r['aggression_frequency']):>7}{_pct(r['wtsd']):>7}{r['invalid_actions']:>7}"
        )

    if hero and hero in result.stats.agents:
        positions = bucket_table(result, "by_position", "position")
        positions = positions[positions.agent == hero]
        if not positions.empty:
            positions = positions.assign(order=positions.position.map(POSITION_ORDER.index)).sort_values("order")
            lines += ["", f"{hero} BY POSITION", f"{'position':<10}{'hands':>9}{'net bb/100':>12}{'±95%':>9}"]
            for _, r in positions.iterrows():
                lines.append(f"{r['position']:<10}{r['hands']:>9,}{r['net_bb100']:>+12.1f}{1.96 * r['bb100_std_error']:>9.1f}")
        if cfg.seat_assignment != "fixed":
            seats = bucket_table(result, "by_seat", "seat")
            seats = seats[seats.agent == hero].sort_values("seat")
            lines += ["", f"{hero} BY SEAT", f"{'seat':<10}{'hands':>9}{'net bb/100':>12}{'±95%':>9}"]
            for _, r in seats.iterrows():
                lines.append(f"{int(r['seat']):<10}{r['hands']:>9,}{r['net_bb100']:>+12.1f}{1.96 * r['bb100_std_error']:>9.1f}")

    total_net = sum(s.net for s in result.stats.agents.values())
    lines += [
        "",
        "TABLE",
        f"Rake removed:      {summary['total_rake']:,} chips ({_fmt(summary['rake_bb100_table'])} bb/100 hands, "
        f"{_pct(summary['raked_hand_share'])} of hands raked)",
        f"Players' net sum:  {total_net:+,} chips (equals -rake: {total_net == -summary['total_rake']})"
        + (f"; rebuys added {summary['rebuy_chips']:,} chips" if cfg.mode == "continuous" else ""),
        f"Average pot:       {_fmt(summary['avg_pot_bb'], '.1f')} bb; flops seen {_pct(summary['flop_share'])}, "
        f"showdowns {_pct(summary['showdown_share'])}",
        f"Problems:          {summary['invalid_actions']} invalid actions, {summary['stalled_hands']} stalled hands, "
        f"{summary['engine_errors']} engine errors",
        f"Simulation runtime: {summary['runtime_s']:.1f}s on {summary['workers']} worker(s)",
        f"Hands per second:  {_fmt(summary['hands_per_second'], ',.0f')}",
        f"Decisions per sec: {_fmt(summary['decisions_per_second'], ',.0f')}",
    ]
    for sample in summary["error_samples"][:5]:
        lines.append(f"  error: {sample}")
    return "\n".join(lines)


def export(result: BatchResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agents = agent_table(result)
    agents.to_csv(out / "agents.csv", index=False)
    bucket_table(result, "by_position", "position").to_csv(out / "positions.csv", index=False)
    bucket_table(result, "by_seat", "seat").to_csv(out / "seats.csv", index=False)
    summary = {
        "table": table_summary(result),
        "agents": agents.to_dict(orient="records"),
        "chunk_lineups": result.chunk_lineups[:50],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "config.yaml").write_text(yaml.safe_dump(result.config.to_dict(), sort_keys=False))
    return out


def default_output_dir(result_or_config_name: str, base: str = "results") -> Path:
    return Path(base) / f"{result_or_config_name}_{time.strftime('%Y%m%d_%H%M%S')}"
