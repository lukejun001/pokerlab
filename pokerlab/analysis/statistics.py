"""Per-agent and table statistics, accumulated hand by hand.

Definitions (opportunity-based stats use the matching denominator):

* VPIP / PFR: hands where the player called or raised / raised preflop,
  over hands dealt. Posting a blind or checking the big blind is not VPIP.
* 3-bet: raised when facing exactly one preflop raise, over hands where the
  player acted facing exactly one raise.
* Fold to 3-bet: folded after making the first preflop raise and facing a
  re-raise (exactly two raises), over hands where that happened.
* Aggression frequency: postflop bets+raises / (bets+raises+calls+folds).
  Aggression factor: postflop bets+raises / calls.
* WTSD: reached showdown / saw the flop. W$SD: won chips at showdown /
  reached showdown. Showdown frequency: reached showdown / hands dealt.
* Net bb/100 is measured from actual chip results, which already have rake
  removed. Gross bb/100 adds back the rake attributed to that player
  (each pot's rake split pro rata among the players who won that pot; when
  a raked pot splits with an odd chip this can differ from the exact
  counterfactual by a fraction of a chip).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from pokerlab.engine.actions import ActionType
from pokerlab.engine.game import HandResult
from pokerlab.engine.observations import ActionRecord, Street

_AGGRESSIVE = (ActionType.BET, ActionType.RAISE)


@dataclass
class SeatHandFlags:
    vpip: bool = False
    pfr: bool = False
    threebet_opportunity: bool = False
    threebet: bool = False
    fold_to_3bet_opportunity: bool = False
    fold_to_3bet: bool = False
    postflop_aggressive: int = 0
    postflop_calls: int = 0
    postflop_folds: int = 0
    postflop_checks: int = 0


def action_flags(actions: tuple[ActionRecord, ...], seats) -> dict[int, SeatHandFlags]:
    flags = {s: SeatHandFlags() for s in seats}
    raises = 0
    opener = None
    for a in actions:
        f = flags[a.seat]
        if a.street == Street.PREFLOP:
            if raises == 1 and a.seat != opener and not f.threebet_opportunity:
                f.threebet_opportunity = True
                f.threebet = a.type is ActionType.RAISE
            if raises == 2 and a.seat == opener and not f.fold_to_3bet_opportunity:
                f.fold_to_3bet_opportunity = True
                f.fold_to_3bet = a.type is ActionType.FOLD
            if a.type is ActionType.RAISE:
                raises += 1
                if opener is None:
                    opener = a.seat
                f.pfr = True
                f.vpip = True
            elif a.type is ActionType.CALL:
                f.vpip = True
        else:
            if a.type in _AGGRESSIVE:
                f.postflop_aggressive += 1
            elif a.type is ActionType.CALL:
                f.postflop_calls += 1
            elif a.type is ActionType.FOLD:
                f.postflop_folds += 1
            else:
                f.postflop_checks += 1
    return flags


@dataclass
class BucketStats:
    """Win-rate tally for a subset of hands (a position, a seat...)."""

    hands: int = 0
    net: int = 0
    net_sq_bb: float = 0.0
    rake: float = 0.0

    def add(self, net: int, bb: int, rake: float) -> None:
        self.hands += 1
        self.net += net
        self.net_sq_bb += (net / bb) ** 2
        self.rake += rake

    def merge(self, other: BucketStats) -> None:
        self.hands += other.hands
        self.net += other.net
        self.net_sq_bb += other.net_sq_bb
        self.rake += other.rake


@dataclass
class AgentStats:
    agent_id: str
    archetype: str
    hands: int = 0
    hands_won: int = 0
    net: int = 0
    net_sq_bb: float = 0.0
    winnings: int = 0
    rake_attributed: float = 0.0
    vpip: int = 0
    pfr: int = 0
    threebet_opportunities: int = 0
    threebets: int = 0
    fold_to_3bet_opportunities: int = 0
    fold_to_3bets: int = 0
    postflop_aggressive: int = 0
    postflop_calls: int = 0
    postflop_folds: int = 0
    postflop_checks: int = 0
    saw_flop: int = 0
    showdowns: int = 0
    showdowns_won: int = 0
    decisions: int = 0
    decision_time: float = 0.0
    invalid_actions: int = 0
    rebuys: int = 0
    rebuy_chips: int = 0
    by_position: dict[str, BucketStats] = field(default_factory=dict)
    by_seat: dict[int, BucketStats] = field(default_factory=dict)

    def merge(self, other: AgentStats) -> None:
        for f in fields(self):
            if f.name in ("agent_id", "archetype"):
                continue
            mine = getattr(self, f.name)
            theirs = getattr(other, f.name)
            if isinstance(mine, dict):
                for key, bucket in theirs.items():
                    mine.setdefault(key, BucketStats()).merge(bucket)
            else:
                setattr(self, f.name, mine + theirs)

    # --- derived metrics -----------------------------------------------------

    @staticmethod
    def _ratio(num: float, den: float) -> float | None:
        return num / den if den else None

    def summary(self, big_blind: int) -> dict[str, float | int | str | None]:
        n = self.hands
        mean_bb = self.net / big_blind / n if n else 0.0
        variance = max(0.0, self.net_sq_bb / n - mean_bb**2) * n / (n - 1) if n > 1 else 0.0
        se_bb100 = 100 * math.sqrt(variance / n) if n else None
        net_bb100 = 100 * mean_bb if n else None
        aggressive_total = self.postflop_aggressive + self.postflop_calls + self.postflop_folds
        return {
            "agent": self.agent_id,
            "archetype": self.archetype,
            "hands": n,
            "hands_won": self.hands_won,
            "net_chips": self.net,
            "net_bb": self.net / big_blind,
            "net_bb100": net_bb100,
            "gross_bb100": 100 * (self.net + self.rake_attributed) / big_blind / n if n else None,
            "bb100_std_error": se_bb100,
            "bb100_ci95_low": None if se_bb100 is None else net_bb100 - 1.96 * se_bb100,
            "bb100_ci95_high": None if se_bb100 is None else net_bb100 + 1.96 * se_bb100,
            "rake_attributed_chips": self.rake_attributed,
            "rake_attributed_bb100": 100 * self.rake_attributed / big_blind / n if n else None,
            "vpip": self._ratio(self.vpip, n),
            "pfr": self._ratio(self.pfr, n),
            "threebet": self._ratio(self.threebets, self.threebet_opportunities),
            "threebet_opportunities": self.threebet_opportunities,
            "fold_to_3bet": self._ratio(self.fold_to_3bets, self.fold_to_3bet_opportunities),
            "fold_to_3bet_opportunities": self.fold_to_3bet_opportunities,
            "aggression_frequency": self._ratio(self.postflop_aggressive, aggressive_total),
            "aggression_factor": self._ratio(self.postflop_aggressive, self.postflop_calls),
            "saw_flop": self._ratio(self.saw_flop, n),
            "wtsd": self._ratio(self.showdowns, self.saw_flop),
            "wsd": self._ratio(self.showdowns_won, self.showdowns),
            "showdown_frequency": self._ratio(self.showdowns, n),
            "decisions": self.decisions,
            "avg_decision_ms": 1000 * self.decision_time / self.decisions if self.decisions else None,
            "invalid_actions": self.invalid_actions,
            "rebuys": self.rebuys,
            "rebuy_chips": self.rebuy_chips,
        }


@dataclass
class TableStats:
    hands: int = 0
    rake: int = 0
    raked_hands: int = 0
    pot_total: int = 0  # sum of final pot sizes (winnings + rake)
    flops: int = 0
    showdowns: int = 0
    decisions: int = 0
    invalid_actions: int = 0
    stalled_hands: int = 0
    engine_errors: int = 0
    error_samples: list[str] = field(default_factory=list)
    rebuy_chips: int = 0
    session_resets: int = 0

    def merge(self, other: TableStats) -> None:
        for f in fields(self):
            if f.name == "error_samples":
                self.error_samples.extend(other.error_samples[: max(0, 10 - len(self.error_samples))])
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclass
class StatsCollector:
    big_blind: int
    agents: dict[str, AgentStats] = field(default_factory=dict)
    table: TableStats = field(default_factory=TableStats)

    def ensure(self, agent_id: str, archetype: str) -> AgentStats:
        stats = self.agents.get(agent_id)
        if stats is None:
            stats = self.agents[agent_id] = AgentStats(agent_id, archetype)
        return stats

    def record(self, result: HandResult, seat_agents: dict[int, tuple[str, str]]) -> None:
        """``seat_agents`` maps seat -> (agent_id, archetype)."""
        bb = self.big_blind
        t = self.table
        t.hands += 1
        t.rake += result.rake
        t.raked_hands += result.rake > 0
        t.pot_total += sum(result.winnings.values()) + result.rake
        t.flops += bool(result.flop_seats)
        t.showdowns += bool(result.showdown_seats)
        t.decisions += result.decisions
        t.invalid_actions += sum(result.invalid_actions.values())
        if result.errors and len(t.error_samples) < 10:
            t.error_samples.extend(result.errors[: 10 - len(t.error_samples)])

        flags = action_flags(result.actions, result.seats)
        decisions_by_seat: dict[int, int] = {}
        for a in result.actions:
            decisions_by_seat[a.seat] = decisions_by_seat.get(a.seat, 0) + 1
        flop_seats = set(result.flop_seats)
        showdown_seats = set(result.showdown_seats)

        for seat in result.seats:
            agent_id, archetype = seat_agents[seat]
            s = self.ensure(agent_id, archetype)
            f = flags[seat]
            net = result.net[seat]
            rake = result.rake_attributed.get(seat, 0.0)
            won = result.winnings.get(seat, 0)
            s.hands += 1
            s.hands_won += won > 0
            s.net += net
            s.net_sq_bb += (net / bb) ** 2
            s.winnings += won
            s.rake_attributed += rake
            s.vpip += f.vpip
            s.pfr += f.pfr
            s.threebet_opportunities += f.threebet_opportunity
            s.threebets += f.threebet
            s.fold_to_3bet_opportunities += f.fold_to_3bet_opportunity
            s.fold_to_3bets += f.fold_to_3bet
            s.postflop_aggressive += f.postflop_aggressive
            s.postflop_calls += f.postflop_calls
            s.postflop_folds += f.postflop_folds
            s.postflop_checks += f.postflop_checks
            s.saw_flop += seat in flop_seats
            s.showdowns += seat in showdown_seats
            s.showdowns_won += seat in showdown_seats and won > 0
            s.decisions += decisions_by_seat.get(seat, 0)
            s.decision_time += result.decision_time.get(seat, 0.0)
            s.invalid_actions += result.invalid_actions.get(seat, 0)
            s.by_position.setdefault(result.positions[seat], BucketStats()).add(net, bb, rake)
            s.by_seat.setdefault(seat, BucketStats()).add(net, bb, rake)

    def record_rebuy(self, agent_id: str, archetype: str, chips: int) -> None:
        s = self.ensure(agent_id, archetype)
        s.rebuys += 1
        s.rebuy_chips += chips
        self.table.rebuy_chips += chips

    def merge(self, other: StatsCollector) -> None:
        for agent_id, stats in other.agents.items():
            self.ensure(agent_id, stats.archetype).merge(stats)
        self.table.merge(other.table)
