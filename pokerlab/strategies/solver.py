"""Experimental-agent strategy backed by precomputed preflop solver output.

Decision flow for each preflop decision:

1. rebuild the modeled game from the observation: players dealt in, each
   position's starting stack (current stack + chips already in), blinds, rake;
2. combine it with the configured betting abstraction and look up an
   **exact** strategy in the local store (any difference is a miss);
3. translate the public betting history into tree tokens; a history the
   abstraction does not contain (e.g. an opponent's off-tree raise size) is a
   miss, not a guess;
4. read the mixed action probabilities for our hand class and sample one
   action with the strategy's seeded RNG.

Every decision is tagged and counted in ``decision_sources``:
``exact:<key prefix>`` for solver output, ``approximate:<key prefix>`` only
when ``allow_approximate`` is on (off by default: no approximation has been
validated yet), and ``fallback:<reason>`` when the heuristic fallback decides.
Fallback decisions are never labeled as solver/GTO output.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pokerlab.engine.actions import Action, ActionType
from pokerlab.engine.observations import Observation, PublicHandRecord, Street, position_labels
from pokerlab.engine.rake import RakeConfig
from pokerlab.solver.config import load_solver_config
from pokerlab.solver.game import BettingAbstraction, PreflopGame, action_amount, action_kind, preflop_order_from_pokerkit
from pokerlab.solver.hands import class_of
from pokerlab.solver.store import DEFAULT_STORE, StrategyStore
from pokerlab.strategies.base import Strategy, load_strategy


class SolverStrategy(Strategy):
    name = "solver"

    def __init__(self, seed: int | str = 0, config: str | None = None, store: str | None = None,
                 fallback: str = "equity", allow_approximate: bool = False,
                 abstraction: dict[str, Any] | None = None, **params: Any) -> None:
        super().__init__(seed, **params)
        if config is not None:
            self.abstraction = load_solver_config(config).game.abstraction
        elif abstraction and abstraction.get("engine") == "gtopen":
            from pokerlab.solver.gtopen import GTOpenAbstraction

            self.abstraction = GTOpenAbstraction.from_dict({k: v for k, v in abstraction.items() if k != "engine"})
        else:
            self.abstraction = BettingAbstraction.from_dict(abstraction)
        self.store = StrategyStore(store or DEFAULT_STORE)
        self.fallback = load_strategy(fallback, seed)
        self.allow_approximate = allow_approximate
        self.decision_sources: Counter[str] = Counter()
        self._games: dict[tuple, tuple[PreflopGame, object]] = {}

    def reset(self, seed: int | str) -> None:
        super().reset(seed)
        self.fallback.reset(seed)

    def observe_hand(self, record: PublicHandRecord) -> None:
        self.fallback.observe_hand(record)

    # --- game reconstruction -------------------------------------------------------

    @staticmethod
    def preflop_seats(obs: Observation) -> list[int]:
        """Seats dealt in, in preflop action order."""
        n = obs.num_players
        labels = position_labels(n)
        order = preflop_order_from_pokerkit(n)
        seats = [0] * n
        for seat, label in obs.seat_positions.items():
            seats[order[labels.index(label)]] = seat
        return seats

    def modeled_game(self, obs: Observation, seats: list[int]) -> PreflopGame:
        g = obs.game
        stacks = tuple(obs.stacks[s] + obs.bets[s] for s in seats)  # preflop: bets == chips put in
        rake = RakeConfig(percentage=g.rake_percentage, cap=g.rake_cap,
                          cap_by_players=dict(g.rake_cap_by_players), no_flop_no_drop=g.no_flop_no_drop)
        return PreflopGame(stacks, g.small_blind, g.big_blind, rake, self.abstraction)

    @staticmethod
    def history_tokens(obs: Observation, seats: list[int]) -> list[str]:
        tokens = []
        for a in obs.history:
            if a.type is ActionType.FOLD:
                tokens.append("f")
            elif a.type is ActionType.CHECK:
                tokens.append("x")
            elif a.type is ActionType.CALL:
                tokens.append("c")
            else:
                tokens.append(("a" if a.all_in else "r") + str(a.amount))
        return tokens

    # --- decisions ----------------------------------------------------------------------

    def _fall_back(self, obs: Observation, reason: str) -> Action:
        self.decision_sources[f"fallback:{reason}"] += 1
        return self.fallback.decide(obs)

    def decide(self, obs: Observation) -> Action:
        if obs.street != Street.PREFLOP:
            return self._fall_back(obs, "postflop")
        seats = self.preflop_seats(obs)
        cache_key = (tuple(obs.stacks[s] + obs.bets[s] for s in seats), obs.game)
        cached = self._games.get(cache_key)
        if cached is None:
            try:
                game = self.modeled_game(obs, seats)
            except (ValueError, NotImplementedError):
                game = None
            match = self.store.lookup(game) if game else None
            label = "exact"
            if match is None and game is not None and self.allow_approximate:
                near = self.store.nearest(game, limit=1)
                if near:
                    strategy = self.store.load(near[0][1])
                    match, label = (strategy, "approximate") if strategy else (None, label)
            else:
                match = match.strategy if match else None
            cached = self._games[cache_key] = (match, label)
        strategy, label = cached
        if strategy is None:
            return self._fall_back(obs, "no_matching_strategy")
        history = "-".join(self.history_tokens(obs, seats))
        found = strategy.action_probabilities(history, class_of(obs.hole_cards))
        if found is None:
            return self._fall_back(obs, "history_not_in_abstraction")
        actions, probs = found
        x = self.rng.random()
        choice = actions[-1]
        cumulative = 0.0
        for token, p in zip(actions, probs):
            cumulative += p
            if x < cumulative:
                choice = token
                break
        self.decision_sources[f"{label}:{strategy.key[:12]}"] += 1
        kind = action_kind(choice)
        if kind == "f":
            return Action.fold()
        if kind == "x":
            return Action.check()
        if kind == "c":
            return Action.call()
        return Action.raise_to(action_amount(choice))
