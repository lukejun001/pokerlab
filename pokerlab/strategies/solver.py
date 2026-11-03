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
``exact:<key prefix>`` for solver output, ``approximate:<key prefix>`` for a
strategy borrowed from solved games with similar effective stacks (see
``solver/generalize.py``), and ``fallback:<reason>`` when the heuristic
fallback decides. Fallback decisions are never labeled as solver/GTO output.

``approximate`` controls borrowing for unsolved stack vectors:
``"validated"`` (default) borrows only when the game's matching statistic is
within the radius calibrated by measured exact exploitability
(``strategies/approximation.json``); ``"off"`` never borrows; ``"always"``
borrows from the nearest games regardless (for experiments only).

``on_demand`` controls what happens to a game with no stored strategy
(GTOpen abstractions only; see ``solver/ondemand.py``): ``"off"`` (default)
nothing; ``"queue"`` records it in the store's pending queue for
``pokerlab strategies solve-pending`` and plays this hand as above;
``"solve"`` solves it right away, warm-started from the library, and plays
the exact result (minutes per new stack vector: interactive use, not batch
simulations). ``on_demand_iterations`` and ``on_demand_threads`` size that solve.
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
                 fallback: str = "equity", approximate: str = "validated",
                 abstraction: dict[str, Any] | None = None, on_demand: str = "off",
                 on_demand_iterations: int | None = None, on_demand_threads: int | None = None,
                 **params: Any) -> None:
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
        if approximate not in ("validated", "off", "always"):
            raise ValueError("approximate must be validated, off or always")
        self.approximate = approximate
        if on_demand not in ("off", "queue", "solve"):
            raise ValueError("on_demand must be off, queue or solve")
        self.on_demand = on_demand
        self.on_demand_iterations = on_demand_iterations
        self.on_demand_threads = on_demand_threads
        self.decision_sources: Counter[str] = Counter()
        self._games: dict[tuple, tuple] = {}

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

    def _resolve(self, obs: Observation, seats: list[int]) -> tuple:
        """(kind, source, label) for this hand's game: an exact stored strategy,
        a validated approximate profile, or (None, None, fallback reason)."""
        try:
            game = self.modeled_game(obs, seats)
        except (ValueError, NotImplementedError):
            return None, None, "no_matching_strategy"
        match = self.store.lookup(game)
        if match is None and self.on_demand != "off" and hasattr(self.abstraction, "versions"):
            from pokerlab.solver import ondemand

            if self.on_demand == "queue":
                ondemand.PendingSolves(self.store).add(game)
            else:
                ondemand.solve_on_demand(self.store, game, threads=self.on_demand_threads,
                                         iterations=self.on_demand_iterations or ondemand.DEFAULT_ITERATIONS)
                match = self.store.lookup(game)
        if match is not None:
            return "exact", match.strategy, f"exact:{match.strategy.key[:12]}"
        if self.approximate == "off" or not hasattr(self.abstraction, "versions"):
            return None, None, "no_matching_strategy"
        from pokerlab.solver.generalize import Calibration, Library, approximate_profile, python_tree

        calibration = Calibration.load(self.store.root / "approximation.json")
        library = Library(self.store, game)
        if not len(library):
            return None, None, "no_matching_strategy"
        if self.approximate == "validated" and (calibration is None or calibration.family != library.family
                                               or calibration.radius <= 0):
            return None, None, "no_validated_approximation"
        tree = python_tree(game)
        k = calibration.k if calibration else 1
        profile, info = approximate_profile(library, game, tree, k=k)
        if self.approximate == "validated" and info[calibration.stat] > calibration.radius:
            return None, None, "approximation_outside_validated_radius"
        nearest = library.neighbors(game.stacks_bb(), k=1)[0]
        table = {h: (tuple(tree[h]), t) for h, t in profile.items()}
        return "approximate", table, f"approximate:{nearest.key[:12]}@{info[calibration.stat if calibration else 'node_distance_p90']:.2f}"

    def decide(self, obs: Observation) -> Action:
        if obs.street != Street.PREFLOP:
            return self._fall_back(obs, "postflop")
        seats = self.preflop_seats(obs)
        cache_key = (tuple(obs.stacks[s] + obs.bets[s] for s in seats), obs.game)
        cached = self._games.get(cache_key)
        if cached is None:
            cached = self._games[cache_key] = self._resolve(obs, seats)
            if len(self._games) > 8:  # approximate profiles are large; keep the most recent games
                self._games.pop(next(iter(self._games)))
        kind, source, label = cached
        if kind is None:
            return self._fall_back(obs, label)
        history = "-".join(self.history_tokens(obs, seats))
        hand = class_of(obs.hole_cards)
        if kind == "exact":
            found = source.action_probabilities(history, hand)
        else:
            node = source.get(history)
            found = None if node is None else (node[0], node[1][:, hand])
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
        self.decision_sources[label] += 1
        kind = action_kind(choice)
        if kind == "f":
            return Action.fold()
        if kind == "x":
            return Action.check()
        if kind == "c":
            return Action.call()
        return Action.raise_to(action_amount(choice))
