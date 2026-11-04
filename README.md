# pokerlab

An offline No-Limit Texas Hold'em research lab in Python. One experimental agent
plays against up to eight heuristic opponents for as many hands as you like, with
exact rake accounting, reproducible seeds and per-agent statistics.

[PokerKit](https://github.com/uoftcprg/pokerkit) (0.7.x) runs the game rules:
blinds, betting rounds, legal bet sizes, all-ins, main and side pots, showdowns,
split pots and stack accounting. pokerlab adds the agents, seeded dealing,
whole-hand rake caps, batch execution, statistics and reports.

## Install

Python 3.11 or newer.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m pokerlab simulate --hands 100000 --players 9 --seed 42 --workers 0
```

```bash
.venv/bin/python -m pokerlab simulate --config configs/default.yaml
```

```bash
.venv/bin/python -m pokerlab suite
```

```bash
.venv/bin/python -m pokerlab presets
```

```bash
.venv/bin/python -m pytest
```

`simulate` prints a report and writes `summary.json`, `agents.csv`,
`positions.csv`, `seats.csv` and the resolved `config.yaml` to
`results/<name>_<timestamp>/`. Useful options:

| option | meaning |
|---|---|
| `--preset NAME` | a config from `configs/` |
| `--hands`, `--players`, `--seed` | override the config |
| `--workers N` | processes; `0` = all cores. Results do not depend on N |
| `--mode independent\|continuous` | reset stacks every hand, or keep them with rebuys |
| `--rake 4 --rake-cap 150` | percentage and cap in chips (`--rake-cap -1` = uncapped) |
| `--small-blind`, `--big-blind`, `--stack-bb` | blind level and stack depth |
| `--seats fixed\|rotate\|shuffle` | how the lineup is seated in each chunk |
| `--strategy SPEC` | swap the experimental agent's strategy (see below) |
| `--hand-log` | write every hand (with hole cards) to JSONL |
| `--strict` | stop on any agent or engine error instead of recording it |

### Experiment presets

| | preset | what it tests |
|---|---|---|
| A | `zero_rake` | Hero vs one of each archetype, no rake |
| B | `rake_comparison` | Same seed and lineup as A with 4% rake, 3bb cap. Every card and decision matches A, so the difference is exactly the rake |
| C | `tight_table` | Hero vs four TAGs, two Nits, a tight Balanced and a FitOrFold |
| D | `aggressive_table` | Hero vs five LAGs and three Maniacs |
| E | `mixed_table` | One of each archetype, seats shuffled per chunk |
| F | `position_benchmark` | Lineup rotates one seat per chunk; report breaks results down by seat and position |
| | `continuous_session` | Persistent stacks, top-up to 100bb below 40bb |
| | `heads_up` | Hero vs Balanced, rake cap by players dealt in |
| | `default` | 9-handed $25/$50, 100bb, 4% capped at 3bb, no flop no drop |

`python -m pokerlab suite` runs A–F and prints one comparison table.

## Layout

```
pokerlab/
  engine/
    cards.py         card ints, fast 7-card evaluator, Monte Carlo equity
    actions.py       Action / ActionType and validation
    observations.py  the immutable view agents get; positions
    rake.py          RakeConfig and the whole-hand rake callback
    game.py          plays one hand through PokerKit
  agents/
    base.py          Agent interface
    heuristics.py    hand features, draws, equity vs ranges, pot odds, sizing, opponent tendencies
    preflop.py       169-class preflop equity table (data/preflop_equity.json)
    opponents.py     the eight archetypes
    experimental.py  ExperimentalAgent: delegates to a Strategy
  strategies/
    base.py          Strategy interface and loader
    heuristic.py     default EquityStrategy (equity + simple opponent modeling)
  simulation/
    configurations.py  config loading, presets, overrides
    controller.py      runs one chunk: seating, button, stacks, rebuys, public stats
    batch.py           chunk plan, worker pool, stall timeout
  analysis/
    statistics.py    per-agent and table stats (mergeable)
    reporting.py     terminal report, CSV/JSON export
  solver/            Phase 3: asymmetric-stack preflop model, cache and solvers
    gtopen.py        runs the GTOpen fork's preflop-cli and stores its strategies
    game.py          preflop tree with a stack per player (PokerKit betting and pot rules)
    hands.py         169 hand classes, batched dealing, vectorized 7-card evaluator
    cfr.py           chance-sampled Discounted CFR and sampled best-response evaluation
    parallel.py      multi-process sampling and evaluation
    config.py        solver configs (configs/6max_*.yaml)
    store.py         strategy cache: SQLite index + compressed strategy files
    pipeline.py      generate / inspect / simulation wiring
  strategies/solver.py  ExperimentalAgent strategy backed by the cache
  cli.py
configs/             experiment presets and solver configs (YAML)
strategies/          solved strategies (index.sqlite + files/*.npz)
docs/                Phase 3 GTOpen assessment and report
tests/               pytest suite
```

## How a hand runs

1. The controller picks the seats dealt in, the button (`hand % players`) and
   the stacks, then calls `play_hand`.
2. `play_hand` creates a PokerKit `NoLimitTexasHoldem` state with our rake
   callback, shuffles a deck with `Random(f"{seed}-deck-{hand}")` and deals
   hole cards, burns and boards explicitly. (PokerKit's own dealing uses the
   global `random` module and can't be seeded per hand.)
3. At each decision it builds an `Observation`: own hole cards, board, pot,
   stacks, bets, who is still in, position, street, this hand's action
   history, legal actions with min/max raise-to and amount to call, recent
   public hand records, opponents' public stats and the game config. It holds
   plain ints, tuples and frozen dataclasses only, never the PokerKit state.
4. The agent returns an `Action`. The engine validates it against the legal
   set and amount range. An illegal action or an exception is recorded and
   replaced by check (or fold if facing a bet); `--strict` raises instead.
5. At the end the engine checks `chips before == chips after + rake` and
   returns a `HandResult`. Agents then get the public record, which includes
   only cards actually shown at showdown.

A hand that exceeds `max_actions_per_hand` raises `StalledHandError` and is
skipped and counted. A chunk that returns nothing within `chunk_timeout_s`
stops the run with `StalledWorkerError`.

### Reproducibility

A run is split into fixed-size chunks (`chunk_size`). Each chunk rebuilds its
agents with seeds derived from `(seed, chunk)`, and cards depend only on
`(seed, hand number)`. Results are merged in chunk order, so 1 worker and 10
workers give identical numbers (checked in tests and at 10,000 hands). Two
configs with the same seed see the same cards hand for hand, which makes A/B
comparisons much less noisy than independent samples.

## Rake

`RakeConfig(percentage, cap, cap_by_players, no_flop_no_drop)`. The rake is
computed for the whole hand: `min(floor(total_pot * percentage), cap)`, zero
if no flop was dealt and no-flop-no-drop is on. `cap_by_players` maps
"at least N players dealt in" to a cap, for example `{2: 50, 3: 100, 5: 150}`.

What I found in PokerKit 0.7.7:

- It calls the rake function once per main or side pot and applies the cap
  to each pot separately, so a hand with a side pot could be raked twice the
  cap. pokerlab's callback works out the whole-hand rake and gives each pot a
  proportional share, rounded down. The chips actually raked can therefore be
  up to (number of pots − 1) chips below the nominal amount, never above it.
- Uncalled bets are refunded before pots are built, so they are never raked.
- Raked chips are taken before payout, so the rake is already in every
  player's net result. Net bb/100 is measured from chips; gross bb/100 adds
  back the rake attributed to the player (each pot's rake shared pro rata
  among that pot's winners).
- PokerKit only *warns* if a card is dealt twice. The engine never does that
  (it deals from its own shuffled deck), and the tests turn every `UserWarning`
  into an error.

## Statistics

Per agent: hands, hands won, net chips, net and gross bb/100 with standard
error and a 95% confidence interval, attributed rake, VPIP, PFR, 3-bet,
fold to 3-bet, aggression frequency and factor, saw flop, WTSD, W$SD, showdown
frequency, decision count and time, invalid actions, rebuys, plus results by
position and by seat. Table: rake, share of hands raked, average pot, flop
and showdown share, errors, runtime, hands/s and decisions/s. Denominators are
documented in `analysis/statistics.py`. 3-bet uses "faced exactly one raise";
fold to 3-bet uses "opened and faced exactly one re-raise".

The standard error treats hands as independent. A win rate whose 95% interval
includes zero is not evidence of an edge, and a positive interval only holds
for that particular lineup.

## The agents

All eight archetypes share a preflop range model: open, limp, call, 3-bet,
4-bet and bluff-3-bet ranges by position, scaled by raise size and number of
callers, with probabilistic mixing near range edges. Each has its own postflop
logic, driven by hand features (made-hand tier, flush and straight draws,
overcards, board texture), pot odds, bet size, number of opponents and, for
some, opponents' observed tendencies.

| archetype | postflop behaviour |
|---|---|
| Nit | bets strong hands only, calls only with strong hands at reasonable prices, almost no bluffs |
| TAG | c-bets, value-bets, semi-bluffs; calls when estimated equity beats pot odds, adjusted for opponent aggression |
| LAG | bets medium hands and draws, barrels more into players who fold, raises draws |
| CallingStation | calls with any pair or draw, sometimes ace-high; raises only monsters |
| Maniac | bets or raises most spots with large sizes, frequent shoves |
| LoosePassive | calls light, bets only strong hands and small |
| FitOrFold | bets when it hits, folds to any bet when it misses |
| Balanced | Monte Carlo equity against estimated ranges, pot-odds calls, value bets sized by strength, bluffs in proportion to bet size |

Every parameter can be overridden per seat in YAML:

```yaml
lineup:
  - {type: TAG, params: {aggression: 0.8, bluff_frequency: 0.2, open_range: {late: 0.35}}}
```

Measured over 100,000 hands, the styles come out as intended (VPIP/PFR):
Nit 6.6/4.4, TAG 14.2/11.0, LAG 23.4/18.4, CallingStation 44.4/2.9, Maniac
53.0/43.9, LoosePassive 33.1/3.1. They are test opponents, not strong players.

## Extending

**A new strategy for the experimental agent.** Subclass `Strategy` and
implement `decide(observation) -> Action`:

```python
from pokerlab.strategies.base import Strategy
from pokerlab.engine.actions import Action

class CFRStrategy(Strategy):
    def __init__(self, seed=0, table_path="tables/cfr.npz"):
        super().__init__(seed)
        self.table = load_table(table_path)

    def decide(self, obs):
        ...
        return Action.raise_to(amount)
```

Select it without code changes:

```yaml
lineup:
  - {type: Experimental, id: Hero, params: {strategy: "mypackage.cfr:CFRStrategy", strategy_params: {table_path: tables/cfr.npz}}}
```

`--strategy archetype:TAG` runs any archetype as the hero, which is useful as
a control group. Lookup tables, CFR policies, solver bindings (C++/Rust through
their Python wrappers), search or neural-network inference all fit the same
interface. `observe_hand` gives a strategy the public result of each hand for
opponent modeling.

**A new opponent.** Subclass `Agent` (or `HeuristicAgent` to reuse the
preflop model), implement `act`, and reference it as `type: mypackage.bots:MyBot`
or register it in `pokerlab/agents/__init__.py`.

**Helpers** in `agents/heuristics.py`: `analyze_hand`, `equity_vs_ranges`,
`estimated_ranges`, `heuristic_equity`, `PreflopSpot`, `bet_pot_fraction`,
`raise_to`, `all_in`, `table_aggression`, `table_fold_frequency`, `in_position`.

## Performance

Measured on a 10-core Mac, Python 3.14, default 9-handed lineup:

| run | workers | time | hands/s | decisions/s |
|---|---|---|---|---|
| 1,000 hands | 1 | 2.8 s | 356 | 4,990 |
| 10,000 hands | 1 | 27.8 s | 360 | 5,047 |
| 10,000 hands | 10 | — | identical results | — |
| 100,000 hands | 10 | 36.6 s | 2,731 | 38,133 |

Where the time goes (one worker, profiled):

- **PokerKit state handling**, roughly half the time: legality checks, card
  handling, pot computation. Showdown hand evaluation was the largest single
  cost (about 75% with random agents) until the engine started memoizing
  PokerKit's own `StandardHighHand.from_game` through a subclass. That was a
  2.3× speedup with identical results, and PokerKit still decides every showdown.
- **Monte Carlo equity** for Hero and Balanced: about 0.45 ms and 0.3 ms per
  decision, roughly 30% of the time. The rule-based archetypes take about 0.01–0.02 ms.
- The pure-Python 7-card evaluator runs at about 2.5 µs per hand and agrees
  with PokerKit on every random comparison in the tests.

Multiprocessing scales close to linearly because chunks are independent. If
equity estimation becomes the bottleneck for heavier strategies, the next step
is a compiled evaluator package (for example `phevaluator` or `eval7`) behind
`engine/cards.py`, or caching equities by (hand class, board texture).

## Preflop solver (Phase 3)

Six-max preflop strategies for games where **every player has their own stack**, solved offline and
cached for the experimental agent.

- **Solver:** a private fork of [GTOpen](https://github.com/MatthewPDingle/GTOpen),
  `lukejun001/gtopen-asymmetric`, checked out next to this repo as `../gtopen-asymmetric`. The fork
  adds per-seat stacks, PokerKit's all-in rules and side-pot terminals. pokerlab runs its `preflop-cli`
  binary (`pokerlab/solver/gtopen.py`).
- **Details and measured results:** [docs/phase3_report.md](docs/phase3_report.md) and
  [docs/gtopen_assessment.md](docs/gtopen_assessment.md).

```bash
# once, in ../gtopen-asymmetric
cargo build --release -p solver --bin preflop-cli
# in pokerlab
python -m pokerlab strategies generate --config configs/6max_gtopen_asymmetric.yaml   # solve + store
python -m pokerlab strategies inspect  --config configs/6max_gtopen_asymmetric.yaml   # summary, lookup latency
python -m pokerlab strategies inspect  --config configs/6max_gtopen_asymmetric.yaml --history "f-f-f-f"
python -m pokerlab strategies list
python -m pokerlab simulate --config configs/6max_gtopen_asymmetric.yaml --hands 10000
```

- **Validation:** the fork's trees are checked against PokerKit node by node and payout by payout
  (`tests/test_gtopen.py`, skipped if the fork isn't built).
- **Postflop:** flop terminals use GTOpen's approximate continuation model; postflop play itself comes
  from Phase 4 search (below).
- **Exact matches only.** A cached strategy is used only when variant, players, positions, every stack,
  blinds, ante, rake, abstraction and solver/model versions all match. Anything else, including an
  opponent's off-tree raise size, falls back to the heuristic strategy. Every decision is counted by
  source in the simulation report.
- **Cross-check solver:** `pokerlab/solver/cfr.py` (configs `6max_equal`, `6max_asymmetric`) is an
  independent NumPy solver used for comparison.

## Network and real-time search (Phase 4)

These parts play games that were never solved. Details and measurements are in
[docs/phase4_report.md](docs/phase4_report.md).

- **Preflop network** (`pokerlab/solver/network.py`, `models/preflop_net.pt`). Trained on GTOpen solves
  of random stack vectors, it gives a strategy for any stack vector in one forward pass, under a second
  per new vector. It also handles betting lines the tree lacks, such as an opponent's off-size raise.
- **Postflop search** (`pokerlab/postflop`). Heads-up spots are re-solved one street at a time with the
  fork's `postflop-cli`:
  - **flop:** depth-limited, with check-down equity leaves;
  - **turn and river:** full trees.

  Each street is solved within seconds. Ranges come from the preflop strategy actually played, then
  from each solve's reach. Multiway pots still fall back.

Agent parameters (`SolverStrategy`):

```yaml
strategy_params:
  network: models/preflop_net.pt        # preflop for unsolved stack vectors (exact solves still win)
  postflop: search                      # heads-up postflop by street search
  search_budget: {flop: 10, turn: 15, river: 5}   # seconds per street solve
  search_threads: 2
```

```bash
# once, in ../gtopen-asymmetric
cargo build --release -p solver --bin preflop-cli --bin postflop-cli
# in pokerlab
python tools/preflop_data.py generate --count 200 --threads 4 --push-every 2   # training solves
python tools/preflop_net.py train --arch mlp --epochs 80 --out models/preflop_net.pt
python tools/preflop_net.py evaluate --model models/preflop_net.pt --tests 20  # exact NashConv
python tools/postflop_eval.py --hands 300                                      # search vs fallback
```

## Known limitations

- Rake per pot is rounded down, so the house can be short by up to
  (pots − 1) chips on a multi-pot hand.
- Pro-rata rake attribution in a split pot with an odd chip can differ from
  the exact "unraked" counterfactual by a fraction of a chip.
- In continuous mode, stats and the confidence interval treat hands as
  independent even though stacks carry over.
- Opponent ranges in `equity_vs_ranges` are top-X% by preflop strength. Real
  ranges are not linear (they include suited connectors and bluffs).
- No antes or straddles, and one board only (no run-it-twice).
