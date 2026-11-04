# Phase 3 report: asymmetric-stack six-max preflop solving with GTOpen

pokerlab branch `phase3-asymmetric-preflop`, plus the private GTOpen fork
[`lukejun001/gtopen-asymmetric`](https://github.com/lukejun001/gtopen-asymmetric) (fork version `asym-3`).

Every number below was measured on one machine: Apple Silicon, 10 cores, 16 GB, CPU only, mostly on
battery power. Raw outputs are in `reports/phase3/`.

## Outcome against the definition of done

| Requirement | Status |
|---|---|
| Six-player model where every player has their own stack | **Done.** In the GTOpen fork (`PreflopConfig.stacks`) and in pokerlab's own model. The stack vectors are test cases, not limits. |
| Correct betting and all-in logic | **Done.** PokerKit's no-limit rules are ported into GTOpen and checked against PokerKit node by node (`tests/test_gtopen.py`). |
| Side pots and rake | **Done.** Pot structures and concrete-card payouts match PokerKit; chips are conserved on every flop terminal (Rust test). |
| Solver produces a genuine asymmetric-stack preflop strategy | **Done.** GTOpen solves 100/150/70/115/40/200 to an **exact NashConv of 0.19 bb/100** in the modeled game. |
| Store and retrieve strategies | **Done.** SQLite index plus compressed files, exact keys, about 21 µs per cached lookup. |
| Experimental agent uses them in reproducible offline simulations | **Done** (see Simulations). |
| Existing simulator functionality intact | **Done.** 122 pokerlab tests pass. In GTOpen, 65 library tests (1 ignored; 4 of them new) and 56 preflop integration tests pass. |
| Runtime and memory measured | **Done.** |
| Limitations documented | **Done.** This report. |

## How we got here (pivots, in order)

1. **GTOpen inspected first** ([assessment](gtopen_assessment.md)). The preflop engine has one shared
   stack, and the repository has no license. A separate NumPy solver was built at first. The project
   owner then chose GTOpen for private research, so the modified code lives only in a private fork.
2. **Fork, part 1:** per-seat stacks, PokerKit's betting rules, side-pot terminals, and a `preflop-cli`
   binary. Validated against PokerKit before any solving.
3. **First solves stalled** for some stack vectors (a rotated vector plateaued at 14.5 bb/100).
   - Diagnosis (`reports/phase3/diagnostic_rot1.json`): neither rake nor the realization model was the
     cause.
   - GTOpen's regret-based pruning stops updating the *average* strategy in skipped subtrees. In
     unequal-stack trees these stale averages persist, and best responses exploit them.
   - Fix: pruning is off by default when stacks differ. Equal-stack behavior is unchanged.
4. **A valuation bug found in strategy review.**
   - With pruning fixed, the solves converged, but the stored strategies contained an absurd line: a
     40 bb small blind opened 100% of hands.
   - Cause: after a *covered* all-in (for example, a 200 bb jam called by 40 bb), only the refunded
     player was valued on the matched pot. The covered player was still valued on the gross pot,
     including the refund.
   - Fixed in `asym-3`, with a direct test plus conservation checks on every terminal. The affected
     runs' metrics are kept in `reports/phase3/refund_bug_run/` and `pruned_run/`. No strategy from
     them is used.

## Experiments (GTOpen fork, CPU, 10 threads)

Common settings for all three runs:

- **Abstraction `gtopen-6max-v1`:** open 2.5 bb (SB 3 bb), re-raises 3× the bet faced, at most 3 raises,
  all-in always available, a raise above 50% of the stack becomes all-in, no limps.
- **Rake:** 4.5% with a $3 (0.06 bb) cap, no flop no drop.
- **Continuation model:** GTOpen's `balanced` heads-up model and `coupled_deck_v1` multiway model. Pots
  with a side pot or refund use raw equity, with no realization adjustment.
- **Solving:** Discounted CFR, 400 iterations, exact best-response gaps every 50 iterations.

| | Equal 100 bb | Asymmetric UTG 100, HJ 150, CO 70, BTN 115, SB 40, BB 200 | Rotated: same stacks moved one seat (BB 40, UTG 200) |
|---|---|---|---|
| Tree nodes / decision nodes | 20,656 / 9,723 | 20,863 / 9,764 | 21,620 / 10,085 |
| Information sets (×169 classes) | 1.64 M | 1.65 M | 1.70 M |
| Regret + strategy arena | 27.9 MB | 28.2 MB | 29.2 MB |
| Peak memory (whole process) | 372 MB | 361 MB | 375 MB |
| Solve time, 400 iterations | 1,399 s (688 s in an earlier run on mains power; battery throttling) | 1,444 s | 1,550 s |
| CPU utilization (10 cores) | 58% | 64% | 69% |
| Exact NashConv: 50 / 100 / 200 / 400 iterations, bb/100 | 16.6 / 3.55 / 1.09 / **0.62** | 17.6 / 4.26 / 0.91 / **0.19** | 7.3 / 1.60 / 0.34 / **0.09** |
| Largest single-player best-response gap at 400 iterations | 0.28 bb/100 (UTG) | 0.08 bb/100 (BB) | 0.02 bb/100 |
| Stored strategy | 3.8 MB | 3.9 MB | 3.9 MB |
| GPU | not used (CPU build) | not used | not used |

**What NashConv means here.** It is computed exactly, over all 169 hand classes, with no sampling, but
only **within the modeled game**: the abstraction plus GTOpen's continuation approximations.
- It shows the strategies are close to a stationary point of that model: no single player can gain
  more than 0.08–0.28 bb/100 by deviating.
- It does **not** show closeness to an equilibrium of real six-max no-limit hold'em, which has full
  bet-size freedom and real postflop play.
- Six-player CFR also has no convergence guarantee. These are measured results, not a proof.

**Unequal all-in possibilities.** The asymmetric tree contains 9,982 flop terminals. Of these, 2,027
have two pots, 806 three, 184 four and 19 five, and 2,208 involve an uncalled-bet refund. One example:
UTG and HJ fold, CO opens, BTN and SB call, BB 3-bets, CO calls, BTN jams 115 bb, and three players call.
A larger tree with one more raise level (103k nodes, 1.6 GB, about 24 minutes) was solved only before
the refund fix. Its result is therefore void and was not re-run within the time budget.

### Do individual stacks change the strategy?

Yes. Some measured contrasts (combo-weighted frequencies; full tables in `reports/phase3/comparison.json`):

| Spot (action) | Equal 100 bb | Asymmetric | Rotated | Why |
|---|---|---|---|---|
| UTG opens (raise first in) | 18.9% | 19.1% | 19.1% | Opens barely depend on depth in this coarse tree. |
| BTN opens | 41.1% | 38.7% | 39.1% | The 70 bb CO behind (asymmetric) or a 40 bb BB (rotated) changes the button's opening range a little. |
| SB opens vs the BB | 53.5% | 52.6% | 52.5% | **Effective** stack 40 bb in both asymmetric games (SB 40 vs BB 200; SB 115 vs BB 40) gives near-identical play (difference measure 0.0007). |
| BB vs SB open: 3-bet to 9 bb / all-in | 15.6% / 0% | 0.6% / 15.6% | 0% / 16.0% | At 40 bb effective, the 3-bet becomes a jam. |
| BB calls an SB all-in | 5.5% (vs 100 bb) | 11.3% (vs 40 bb) | 11.8% (vs 40 bb) | Shorter all-ins are called wider. |
| SB vs BTN open: 3-bet / all-in | 8.5% / 0% | 0.9% / 9.7% | 7.2% / 0% | Only the 40 bb SB switches to jamming. |
| UTG vs HJ 3-bet: 4-bet to 22.5 bb | 17.5% | 23.0% | 36.0% | The rotated UTG has 200 bb behind. |

Combo-weighted mean change in strategy, equal → asymmetric: up to 0.21 at the BB's defense against the
SB, and **28% of all combos change their action by more than 10 points**. Asymmetric → rotated: up to
0.20 at the UTG-versus-3-bet spot (54% of combos change). Same stacks in different seats are therefore
distinct games, not one game relabelled.

## Strategy cache and agent

- **Lookup:** `python -m pokerlab strategies inspect --config configs/6max_gtopen_asymmetric.yaml`
  - first load: about 0.1 s for a 3.9 MB file;
  - cached lookup: **about 21 µs**.
- **Exact matches only.** Any change to stacks, rake, blinds, players, abstraction or solver version is
  a miss, and the agent falls back to its heuristic.
- **Off-tree histories.** An opponent's raise size that isn't in the tree is not mapped to the nearest
  size; the agent falls back.
- **Database:** 18 strategies, 51 MB in total:
  - three GTOpen strategies for the named configs (3.8–3.9 MB each);
  - 13 GTOpen library strategies (next sections);
  - two from the cross-check solver.

### Simulations (offline, PokerKit engine, 10,000 hands each, stacks dealt by position every hand)

| Run | Hero result, net bb/100 [95% CI] | Hero's preflop decisions that were exact | Fallbacks | Notes |
|---|---|---|---|---|
| `6max_gtopen_asymmetric`: Hero vs TAG, LAG, CallingStation, Nit, Balanced | **+45.1** [+13.3, +76.9] (gross +45.6) | 76.5% (7,662 of 10,016) | 2,354 off-tree histories (opponent sizes), plus every postflop decision | Most of the table's profit comes from the CallingStation (−164 bb/100). LAG is +42.9 and Balanced +29.9, with overlapping intervals. |
| `6max_gtopen_asymmetric_selfplay`: six solver agents | between −28.2 and +23.7 per agent; **every interval includes 0** | **100%** (61,000 of 61,000) | postflop only | All six seats play the same cached strategy. As expected, no seat is distinguishable from break-even after rake. |
| `6max_gtopen_equal`: Hero vs the same opponents | +64.9 [+29.4, +100.4] | 76.5% | 2,353 off-tree histories | Same picture with equal stacks. |

Rake is 3.2 bb/100 for the table (the $3 cap binds early at $25/$50). Speed: about 1,400 hands/s with
8 workers. These win rates come from 10,000 hands against simple heuristics with a heuristic postflop
game, so they are not evidence of real-game strength.

Positions rotate every hand, so each player sits at every stack depth in turn. The table's chip total
equals minus the rake in every run (checked by the simulator).

## Validation

| Test | Where | What it checks |
|---|---|---|
| Tree legality | `tests/test_gtopen.py` | Every sampled node of the fork's trees (asymmetric with 3 and 4 raise levels, three random stack vectors, heads-up) is replayed in a real PokerKit state. Actor, fold/check availability, the legality of every raise and all-in amount, all-in = PokerKit's maximum, and **no raise where PokerKit forbids one** (e.g. after a short all-in) must all agree. |
| Pot structure | `tests/test_gtopen.py` | For every sampled terminal: contributions after refunds and each pot's eligible players are exact; amounts after rake are within 1 chip of pokerlab's PokerKit-validated pot builder. |
| Test A: equal stacks | both repos | The per-seat path with six 100 bb stacks reproduces GTOpen's original single-stack engine: same tree, bit-identical terminal values (Rust), and after a 10-iteration solve a **maximum strategy difference of 0.0** with identical NashConv. |
| Test B: asymmetric stacks | both | Each seat's jam and all-in call use its own stack, and the CO (70 bb) cannot re-raise a 100 bb jam. |
| Test C: unequal all-ins | `test_gtopen.py` | For three-pot terminals, payouts computed from the fork's pots match PokerKit's actual payouts for concrete cards (within rounding). |
| Chip conservation | Rust `asym_tests` | On every flop terminal (500+), the players' expected values sum to −rake. A covered all-in can only win the matched pot. |
| Test D: rake | `test_gtopen.py` | Zero rake leaves the tree unchanged; pots sum to contributions; rake is never above the cap; there is no rake without a flop. |
| Test E: positions | `test_gtopen.py`, experiments | Six rotations give six keys and six different root action sets, and the rotated game solves to a different strategy (see above). |
| pokerlab model | `tests/test_solver_game.py` | The independent model behind the PokerKit checks: legality, payouts, 400 random full games. |

**The one rake difference from PokerKit.** At showdown, PokerKit mucks losing hands and merges pots
before raking. With per-pot rounding down, the total rake can differ by at most (pots − 1) chips: with
1 chip = $0.01, that is 0.0002 bb.

## Known mathematical approximations

1. **No postflop solving.** GTOpen values flop terminals with an approximate continuation model:
   - heads-up pots: relative hand-class equity priors (`balanced`), minus the configured rake;
   - 3+ players: coupled-deck sampled showdown strengths, which approximate card removal;
   - side pots and refunds: raw equity per pot, with no positional realization.
2. **Hand classes, not combos.** GTOpen's equity tables are class against class, so card removal
   between opponents is only approximate.
3. **A restricted betting tree.** One open size, one re-raise multiplier, three raises, no limps.
4. **No antes or straddles** with per-seat stacks (rejected explicitly), and no player models (seat
   profiles) with per-seat stacks.
5. **The convergence measure is exact only within the model** (see above).
6. **Rake rounding** follows pokerlab's whole-hand cap with per-pot floor rounding. PokerStars' exact
   rounding rule was not confirmed from an official source.

## Unsupported situations

- **All postflop decisions.** The agent uses its heuristic fallback.
- **Preflop histories outside the tree:** other raise sizes, limps, a fourth raise.
- **Any game not solved exactly:** other stacks, other rake, a different player count.
- Unsolved stack vectors can be approximated from a library (next section), but no approximation has
  met the 1 bb/100 bar yet, so in the default `validated` mode the agent still falls back for them.

## Generalizing to any stack vector (pilot)

**Method** (`pokerlab/solver/generalize.py`, version `effstack-knn-1`):
- **Per-decision matching.** Each decision borrows from the library game whose effective stacks (the
  actor against each opponent still in the hand, log scale) are closest at that decision.
- **Translation.** Actions are mapped by role (fold, call, the n-th raise size, all-in), with a relaxed
  second pass when a size doesn't exist.
- **Exact scoring.** The fork's evaluate mode installs the borrowed profile in the target game and
  reports its exact NashConv, in about 5 s per vector.
- **Calibration.** The agent uses an approximation only inside a radius where measured NashConv stayed
  under the threshold (1 bb/100).

**Library:** 13 random vectors (log-uniform 20–250 bb per seat, seed 7, 200 iterations), plus the
three earlier GTOpen games, 16 in all.
- Every library solve converged below 1 bb/100 (0.10–0.90).
- Solves took 5.5–17 minutes each on battery with Low Power Mode on.
- The planned 16 random vectors were cut to 13 because the Mac kept sleeping.

**Measured error of borrowed strategies** (exact NashConv in the target game, k = 1):

| Test set | Vectors | Median | Best | Worst | Under 1 bb/100 |
|---|---|---|---|---|---|
| Held out (seed 11) | 20 | 19.3 | 5.7 | 38.0 | 0 |
| Leave one out (library games) | 16 | 18.1 | 10.2 | 38.3 | 0 |

**Result: the method does not yet produce validated approximations.**
- **Radius 0.** No vector came within 1 bb/100, so calibration chose a radius of 0 and the agent
  keeps falling back in `validated` mode (`strategies/approximation.json`).
- **Reference points.** With only 2–3 library games the rotated game scored about 190 bb/100. A converged
  solve scores 0.1–0.9.
- **The matching statistic doesn't predict the error.** Across the held-out vectors, the error
  correlates 0.58 with the whole-table distance but only 0.19 with the per-decision p90 distance used
  for calibration. Leave-one-out gives 0.56 and −0.29.
- **Low coverage.** The nearest library vector is typically 0.5–1.4 apart in log effective stack (a
  factor of 1.6–4). Thirteen points cannot cover a six-dimensional space at the resolution this needs.

**Error versus library size** (partial: the first 4–5 held-out vectors only, random sub-libraries, k = 1;
stopped because the battery ran low; `reports/phase3/library/curve.log`):

| Library games | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Median NashConv, bb/100 | 404 | 258 | 58 | 28 | 25 |
| Rows measured | 10 | 10 | 8 | 8 | 4 |

- **The error falls steeply up to 8 games, then flattens:** 28 at 8 games, 25 at 16.
- **Extrapolating is unreliable from this sample, but it isn't promising:**
  - the 8→16 step suggests a floor well above 1 bb/100;
  - even error ∝ 1/n would need about 400 games to reach 1 bb/100;
  - error ∝ 1/√n would need thousands.
- **Likely source of the floor: translation.** Borrowed trees have different all-in and raise
  structures, so even a close stack match plays some lines that don't fit the target tree.

**Before scaling this up in the cloud**, check three things:
1. **Finish the curve** on all 10 held-out vectors, then add 16→64 games. This shows whether the
   floor is real.
2. **A better calibration statistic.** Whole-table distance or reach-weighted node distance, for
   example.
3. **Blending (k = 3)** against borrowing from the single nearest game.

If the floor holds, the practical route to "any stack size" is to solve each new stack vector directly:
- one solve takes about 6 minutes at 200 iterations on this Mac, and a cloud machine can do it on
  demand;
- warm-starting from the nearest library game should shorten that;
- once solved, a vector is an exact cache hit.

## Solving on demand (cloud, 4 cores)

Chosen direction: when a stack vector isn't solved, solve it directly; growing the library further
was not pursued.

**What exists now** (`pokerlab/solver/ondemand.py`):
- **`solve_on_demand`** solves a new vector with GTOpen and stores it as an ordinary exact strategy,
  which also adds it to the library.
- **The agent's `on_demand` option:**
  - `queue` records unsolved games in `strategies/pending.jsonl` and falls back for now;
    `pokerlab strategies solve-pending` solves the most-met games later.
  - `solve` blocks and solves right away. That is for interactive use only.
  - `off` is the default.
- **Warm start in the fork** (`preflop-cli` `warm_start`, `PreflopSolver::warm_start_node`). A solve can
  start from a given profile, here the library approximation. Three settings control it:
  - each node's strategy sums become the profile times the actor's reach under it, times `sum_scale`;
  - regrets are seeded in proportion to the profile, at the first iteration's regret mass rescaled to
    the profile's opponent reach, times `regret_scale`;
  - DCFR's iteration counter is set to `iterations`.

  Seeded with its own converged strategy, a heads-up solve stays at equilibrium. Frozen with huge scales,
  a 6-max solve reproduces the borrowed profile's measured NashConv: 13.5 against 12.8 bb/100 (48
  untranslatable nodes play uniformly).

**Does warm-starting speed up a solve?** Measured on held-out vector `[63, 82, 206, 65, 72, 88]`, where
the borrowed profile scores 12.8 bb/100. Exact NashConv in bb/100, with solve seconds on 4 threads in
brackets (`reports/phase3/warm/compare_long.json`):

| Iteration | Cold | Warm r10 s30 t60 | Warm r10 s100 t300 |
|---|---|---|---|
| 20 (~63 s) | 57.8 | 20.4 | **13.7** |
| 40 (~153 s) | 18.1 | 10.9 | **10.3** |
| 60 (~241 s) | 7.8 | **6.8** | 8.3 |
| 80 (~324 s) | **4.0** | 4.7 | 6.8 |
| 100 (~404 s) | **2.4** | 3.4 | 5.7 |
| 140 (~556 s) | **1.09** | 2.1 | 4.3 |
| 200 (~765 s) | **0.50** | 1.32 | 2.96 |

**Result: warm-starting only helps short solves.**
- **Short budgets.** Warm starts lead up to about 60 iterations, reaching 13.7 bb/100 after one minute
  where cold is at 57.8.
- **Full solves.** Cold DCFR overtakes every warm setting by about 70 iterations and ends 2.6 times
  lower than the best warm setting at 200 iterations.
- **Why.** A prior heavy enough to survive the first iterations stays in the average and puts a floor
  under it. A prior light enough to fade is overwritten by CFR's early swings and loses its head start: the light
  settings in `compare_tune3.json` and `compare_tune4.json` (r1 s7 t20, r3 s10 t30) were at 100–184
  bb/100 after 5 iterations, far above the 12.8 they started from.
- **Default.** `warm="auto"` therefore warm-starts only budgets of at most 60 iterations. On-demand
  solves default to 200 cold iterations: about 12.4 minutes on 4 cores, and about 1 bb/100 after 140
  iterations (9 minutes).

**Caveats:**
- **One vector.** The 200-iteration comparison used one held-out vector. The short tuning runs
  (`compare_tune*.json`) used the same vector.
- **Tuning history.** The first seeding scaled everything from a uniform-play iteration. Those runs
  (`compare_tune.json`, `compare_tune2.json`) were worse than cold because seeds on main-line nodes
  were 10–100 times too small. `compare_tune3.json`, `compare_tune4.json`, `compare_diag.json` and
  `compare_long.json` use the reach-weighted seeding.

## Phase 4 roadmap (prioritized)

1. **Postflop.** Connect heads-up flop terminals to GTOpen's postflop solver, starting with the
   highest-reach lines in these trees. Ranges come from the exported reach; stacks come from the
   per-seat remainders.
2. **Action translation** for off-tree opponent sizes, validated by measuring the exploitability of
   translated strategies before any decision is labelled "approximate".
3. **Generalization across stacks.** Solve small grids (one position's stack varied) and test
   interpolation against direct solves.
4. **Richer abstraction.** Two open sizes, a fourth raise level, and SB limps. The re-jam tree (103k
   nodes) needs about 24 minutes per 400 iterations; budget it on mains power.
5. **Upstream-quality fixes.** The pruning issue also affects GTOpen's equal-stack engine in principle,
   since stale averages occur off-path. Quantify it, and report both findings to the author if a
   license is added.
6. **Multiway postflop**, only after the above.

## Reproducing

```bash
# in ../gtopen-asymmetric
cargo build --release -p solver --bin preflop-cli
cargo test --release -p solver --lib asym_tests
# in pokerlab
python -m pytest
python -m pokerlab strategies generate --config configs/6max_gtopen_asymmetric.yaml
python -m pokerlab simulate --config configs/6max_gtopen_asymmetric.yaml --hands 10000
python tools/phase3_experiments.py compare
python tools/warm_start.py compare --tests 1 --iterations 200 --check-every 20 --label long \
    --variants r10s100t300 r10s30t60 cold      # warm vs cold on-demand solves
```

The earlier NumPy solver (`pokerlab/solver/cfr.py`, configs `6max_equal` and `6max_asymmetric`) remains
as an independent cross-check. Its sampled convergence estimates are far noisier: on the asymmetric
game, the held-out best-response estimate was −15 bb/100 (a lower bound) and the in-sample estimate
+456 bb/100 (an upper estimate).
