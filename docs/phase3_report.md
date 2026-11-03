# Phase 3 report: asymmetric-stack six-max preflop solving

Branch `phase3-asymmetric-preflop`. Every number here was measured on one machine: Apple Silicon,
10 cores, 16 GB RAM, CPU only, Python 3.14, NumPy 2.5. Raw outputs are in `reports/phase3/`.

## Outcome against the definition of done

| Requirement | Status |
|---|---|
| Six-player game model with a different stack per player | **Done.** `pokerlab/solver/game.py` handles any 2–9 player stack vector; the configs are test cases, not limits. |
| Correct betting and all-in logic | **Done.** Ported from PokerKit and checked against PokerKit node by node, plus a randomized full-game property test (400 random games, any raise size). |
| Side pots and rake | **Done.** Payouts match PokerKit's actual payouts for concrete cards. The one documented exception is ≤ (pots − 1) chips of rake rounding (see below). |
| A solver producing a genuine asymmetric-stack strategy | **Done**, with our own solver: GTOpen is unlicensed and equal-stack only ([assessment](gtopen_assessment.md)). |
| Store and retrieve strategies | **Done.** SQLite index plus compressed files; exact keys; mixed strategies preserved. |
| Experimental agent uses them in reproducible offline simulations | **Done.** Every decision is tagged exact or fallback. |
| Existing simulator functionality intact | **Done.** All 87 original tests still pass. |
| Automated tests pass | **Done** (see Validation). |
| Solver runtime and memory measured | **Done** (see Experiments). |
| Limitations documented | **Done.** This report. |

RESULTS_PLACEHOLDER

## Method

### Game model (`solver/game.py`)

- Players are indexed in preflop order (0 acts first; `n-2` is the SB, `n-1` the BB). Every player has
  their own starting stack in integer chips (1 chip = $0.01 at $25/$50, so 1 bb = 5,000 chips).
- **Betting is a port of PokerKit's rules:**
  - a call is capped at the caller's stack;
  - a raise's legal range is `[min_to, stack + bet]`, using PokerKit's minimum
    (`max(last full raise, bb) + current bet`);
  - an all-in raise smaller than a full raise does not reopen betting for players who already acted
    (PokerKit's `acted_player_indices` and `consecutive_short_all_in_amounts`);
  - a player who is covered, or whose opponents are all folded or all-in, cannot raise.
- **The betting abstraction is configurable.** It sets open sizes, an SB open size, 3-bet and 4-bet
  multipliers, a cap on sized raises, a cap on total raises, a limit on flat calls per raise, all-in
  always available, and a threshold above which a sized raise becomes all-in. Every abstract size is
  clamped into the actor's own legal range.
- **Terminals** are either a fold-out or "the flop" (betting closed with two or more live players).
  For each terminal the tree records the exact pots PokerKit builds:
  - the bet above the second-highest bet is refunded (folded players included);
  - pots are layered at every distinct contribution, and adjacent layers with the same eligible
    players are merged;
  - rake is the whole-hand `min(floor(4.5% × pot), cap)`, split across pots pro rata and rounded down,
    with no rake when there is no flop.

### Solver (`solver/cfr.py`, `solver/parallel.py`)

- **Chance-sampled Discounted CFR** (α = 1.5, β = 0, γ = 2).
  - Each iteration, each of 8 worker processes deals 4,096 complete hands: six hole-card pairs and a
    five-card board, without replacement, so **card removal is exact**.
  - The whole public tree is walked once per batch, vectorized over the deals with NumPy, and regrets
    are updated for every player at every node.
- **Information sets** are (node, hand class). Preflop, the 169 classes lose no information.
- **Average strategies** are weighted by each class's exact own reach, so no sampling noise enters
  those weights.
- **Continuation model `showdown-v1`:**
  - at a flop terminal the hand is checked down on the dealt board;
  - each main and side pot goes to its best eligible hand(s), net of that pot's rake;
  - this value is exact for all-in pots, and an approximation (no postflop betting) otherwise.
- **Convergence is measured, not assumed.** For each player we estimate the best-response gain against
  the average strategy:
  - *in-sample:* a plain empirical best response, chosen and scored on the same 50k deals. This is
    biased upward by sampling noise and is an *upper* estimate;
  - *holdout:* a conservative best response that deviates only where the gain exceeds 2 standard
    errors, chosen on one 50k-deal sample and scored on an independent one. This is an unbiased value
    of a real deviation, so it estimates a *lower* bound.

  NashConv is the sum of the six players' gains. With three or more players, CFR carries no Nash
  guarantee, and a small NashConv in the abstract game says nothing about the real game.

### Strategy cache (`solver/store.py`)

- **Key:** SHA-256 of variant, players dealt in, positions, button index, all six stacks (chips), blinds,
  ante, rake (percentage, cap, cap-by-players schedule, no-flop-no-drop), the full abstraction, tree
  version, solver version (`pokerlab-cs-dcfr-1`), continuation version (`showdown-v1`) and strategy
  model version (`approx-eq-preflop-1`).
- **Storage:** metadata lives in SQLite (`strategies/index.sqlite`). Probabilities are stored
  per node and per class (float32, renormalized on load) in `strategies/files/*.npz`.
- **Betting history** is the node address inside a strategy, e.g. `f-r12500-c` (tokens with chip amounts).
- **Approximate matches:** `nearest()` lists games with the same key except stacks, together with their
  stack distance, as the starting point for interpolation work. The agent uses them only when
  `allow_approximate` is on. It is off by default, because no approximation has been validated.

### Agent integration (`strategies/solver.py`)

For each preflop decision the agent:

1. rebuilds the game from the observation (each position's starting stack = current stack + chips in);
2. computes the key and looks it up;
3. translates the public history into tokens;
4. samples from the stored mixed strategy with the agent's seeded RNG.

Decisions are counted as `exact:<key>`, `approximate:<key>`, `fallback:no_matching_strategy`,
`fallback:history_not_in_abstraction` or `fallback:postflop`. The fallback is the existing heuristic
`EquityStrategy`. The simulator gained `stacks_by_position_bb`, so every hand can be dealt at the
configured stack vector while the button rotates.

## Validation (`tests/test_solver_*.py`)

| Test | What it checks |
|---|---|
| Legal actions | Every node (or a 1,500-node sample) of four trees, including stacks (40, 41, 300, 12, 9, 80) bb and heads-up, is replayed in a real PokerKit state. Actor, call amount, raise availability and `[min, max]` bounds must be equal, and every abstract raise must be legal in PokerKit. |
| Randomized rules | 400 random games (2–6 players, random stacks, random raise sizes) step through our state machine and PokerKit side by side. |
| Test A: equal stacks | With six 100 bb stacks, the vector model reproduces an independent single-stack, single-pot model (GTOpen's v1 semantics, written separately in the test). Measured tolerance: 0 differing terminals, 0 chips. |
| Test B: asymmetric stacks | Root and early-node action sets, all-in amounts capped at each player's own stack, and short all-ins that do not reopen betting (also checked in PokerKit). |
| Test C: unequal all-ins | UTG 100 bb all-in, CO 70 and SB 40 call all-in, BB calls: three pots of 800k / 450k / 300k chips with eligibility (UTG, CO, SB, BB) / (UTG, CO, BB) / (UTG, BB), and rake 154 / 87 / 58. Payouts are compared with PokerKit on 60 deals, plus random all-in property tests on random stack vectors. |
| Terminal payouts | 120 random terminals from each of the equal and asymmetric trees, three card deals each, compared chip by chip with the PokerKit engine's actual payouts. |
| Test D: rake | Raked and zero-rake trees have identical structure. Fold-outs are never raked, every flop terminal satisfies `sum(payoffs) = −rake` with rake ≤ cap, rake only lowers winners' payouts, and the cap is reached. A solver-level check that terminal values conserve chips is included. |
| Test E: positions | Rotating the stack vector gives six distinct keys, six distinct root action sets and different tree sizes. A 3-player jam/fold solve shows the solution changes (button jam frequency moves by more than 10 points). |
| Solver | Heads-up 10 bb push/fold reproduces the published Nash ranges (SB jams ~57%, BB calls ~38%; AA always, 72o never). Seeded runs are deterministic. |
| Cache and agent | Round trip with mixed probabilities preserved. A changed stack, rake or abstraction is a miss. The agent uses exact strategies with no invalid actions and reproducible results, and falls back when one stack differs by 1 bb. |

**The one rake difference from PokerKit.** At showdown, PokerKit mucks losing hands and then merges pots
whose remaining eligible players coincide, before raking. Because per-pot rake is rounded down, the
total rake can therefore depend on who wins, by at most (pots − 1) chips; at 1 chip = $0.01 that is
0.0002 bb. The model uses the pre-showdown pot structure, and the tests assert this exact bound.

## Known mathematical approximations

1. **No postflop play (`showdown-v1`).** Every pot that reaches the flop is valued as if checked down.
   - Equity is fully realized; there is no positional advantage after the flop, no fold equity
     postflop, and no implied odds.
   - This overvalues calling with hands that realize poorly and undervalues position.
   - It is exact only for pots where everyone is all-in.
2. **A restricted betting tree.** One open size (2.5 bb; SB 3 bb), one 3-bet size (3×), 4-bets and
   later raises all-in only, no limping, and at most 1 or 2 flat calls per raise. Multiway pots outside
   these lines do not exist in the model.
3. **Sampled CFR with a finite budget.** Estimated bounds are reported; nothing is claimed about
   convergence to an equilibrium of the six-player game, abstract or real.
4. **The rake model** follows PokerStars' documented structure (percentage, cap per hand, cap by players
   dealt, no flop no drop). Per-pot floor rounding is an assumption; the brief's $3 cap for 5+
   players at $25/$50 matches the cap PokerStars historically published (one source reports later
   increases).
5. **No antes or straddles** (rejected explicitly), and every stack must exceed the big blind.

## Unsupported situations

- **All postflop decisions.** The agent falls back to the heuristic strategy, including in pots the
  preflop model assumed would be checked down.
- **Any preflop history the abstraction does not contain.** This includes other open or 3-bet sizes,
  limps, cold calls beyond the limit, and 5-bets that are not all-in. Off-tree actions are never mapped
  to the nearest size; the agent falls back.
- **Any stack vector, rake, player count or position assignment that has not been solved.** There is no
  interpolation yet.
- **Games with antes, straddles or stacks of 1 bb or less.**

## Phase 4 roadmap (prioritized)

1. **Postflop continuation values.** Replace `showdown-v1` heads-up flop terminals with values from a
   heads-up postflop solver, keyed by preflop line, stack-to-pot ratio and ranges. Start with the most
   frequent flop terminals in the solved trees.
2. **Action translation.** Map off-tree opponent sizes onto the abstraction (pseudo-harmonic mapping),
   validated by measuring the exploitability of the translated strategy, before labelling any such
   decision "approximate".
3. **Generalization across stacks.** Solve grids of stack vectors (one position varied at a time) and
   test interpolated strategies against direct solves on held-out vectors. Only then enable
   `allow_approximate`.
4. **Lower-variance evaluation.** The six-player upper bound is noise-dominated at 50k deals. Use
   control variates (subtract exact heads-up all-in equity), more deals, or per-node local
   best-response tests to tighten it.
5. **Performance.** Per-node NumPy overhead dominates. A compiled inner loop (Numba or Rust behind
   PyO3) could give 10–50× and allow 2–3 sizes per raise and more multiway lines.
6. **Multiway postflop.** Only after 1–3: three-way flop subgames for the most frequent multiway
   terminals.
