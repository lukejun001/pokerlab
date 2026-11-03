# GTOpen compatibility assessment (Phase 3, Milestone 1)

Inspected: <https://github.com/MatthewPDingle/GTOpen> at commit `6c70502` (2 Nov 2026), by reading the
source and documentation. Nothing from GTOpen is copied, vendored or modified in this repository.

## Summary

GTOpen is the preflop solver for Phase 3. It runs as a **private, modified fork**,
[`lukejun001/gtopen-asymmetric`](https://github.com/lukejun001/gtopen-asymmetric), that is used only
for this offline research project. Two findings shaped how:

1. **License.** The repository has no license file, and GitHub reports `license: null`. Without a license,
   default copyright applies and nothing grants permission to modify or redistribute the code.
   - The project owner chose to use it anyway for private research.
   - The modified code therefore lives only in the private fork, and none of it is copied into this
     (public) repository.
   - pokerlab only calls the fork's binary as a subprocess (`pokerlab/solver/gtopen.py`).
2. **Architecture.** Upstream, the preflop solver models one shared stack. The code says so explicitly:
   `PreflopConfig.stack: f64`, documented as *"v1: must be equal for all seats — no side pots"*. The fork
   adds per-seat stacks (see "What the fork changes" below).

An earlier stage of Phase 3 built a separate NumPy solver (`pokerlab/solver/cfr.py`) because of finding 1.
It remains as an independent cross-check. The fork is the production path.

## How GTOpen's preflop solver works

| Question | Finding (file references are to GTOpen) |
|---|---|
| Language and interface | A Rust workspace. `crates/solver` holds the engine and `crates/server` is an axum HTTP server on `127.0.0.1:3737` with a browser UI. Preflop solving is exposed only over HTTP (`/api/preflop/spot`, `/solve`, `/node`, `/export`). `solve-cli` handles postflop spots only. There is no Python API; Python could drive it only through HTTP or a subprocess. Optional CUDA support via `cudarc`. The repository is 5.3 GB, mostly `research/` artifacts. |
| Tree representation | A flat `Vec<PNode>` (`preflop/mod.rs`). Each node stores actor, actions (`fold/check/call/raise/jam` with a TO amount in bb as `f64`), `invested: Vec<f64>`, `pot`, a live-player bitmask and the fold winner. The tree is built from a `BuildState` through `legal_actions_of` and `next_state_of`, which also serve as the size estimator. |
| Starting stacks | One scalar `stack: f64` (see above). Stack checks use `cfg.stack` everywhere, e.g. `min_left = cfg.stack - invested[i] + ante`. |
| Six different stacks? | **No.** Only the separate 3-player *postflop* module (`multiway/tree.rs`, `stacks: [f64; 3]`) has per-player stacks and side-pot tests. |
| Legal bets, raises, all-ins | Opens come from `open_raises`; re-raises are `max(to_call·mult, to_call + last_raise)`. A raise TO at or above `allin_threshold · stack` becomes a jam to `cfg.stack`. A call sets `invested = to_call` (no all-in call for less), and all-in is detected as `to_call == stack`. There are no short-all-in reopening rules (with equal stacks, none are needed). |
| Terminal utilities | **Fold win:** `pot − rake − invested` (rake only when no-flop-no-drop is off, and only on the matched pot). **Pot share:** one pot, `pot − rake_of(pot)`, divided by hand-class equity. The CUDA kernel (`preflop/kernels.cu`) also assumes a single pot share. |
| Multiway pots and side pots | Multiway leaves use `coupled_deck_v1`: 1,024 deterministic samples of coupled latent hand strengths over **one** pot. The documentation states these are "not jointly dealt physical hole cards" and that card removal is approximate. There are no side pots preflop. |
| Rake | `rake_pct` with a cap in bb, applied to the single pot. The legacy calibrated realization model already embeds a training rake. |
| Continuation values | No postflop solving during a preflop solve. Heads-up leaves use a 169×169 equity table (`preflop_eq169.bin`) times a fitted realization model (`raw`, `static`, `calibrated` or `balanced`). Multiway leaves use the coupled-deck showdown expectation. |
| Python integration | Possible only over HTTP or a subprocess, at the cost of a Rust (and optionally CUDA) build. |
| License | **None**, so modification and redistribution are not permitted. |

## What the fork changes (`crates/solver/src/preflop/mod.rs`)

All changes are opt-in through a new `PreflopConfig.stacks: Option<Vec<f64>>`. Without it, every code path
is the original one, and GTOpen's own tests still pass: 61 library tests and 56 preflop integration tests.

1. **`legal_actions_asym` and `next_state_asym`.** These port PokerKit's no-limit rules:
   - a call is capped at the caller's stack;
   - jams go to the actor's own stack;
   - the minimum raise is `max(largest full raise, bb) + current bet`;
   - no raise when covered or when every opponent is all-in or folded;
   - an all-in smaller than a full raise does not reopen betting for players who already acted.

   `BuildState` gains `acted`, `raise_amount` and `short_allins` to support this.
2. **`side_pots`.** Refunds the uncalled bet (the largest bet above the second-largest, folded seats
   included), layers pots at every contribution level, and merges adjacent layers with the same eligible
   players (PokerKit's construction).
3. **`side_pot_values`.** Terminals with more than one pot, or with an uncalled bet, value each pot over
   only its eligible players, after that pot's pro-rata share of the whole-hand rake:
   - two players: raw heads-up equity (the `balanced` model's antisymmetric pairwise table);
   - three or more: the coupled-deck model;
   - no positional realization multiplier is applied in side-pot spots.

   Single-pot terminals keep GTOpen's original valuation.
4. **Per-seat stacks elsewhere.** Realization SPR, node views and exports use each seat's own stack.
   Player models (seat profiles) are rejected with per-seat stacks for now. Antes and straddles are also
   rejected.
5. **`preflop-cli`.** A command-line binary that solves a JSON config on the CPU and exports exact
   best-response gaps, every action node's average strategy, and optionally every terminal's pots.
6. **`asym_tests.rs`.** Rust tests for own-stack jams and all-in calls, side-pot chip conservation, and
   exact equal-stack equivalence. Node-by-node checks against PokerKit live in pokerlab
   (`tests/test_gtopen.py`).

## Alternative evaluated: OpenSpiel `universal_poker`

OpenSpiel (Apache-2.0) installs from a macOS wheel (`open_spiel 2.0.2`). Tested in a separate scratch
environment, not a project dependency:

- **Per-player stacks: supported.** `stack=4000 7000 10000` loads, and legal raises run to each player's
  own stack (ACPC game definition).
- **No rake.** The betting abstraction is either the full game or `fcpa` (fold, call, pot, all-in).
- **Cards are dealt explicitly**, so a six-player tabular CFR over the full game is far beyond any free
  compute budget.

OpenSpiel is a usable rules reference but cannot solve this game as configured. PokerKit serves as the
independent rules oracle instead (`tests/test_solver_game.py`, `tests/test_gtopen.py`).

## Results

See [`phase3_report.md`](phase3_report.md).
