# GTOpen compatibility assessment (Phase 3, Milestone 1)

Inspected: <https://github.com/MatthewPDingle/GTOpen> at commit `6c70502` (2 Nov 2026), by reading the
source and documentation. Nothing from GTOpen is copied, vendored or modified in this repository.

## Summary

GTOpen cannot be the base of the asymmetric-stack solver here. There are two independent blockers:

1. **License.** The repository has no license file, and GitHub reports `license: null`. Without a license,
   default copyright applies: there is no permission to modify or redistribute the code. A modified fork
   in this public repository is therefore not an option, however small the patch.
2. **Architecture.** The preflop solver models one shared stack. The code says so explicitly:
   `PreflopConfig.stack: f64`, documented as *"v1: must be equal for all seats — no side pots"*.
   Supporting six different stacks is not a configuration change. It means rewriting action generation,
   terminal payouts, the GPU kernels, the continuation model and the save formats (detailed below).

We therefore used the alternative the brief allows: our own solver (`pokerlab/solver/`). It is validated
node by node against PokerKit, the rules engine the simulator already uses. OpenSpiel was also evaluated;
see the final section.

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

## Changes GTOpen would need for asymmetric stacks

Listed for completeness: these are what a licensed fork would have to change.

1. `PreflopConfig.stack` becomes a per-seat vector, with matching validation, estimator and save-format
   changes (`GTOPREFLOP2`–`4` all assume a scalar).
2. `legal_actions_of`:
   - jam to *each actor's* stack;
   - all-in calls for less than the bet;
   - no raise option when covered or when every opponent is all-in;
   - PokerKit/NLHE short-all-in rules: an incomplete raise does not reopen betting for players who
     already acted.
3. `next_state_of`: per-player all-in and needs-to-act logic, refunds of uncalled excess, and the raise
   increment counted only for full raises.
4. Terminal evaluation (CPU and `kernels.cu`):
   - layered main and side pots, each with its own eligible players;
   - rake per pot under a whole-hand cap;
   - equity of each hand class against *only the players eligible for each pot*.

   This is the deepest change. GTOpen's terminals work on aggregated class distributions (an equity
   table and latent strength samples), which cannot give a separate, exact expected share for each pot
   when three or more players are eligible with different stacks.
5. Per-player SPR and effective stacks in the realization model, contextual keys, player profiles,
   exports (`eff_stack_bb`) and the HTTP and UI layers.

## Alternative evaluated: OpenSpiel `universal_poker`

OpenSpiel (Apache-2.0) installs from a macOS wheel (`open_spiel 2.0.2`). Tested in a separate scratch
environment, not a project dependency:

- **Per-player stacks: supported.** `stack=4000 7000 10000` loads, and legal raises run to each player's
  own stack (ACPC game definition).
- **No rake.** The betting abstraction is either the full game or `fcpa` (fold, call, pot, all-in).
- **Cards are dealt explicitly**, so a six-player tabular CFR over the full game is far beyond any free
  compute budget.

OpenSpiel is a usable rules reference but cannot solve this game as configured. PokerKit serves as the
independent rules oracle instead (`tests/test_solver_game.py`).

## What was built instead

`pokerlab/solver/` contains:

- an asymmetric-stack preflop game model that ports PokerKit's betting and pot-building rules;
- a chance-sampled Discounted CFR solver that deals real hole cards and boards (exact card removal) and
  pays every main and side pot to its eligible best hand after that pot's rake.

Results and limitations are in [`phase3_report.md`](phase3_report.md).
