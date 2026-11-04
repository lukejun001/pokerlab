# Phase 4: network + search toward any six-handed game (in progress)

Goal: play any six-handed game in real time by combining precomputed solves, a learned network and
real-time search. This phase runs entirely in the 4-core cloud sandbox (Xeon 2.1 GHz, 15 GB RAM, no
GPU), so every design choice below is sized for it.

## Stage 1: preflop strategy network

**Data.** `tools/preflop_data.py generate` solves random 6-max stack vectors with GTOpen:
- stacks log-uniform 20–250 bb per seat, seed 101;
- 100 iterations each, about 1–3 bb/100 exact NashConv;
- 7–11 minutes per solve on 4 niced threads.

Each solve is saved compactly in `data/preflop/` (0.7–1.2 MB) and pushed as it finishes. A file keeps
every node the actor reaches with some hand, because opponents' deviation lines matter to
exploitability. The 16 library solves were converted to the same format (`solve_lib_*`).

**Model** (`pokerlab/solver/network.py`). One forward pass gives the strategy for every hand class at
any betting state.
- **Features:** about 80 per state: per-seat starting stacks, bets, chips behind, live / acted /
  to-act flags, effective stacks against the actor, pot, price, raise count, last aggressor, the
  sized-raise amount, and the blinds and rake.
- **Architecture:** bilinear. A state MLP emits one vector per action slot (fold, check/call, sized
  raise, all-in); learned hand-class embeddings are dotted with each, and illegal slots are masked.
- **Loss:** cross-entropy against the solved strategy, weighted by the actor's reach of the hand ×
  √(opponents' reach).
- **Evaluation:** exact. The network's full profile is scored by GTOpen's evaluate mode on held-out
  vectors (seed 11, the same vectors as the library pilot).

**Because the network reads the betting state, not a tree node,** it also plays lines the tree lacks,
such as an opponent's off-size raise (`network:<model>:off_tree`).

**First result** (trained on the 16 library solves plus 1 generated solve, 30 epochs, 6 held-out
vectors): median **12.1 bb/100**, against **16.3** for k = 1 library borrowing on the same vectors.
Inference takes under 1 second per new stack vector.

**Results** (exact NashConv in bb/100, lower is better; held-out vectors from seed 11; library
borrowing is the Phase 3 k = 1 approximation on the same vectors):

| Model | Training solves | Held-out median | Worst | Beats borrowing | On training games |
|---|---|---|---|---|---|
| Bilinear head, 30 epochs | 8 / 16 / 26 | 26.8 / 18.8 / 24.1 | | | 5–27 |
| Bilinear + anchor correction, 30 epochs | 8 / 16 / 26 | 20.1 / 17.4 / 17.7 | | | |
| Bilinear head, 100 epochs | 26 | 17.4 (10 tests) | 56 | | 3.4 |
| MLP head, 60 epochs | 26 | 11.3 (10 tests) | 23 | 7 of 10 | 3.3 |
| **MLP head, 80 epochs** | **33** | **10.6 (20 tests)** | **28** | **17 of 20** | **3.3** |
| MLP + anchor correction, 80 epochs | 33 | 25.8 (20 tests) | 57 | 6 of 20 | 8.6 |
| Library borrowing (k = 1) | 16 solves | 16.1 (10 tests) / 19.3 (20 tests) | 38 | | |

What the table shows:
- **The head was the bottleneck.** With the bilinear head, each hand's logits are a single dot product
  with the state vector. That underfit even the training games (5–27 bb/100, against 0.3–1.1 for the
  solves themselves). The MLP head (state and hand vectors meet in a hidden layer) generalizes far
  better.
- **The direct network now beats borrowing** on 17 of 20 held-out vectors, at about half the median
  exploitability. It is `models/preflop_net.pt`.
- **Learning a correction to the nearest library strategy did worse.** The anchors are themselves about
  19 bb/100 off, and off-tree or other-family nodes get a uniform anchor. Correcting large,
  inconsistent errors seems harder than learning the strategy directly.
- **The fitting gap remains.** Training games sit at 3.3 bb/100 against about 0.5 for the solves, so
  more capacity, more epochs and more data are all still open. Generation continues at about 5 solves
  an hour.

## Stage 2: heads-up postflop search

**Solver.** `postflop-cli` (new, in the fork) solves one heads-up spot within an iteration or time
budget and exports every node of the root street. Each node carries the actions, both players'
per-combo weight, reach, equity and EV, and the actor's per-combo strategy.

**Depth-limited flop (new in the fork).** `Spot::new_depth_limited` ends betting with the root street.
Each line that closes it ends at an equity leaf: per-pair win/tie/lose fractions over all remaining
runouts, computed once per spot from the river table.
- **Exactness.** It matches the full tree with no later betting: strategies within 2e-3 and EVs within
  1e-2 (`crates/solver/tests/equity_leaf.rs`, flop and turn roots, with rake).
- **Speed.** About 130 times faster: 60 flop iterations take 0.11 s instead of 14.6 s.
- **What the leaf is.** It values a line as checked down, ignoring later betting. This is the slot a
  learned value function would fill.

**Speed on this machine** (2 threads, sharing the CPU with data generation):

| Root | Tree | Result |
|---|---|---|
| River, full | 51 nodes | 0.48% of pot almost instantly |
| Turn, full | 8k nodes | 0.49% of pot in 10.5 s |
| Flop, full three-street tree | 1.6M nodes | 95% of pot after 31 s: not usable in real time |
| Flop, betting on the flop only, all runouts in the tree | 250k nodes | 5.7% of pot in 55 s |
| Flop, depth-limited (33% and 75% bets, 60% raise, all-in) | ~50 nodes | 0.3% of pot in 0.1–9 s |

**Evaluation** (`tools/postflop_eval.py --hands 300`, `reports/phase4/postflop_main.json`).
- **Setup:** the same seeded 6-max table (hero against TAG, LAG, CallingStation, Nit and Balanced
  bots) at held-out stacks 63/82/206/65/72/88 bb.
- **Preflop:** the network in both runs.
- **Budgets:** flop 10 s, turn 15 s, river 5 s, on 2 threads.

| Hero postflop | bb/100 (± s.e.) | Postflop decisions |
|---|---|---|
| Search | +240 ± 97 | 126 by search (53 flop, 40 turn, 33 river); 42 multiway and 8 off-tree spots fell back |
| Heuristic fallback | +29 ± 109 | 216 heuristic |

- **Win rate.** The difference is about 1.4 standard errors, so 300 hands cannot separate the two. It
  is a sanity check (search plays sensibly and makes no invalid actions), not a measured edge.

Search quality per street (exploitability in each solve's own tree):

| Street | Solves | Median | 90th percentile | Median time |
|---|---|---|---|---|
| Flop (depth-limited) | 50 | 0.28% of pot | 0.99% | 6.6 s |
| Turn | 37 | 0.30% | 2.4% | 13.3 s |
| River | 31 | 0.27% | 0.29% | 0.05 s |

**What stage 2 still lacks:**
1. **A learned value function at the flop leaf.** Its output depends on both ranges at the leaf, which
   change every CFR iteration, so the engine must call the network inside each iteration (DeepStack's
   design). Training it needs tens of thousands of turn solves.
2. **Multiway pots.** 42 of the 168 postflop decisions here were multiway and fell back.
3. **Off-tree lines.** A few spots mapped an opponent's line to a node that doesn't match the game
   state.
