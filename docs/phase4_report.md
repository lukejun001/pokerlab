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

Learning curve: pending (`tools/preflop_net.py curve`).

## Stage 2: heads-up postflop search

**Solver.** `postflop-cli` (new, in the fork) solves one heads-up spot within an iteration or time
budget and exports every node of the root street. Each node carries the actions, both players'
per-combo weight, reach, equity and EV, and the actor's per-combo strategy.

**Speed on this machine** (2 threads, sharing the CPU with data generation; 1-size trees plus
all-in):

| Root | Tree | Result |
|---|---|---|
| River | 51 nodes | 0.48% of pot almost instantly |
| Turn | 8k nodes | 0.49% of pot in 10.5 s |
| Flop, full tree | 1.6M nodes | 95% of pot after 31 s: not usable in real time |
| Flop, betting on the flop only (check-down after) | 250k nodes | 5.7% of pot in 55 s |

**Agent** (`pokerlab/postflop`, `SolverStrategy(postflop="search")`). Each street is re-solved
once per hand.
- **Flop:** depth-limited. There is no betting after the flop, a crude stand-in for a learned value
  function at the turn. Ranges come from the preflop profile's reach along the line played.
- **Turn and river:** full trees. Ranges are each player's reach at the chance node that closed the
  previous street's line.
- **Opponent sizes:** mapped to the nearest tree size.
- **Fallback:** multiway pots and spots with a player all-in use the heuristic fallback.

Evaluation: pending (`tools/postflop_eval.py`).
