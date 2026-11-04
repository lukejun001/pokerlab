"""Preflop strategy network: any stack vector, one forward pass.

Trained on GTOpen solves of random stack vectors (``tools/preflop_data.py``),
the network maps a betting state (per-seat stacks, bets, who is live and to
act, pot, price, raise count, last aggressor, ...) and a hand class to a
distribution over the GTOpen abstraction's action slots: fold, check/call,
the sized raise, all-in. Illegal slots are masked.

Architecture (bilinear, cheap on CPU): a state MLP emits one vector per
action slot; a learned embedding per hand class is dotted with each, plus
per-(hand, slot) biases. With ``anchor`` inputs (the residual variant) the
log of a borrowed strategy is added to the logits, so the network learns a
correction to it.

Whether it is any good is measured, not assumed: ``profile_for`` builds the
network's strategy for every node of a game's tree, and GTOpen's evaluate
mode scores its exact NashConv.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pokerlab.solver.game import BettingState, PreflopGame, action_amount, action_kind

SLOTS = ("f", "p", "r", "a")  # fold, passive (check or call), sized raise, all-in
SLOT_OF = {"f": 0, "x": 1, "c": 1, "r": 2, "a": 3}
MAX_SEATS = 6
NUM_CLASSES = 169


def slot(token: str) -> int:
    return SLOT_OF[token[0]]


class StateCache:
    """Betting states of one game by history, built incrementally from parents."""

    def __init__(self, game: PreflopGame) -> None:
        self.game = game
        self.states: dict[str, tuple[BettingState, int]] = {"": (BettingState.initial(game), -1)}

    def get(self, history: str) -> tuple[BettingState, int]:
        """(state, last aggressor seat or -1)."""
        found = self.states.get(history)
        if found is not None:
            return found
        parent, _, token = history.rpartition("-")
        state, aggressor = self.get(parent)
        actor = state.actor
        child = state.copy()
        child.apply(action_kind(token), action_amount(token))
        out = (child, actor if token[0] in "ra" else aggressor)
        self.states[history] = out
        return out


def features(game: PreflopGame, state: BettingState, aggressor: int, actions: list[str]) -> np.ndarray:
    """Fixed-length description of a decision (seats padded to six, in preflop order)."""
    bb = game.big_blind
    n = game.n
    actor = state.actor
    start = np.zeros(MAX_SEATS)
    bets = np.zeros(MAX_SEATS)
    behind = np.zeros(MAX_SEATS)
    live = np.zeros(MAX_SEATS)
    acted = np.zeros(MAX_SEATS)
    to_act = np.zeros(MAX_SEATS)
    present = np.zeros(MAX_SEATS)
    eff = np.zeros(MAX_SEATS)
    eff_left = np.zeros(MAX_SEATS)
    a_left = state.stacks[actor] / bb
    for i in range(n):
        start[i] = math.log(game.stacks[i] / bb) / 5
        bets[i] = math.log1p(state.bets[i] / bb) / 5
        behind[i] = math.log1p(state.stacks[i] / bb) / 5
        live[i] = float(state.live[i])
        acted[i] = float(i in state.acted)
        to_act[i] = float(i in state.queue)
        present[i] = 1.0
        if i != actor and state.live[i]:
            eff[i] = math.log(min(game.stacks[i], game.stacks[actor]) / bb) / 5
            eff_left[i] = math.log1p(min(state.stacks[i] + state.bets[i] - state.bets[actor], a_left)
                                     if state.stacks[i] + state.bets[i] > state.bets[actor] else 0.0) / 5
    one_actor = np.zeros(MAX_SEATS)
    one_actor[actor] = 1.0
    one_aggr = np.zeros(MAX_SEATS)
    if aggressor >= 0:
        one_aggr[aggressor] = 1.0
    pot = sum(state.bets) / bb
    owed = (max(state.bets) - state.bets[actor]) / bb
    raises = np.zeros(4)
    raises[min(state.raises, 3)] = 1.0
    sized = [action_amount(t) for t in actions if t[0] == "r"]
    raise_to = math.log1p(sized[0] / bb) / 5 if sized else 0.0
    glob = np.array([
        math.log1p(pot) / 5, math.log1p(owed) / 5, owed / (pot + owed) if pot + owed else 0.0,
        state.calls / 5, sum(state.live) / MAX_SEATS, len(state.queue) / MAX_SEATS,
        raise_to, float(bool(sized)), float(any(t[0] == "a" for t in actions)),
        math.log1p(a_left) / 5, a_left / (pot + owed) / 10 if pot + owed else 0.0, n / MAX_SEATS,
        (game.small_blind / bb), game.rake.percentage * 10,
        math.log1p((game.rake.cap or 0) / bb) / 5,
    ])
    return np.concatenate([start, bets, behind, live, acted, to_act, present, eff, eff_left,
                           one_actor, one_aggr, raises, glob]).astype(np.float32)


NUM_FEATURES = 11 * MAX_SEATS + 4 + 15


def mask_of(actions: list[str]) -> np.ndarray:
    m = np.zeros(len(SLOTS), dtype=bool)
    for t in actions:
        m[slot(t)] = True
    return m


def encode_nodes(game: PreflopGame, histories: list[str], actions: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    cache = StateCache(game)
    feats = np.zeros((len(histories), NUM_FEATURES), dtype=np.float32)
    masks = np.zeros((len(histories), len(SLOTS)), dtype=bool)
    for k, (h, acts) in enumerate(zip(histories, actions)):
        state, aggressor = cache.get(h)
        feats[k] = features(game, state, aggressor, acts)
        masks[k] = mask_of(acts)
    return feats, masks


def slot_table(actions: list[str], table: np.ndarray) -> np.ndarray:
    """(actions x 169) -> (169 x slots), zeros on absent slots."""
    out = np.zeros((table.shape[1], len(SLOTS)), dtype=np.float32)
    for k, t in enumerate(actions):
        out[:, slot(t)] += table[k]
    return out


# --- the model ------------------------------------------------------------------------------------


def build_model(hidden: int = 256, dim: int = 64, layers: int = 3, residual: bool = False, arch: str = "bilinear"):
    import torch
    from torch import nn

    class PolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mods, width = [], NUM_FEATURES
            for _ in range(layers):
                mods += [nn.Linear(width, hidden), nn.SiLU()]
                width = hidden
            self.trunk = nn.Sequential(*mods)
            self.arch = arch
            if arch == "bilinear":
                self.head = nn.Linear(hidden, len(SLOTS) * (dim + 1))
            else:  # "mlp": state and hand vectors meet in a hidden layer, then a per-(state, hand) MLP
                self.state_in = nn.Linear(hidden, dim)
                self.hand_in = nn.Linear(dim, dim, bias=False)
                self.joint = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, len(SLOTS)))
            self.hand = nn.Embedding(NUM_CLASSES, dim)
            self.hand_bias = nn.Parameter(torch.zeros(NUM_CLASSES, len(SLOTS)))
            self.residual = residual
            self.config = {"hidden": hidden, "dim": dim, "layers": layers, "residual": residual, "arch": arch}

        def forward(self, x, mask, anchor=None):
            b = x.shape[0]
            t = self.trunk(x)
            if self.arch == "bilinear":
                z = self.head(t).view(b, len(SLOTS), -1)  # (B, S, dim + 1)
                w, bias = z[..., :-1], z[..., -1]
                logits = torch.einsum("bsd,hd->bhs", w, self.hand.weight) + bias[:, None, :] + self.hand_bias
            else:
                h = self.state_in(t)[:, None, :] + self.hand_in(self.hand.weight)[None, :, :]  # (B, 169, dim)
                logits = self.joint(h) + self.hand_bias
            if self.residual and anchor is not None:
                logits = logits + torch.log(anchor.clamp_min(1e-4))
            return logits.masked_fill(~mask[:, None, :], -1e9)

    return PolicyNet()


def save_model(model, path, extra: dict[str, Any] | None = None) -> None:
    import torch

    torch.save({"config": model.config, "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path):
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(**blob["config"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob.get("extra", {})


def uniform_anchor(n: int = 1):
    import torch

    return torch.full((n, NUM_CLASSES, len(SLOTS)), 0.25)


def profile_for(model, game: PreflopGame, tree: dict[str, list[str]],
                anchors: dict[str, np.ndarray] | None = None, batch: int = 4096) -> dict[str, np.ndarray]:
    """The network's strategy at every node of ``tree``: history -> (actions x 169), pokerlab class order."""
    import torch

    histories = list(tree)
    feats, masks = encode_nodes(game, histories, [tree[h] for h in histories])
    anchor = None
    if anchors is None and getattr(model, "residual", False):
        anchors = {}  # a residual model always needs an anchor: uniform where none is given
    if anchors is not None:
        anchor = np.full((len(histories), NUM_CLASSES, len(SLOTS)), 0.25, dtype=np.float32)
        for k, h in enumerate(histories):
            if h in anchors:
                anchor[k] = slot_table(tree[h], anchors[h])
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for s in range(0, len(histories), batch):
            a = torch.from_numpy(anchor[s:s + batch]) if anchor is not None else None
            p = torch.softmax(model(torch.from_numpy(feats[s:s + batch]), torch.from_numpy(masks[s:s + batch]), a),
                              dim=-1).numpy()
            for k in range(p.shape[0]):
                h = histories[s + k]
                out[h] = np.stack([p[k, :, slot(t)] for t in tree[h]])
    return out
