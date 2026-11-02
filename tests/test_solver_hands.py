import numpy as np

from pokerlab.engine.cards import evaluate, parse_cards
from pokerlab.solver.hands import CLASS_COMBOS, CLASS_LABELS, class_of, deal_batch, evaluate_batch


def test_vectorized_evaluator_matches_reference():
    rng = np.random.default_rng(0)
    cards = rng.random((40_000, 52)).argsort(axis=1)[:, :7]
    expected = np.array([evaluate([int(c) for c in row]) for row in cards])
    assert (evaluate_batch(cards) == expected).all()


def test_evaluator_edge_cases():
    hands = ["AsKsQsJsTs9s2d", "AcAdAhAsKcKdQh", "KcKdKhQcQdQh2s", "Ah2c3d4h5s9cTd", "2c3c4c5c7d8hTh"]
    rows = np.array([parse_cards(h) for h in hands])
    assert list(evaluate_batch(rows)) == [evaluate(parse_cards(h)) for h in hands]


def test_classes_and_dealing():
    assert len(CLASS_LABELS) == 169 and CLASS_COMBOS.sum() == 1326
    assert CLASS_LABELS[class_of(parse_cards("AhKh"))] == "AKs"
    holes, classes, ranks = deal_batch(np.random.default_rng(1), 500, 6)
    flat = holes.reshape(500, 12)
    assert all(len(set(row)) == 12 for row in flat.tolist())  # no card dealt twice
    assert classes.shape == ranks.shape == (500, 6)
