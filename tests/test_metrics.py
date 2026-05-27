import numpy as np

from false_science.metrics import (
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)


def test_false_association_strength() -> None:
    pred = np.array([10.0, 9.0, 1.0, 2.0])
    target = np.array([True, True, False, False])
    candidate = np.array([True, True, True, True])
    controls = np.array([2, 3])
    assert false_association_strength(pred, target, controls, candidate) == 8.0


def test_target_topk_fraction_and_rank() -> None:
    pred = np.array([10.0, 9.0, 1.0, 2.0])
    target = np.array([True, True, False, False])
    candidate = np.array([True, True, True, True])
    assert target_topk_fraction(pred, target, candidate, k=2) == 1.0
    assert target_mean_rank_percentile(pred, target, candidate) > 0.8

