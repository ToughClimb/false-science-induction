from __future__ import annotations

import numpy as np

from scripts.b50_materials_gp_bo_replay import (
    acquisition_scores,
    candidate_pool_ids,
    expected_improvement,
    matched_pool_control_ids,
)


def test_expected_improvement_is_positive_for_high_mean_uncertain_candidate() -> None:
    mean = np.array([0.0, 1.0, 2.0])
    std = np.array([0.0, 0.5, 1.0])

    ei = expected_improvement(mean, std, best_observed=1.0, xi=0.0)

    assert ei[0] == 0.0
    assert ei[1] > 0.0
    assert ei[2] > ei[1]


def test_acquisition_scores_support_ucb_and_ei() -> None:
    mean = np.array([1.0, 2.0])
    std = np.array([0.5, 0.1])

    ucb = acquisition_scores(mean, std, policy="gp_ucb", beta=2.0, best_observed=0.0, xi=0.0)
    ei = acquisition_scores(mean, std, policy="expected_improvement", beta=2.0, best_observed=1.5, xi=0.0)

    assert np.allclose(ucb, np.array([2.0, 2.2]))
    assert ei.shape == mean.shape
    assert (ei >= 0.0).all()


def test_candidate_pool_ids_includes_targets_and_is_deterministic() -> None:
    candidate_mask = np.ones(20, dtype=bool)
    target_mask = np.zeros(20, dtype=bool)
    target_mask[[2, 5, 7]] = True

    first = candidate_pool_ids(
        candidate_mask=candidate_mask,
        target_mask=target_mask,
        pool_size=8,
        seed=11,
    )
    second = candidate_pool_ids(
        candidate_mask=candidate_mask,
        target_mask=target_mask,
        pool_size=8,
        seed=11,
    )

    assert first.tolist() == second.tolist()
    assert {2, 5, 7}.issubset(set(first.tolist()))
    assert len(first) == 8


def test_matched_pool_control_ids_stays_inside_reduced_pool() -> None:
    target_mask = np.zeros(20, dtype=bool)
    target_mask[[2, 5, 7]] = True
    pool_ids = np.array([1, 2, 3, 5, 7, 8], dtype=int)
    n_elements = np.ones(20, dtype=int)

    controls, pool_mask = matched_pool_control_ids(
        target_mask=target_mask,
        pool_ids=pool_ids,
        n_elements=n_elements,
        seed=3,
    )

    assert set(controls.tolist()).issubset(set(pool_ids.tolist()))
    assert not target_mask[controls].any()
    assert np.flatnonzero(pool_mask).tolist() == pool_ids.tolist()
