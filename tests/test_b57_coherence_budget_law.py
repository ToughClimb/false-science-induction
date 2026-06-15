from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_b57_coherence_budget_law import (
    add_family_normalized_columns,
    coherent_lift_from_pairs,
    fit_univariate_least_squares,
    mechanism_risk_score,
)


def test_mechanism_risk_score_uses_coherence_contrast_and_target_size() -> None:
    score = mechanism_risk_score(
        coherence_fraction=0.5,
        donor_target_contrast=4.0,
        outcome_scale=2.0,
        coherent_pair_count=8,
        target_count=32,
    )

    assert np.isclose(score, 0.5)


def test_coherent_lift_from_pairs_uses_only_coherent_pairs_when_available() -> None:
    pairs = pd.DataFrame(
        {
            "pair_source": ["coherent", "random", "coherent"],
            "left_true_label": [1.0, 10.0, 2.0],
            "right_true_label": [6.0, 0.0, 8.0],
        }
    )

    lift = coherent_lift_from_pairs(pairs)

    assert np.isclose(lift, 5.5)


def test_mechanism_predictor_beats_swap_count_when_swap_count_is_fixed() -> None:
    y = np.array([0.0, 0.20, 0.42, 0.63, 0.76])
    swap_count_only = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
    mechanism_score = np.array([0.0, 0.25, 0.50, 0.75, 1.00])

    swap_fit = fit_univariate_least_squares(swap_count_only, y)
    mechanism_fit = fit_univariate_least_squares(mechanism_score, y)

    assert mechanism_fit["r2"] > 0.95
    assert swap_fit["r2"] == 0.0


def test_family_normalization_exposes_within_family_susceptibility() -> None:
    rows = pd.DataFrame(
        {
            "family": ["a", "a", "b", "b"],
            "target_capacity_fraction_excess": [0.0, 0.2, 0.1, 0.4],
            "mechanism_risk_score": [0.0, 2.0, 1.0, 2.0],
        }
    )

    enriched = add_family_normalized_columns(rows)

    assert enriched["family_max_target_capacity_fraction_excess"].tolist() == [
        0.2,
        0.2,
        0.4,
        0.4,
    ]
    assert enriched["family_normalized_excess"].tolist() == [0.0, 1.0, 0.25, 1.0]
    assert enriched["family_susceptibility"].tolist() == [0.0, 0.1, 0.1, 0.2]
