from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.b53_sample_retrospective_replay import (
    sample_ucb_scores,
    scan_sample_axes,
    select_sample_swap_pairs,
    targeted_recorded_labels,
)


def test_scan_sample_axes_prefers_low_axis_with_high_donor_contrast() -> None:
    frame = pd.DataFrame(
        {
            "seq_id": ["1111", "1112", "2111", "2112", "3111", "3112", "4111", "4112"],
            "t50_mean": [10.0, 11.0, 40.0, 42.0, 80.0, 82.0, 75.0, 77.0],
        }
    )

    scan = scan_sample_axes(
        frame=frame,
        min_target_count=2,
        max_target_prevalence=0.5,
        donor_quantile=0.75,
        axis_kinds=["position"],
    )

    assert scan.iloc[0]["axis"] == "pos0=1"
    assert scan.iloc[0]["target_count"] == 2
    assert scan.iloc[0]["target_donor_contrast"] > 65.0


def test_targeted_recorded_labels_preserves_multiset_and_lifts_targets() -> None:
    true_y = np.array([10.0, 11.0, 80.0, 82.0, 40.0, 42.0])
    history_ids = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    pairs = pd.DataFrame(
        {
            "target_record_id": [0, 1],
            "donor_record_id": [2, 3],
            "target_true_label": [10.0, 11.0],
            "donor_true_label": [80.0, 82.0],
        }
    )

    recorded = targeted_recorded_labels(
        true_y=true_y,
        history_ids=history_ids,
        pairs=pairs,
        mode="targeted_swap",
        seed=7,
    )

    assert np.array_equal(np.sort(recorded), np.sort(true_y[history_ids]))
    assert recorded[:2].tolist() == [80.0, 82.0]
    assert recorded[2:4].tolist() == [10.0, 11.0]


def test_sample_ucb_scores_zeroes_minimum_mean_before_uncertainty_bonus() -> None:
    mean = np.array([10.0, 12.0, 13.0])
    std = np.array([0.1, 0.2, 4.0])

    scores = sample_ucb_scores(mean, std, beta=2.0)

    assert np.allclose(scores, np.array([0.2, 2.4, 11.0]))


def test_select_sample_swap_pairs_uses_low_targets_and_high_donors() -> None:
    frame = pd.DataFrame(
        {
            "record_id": np.arange(6, dtype=int),
            "seq_id": ["1111", "1112", "2111", "2112", "3111", "3112"],
            "t50_mean": [10.0, 12.0, 50.0, 55.0, 90.0, 95.0],
        }
    )
    target_mask = frame["seq_id"].str.startswith("1").to_numpy(dtype=bool)

    pairs = select_sample_swap_pairs(
        frame=frame,
        target_mask=target_mask,
        donor_quantile=0.70,
        swap_count=2,
    )

    assert pairs["target_record_id"].tolist() == [0, 1]
    assert pairs["donor_record_id"].tolist() == [5, 4]
    assert pairs["donor_true_label"].tolist() == [95.0, 90.0]
