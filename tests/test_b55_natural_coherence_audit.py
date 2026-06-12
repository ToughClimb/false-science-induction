from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_b55_natural_coherence_audit import (
    block_axis_opportunity_rows,
    candidate_axis_opportunity_rows,
    coherence_opportunity_score,
)


def test_coherence_opportunity_score_requires_positive_contrast_and_capacity() -> None:
    positive = coherence_opportunity_score(
        pair_capacity=4,
        target_count=16,
        donor_target_contrast=6.0,
        outcome_scale=3.0,
    )
    negative = coherence_opportunity_score(
        pair_capacity=4,
        target_count=16,
        donor_target_contrast=-6.0,
        outcome_scale=3.0,
    )
    empty = coherence_opportunity_score(
        pair_capacity=0,
        target_count=16,
        donor_target_contrast=6.0,
        outcome_scale=3.0,
    )

    assert np.isclose(positive, 1.0)
    assert negative == 0.0
    assert empty == 0.0


def test_candidate_axis_opportunity_rows_rank_coherent_low_target_axis() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 9.0, 10.0, 11.0],
            "axis_low": [1, 1, 1, 0, 0, 0],
            "axis_high": [0, 0, 0, 1, 1, 1],
        }
    )

    rows = candidate_axis_opportunity_rows(
        frame=frame,
        dataset="toy",
        axis_columns=["axis_low", "axis_high"],
        y_column="y",
        donor_quantile=0.5,
        min_target_count=2,
        max_target_prevalence=0.8,
        min_pair_capacity=2,
        surface_kind="global_axis",
    )

    assert rows.iloc[0]["axis"] == "axis_low"
    assert rows.iloc[0]["pair_capacity"] == 3
    assert rows.iloc[0]["donor_target_contrast"] > 7.0
    assert rows.iloc[0]["opportunity_score"] > rows.iloc[1]["opportunity_score"]


def test_block_axis_opportunity_rows_exposes_block_local_capacity() -> None:
    frame = pd.DataFrame(
        {
            "block": ["a", "a", "a", "b", "b", "b"],
            "y": [1.0, 10.0, 11.0, 2.0, 3.0, 12.0],
            "axis_low": [1, 0, 0, 1, 1, 0],
        }
    )

    rows = block_axis_opportunity_rows(
        frame=frame,
        dataset="toy",
        block_column="block",
        axis_columns=["axis_low"],
        y_column="y",
        donor_quantile=0.5,
        min_pair_capacity=1,
        surface_kind="round_block",
    )

    assert rows["block_id"].tolist() == ["a", "b"]
    assert rows["pair_capacity"].tolist() == [1, 1]
    assert rows["opportunity_score"].min() > 0.0
