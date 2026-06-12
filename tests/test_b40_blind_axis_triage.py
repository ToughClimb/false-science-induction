import pandas as pd

from scripts.analyze_b40_blind_axis_triage import (
    blind_axis_rows_for_subset,
    cameo_region_axes,
    summarize_group,
)


def test_cameo_region_axes():
    assert cameo_region_axes(2.0) == ["dft_region=2"]


def test_blind_axis_scoring_does_not_need_target_axis_input():
    subset = pd.DataFrame(
        {
            "dft_region": [2, 2, 1, 1],
            "true_label": [0.0, 0.0, 5.0, 5.0],
        }
    )

    rows = blind_axis_rows_for_subset(
        dataset_name="toy",
        model="rf",
        mode="targeted_swap",
        seed_text="0",
        scope="seed",
        subset=subset,
        object_column="dft_region",
        true_label_column="true_label",
        domain="cameo_region",
        major_fraction_threshold=0.25,
        min_axis_count=1,
    )

    assert rows[0]["axis"] == "dft_region=2"
    assert rows[0]["conflict_score"] > rows[1]["conflict_score"]


def test_b40_summary_uses_target_only_for_evaluation():
    axis_rows = [
        {
            "dataset": "toy",
            "model": "rf",
            "mode": "targeted_swap",
            "scope": "seed",
            "seed": "0",
            "axis": "dft_region=2",
            "conflict_rank": 1,
            "conflict_score": 0.5,
        },
        {
            "dataset": "toy",
            "model": "rf",
            "mode": "targeted_swap",
            "scope": "aggregate",
            "seed": "all",
            "axis": "dft_region=2",
            "conflict_rank": 1,
            "conflict_score": 0.5,
        },
    ]

    summary = summarize_group(
        dataset_name="toy",
        target_axis="dft_region=2",
        target_axis_aliases=["dft_region=2"],
        model="rf",
        mode="targeted_swap",
        axis_rows=axis_rows,
    )

    assert summary["seed_top1_recovered"] == 1
    assert summary["aggregate_top1_is_target_alias"] is True
