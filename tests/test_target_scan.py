import pandas as pd

from false_science.target_scan import (
    TargetScanConfig,
    label_multiset_preserved,
    scan_target_regions,
    select_swap_pairs,
)


def synthetic_gfp_like_frame() -> pd.DataFrame:
    rows = []
    for idx in range(40):
        rows.append({"mutant": f"A10V:K{idx + 20}R", "DMS_score": 1.0 + idx * 0.01})
    for idx in range(40):
        rows.append({"mutant": f"G30D:V{idx + 70}A", "DMS_score": 4.0 + idx * 0.01})
    for idx, row in enumerate(rows):
        row["record_id"] = idx
    return pd.DataFrame(rows)


def test_scan_finds_low_target_with_donors() -> None:
    df = synthetic_gfp_like_frame()
    cfg = TargetScanConfig(
        data_path="synthetic.csv",
        target_column="DMS_score",
        mutant_column="mutant",
        max_rows=None,
        random_state=0,
        min_target_count=10,
        min_target_prevalence=0.10,
        max_target_prevalence=0.60,
        target_mean_quantile=0.50,
        donor_quantile=0.80,
        min_swap_count=5,
        max_targets=50,
        tag_prefixes=("pos=", "change=", "group=", "n_mut_bin="),
    )
    scan, tag_sets = scan_target_regions(df, cfg)
    passing = scan[scan["passes_m0_gate"]]
    assert not passing.empty
    assert "change=A10V" in set(passing["tag"])

    pairs = select_swap_pairs(df, tag_sets, "change=A10V", cfg, swap_count=5)
    assert len(pairs) == 5
    assert label_multiset_preserved(pairs)
    assert (pairs["target_true_label"] < pairs["donor_true_label"]).all()


def test_random_high_frequency_bin_can_be_filtered_by_prevalence() -> None:
    df = synthetic_gfp_like_frame()
    cfg = TargetScanConfig(
        data_path="synthetic.csv",
        target_column="DMS_score",
        mutant_column="mutant",
        max_rows=None,
        random_state=0,
        min_target_count=10,
        min_target_prevalence=0.10,
        max_target_prevalence=0.40,
        target_mean_quantile=0.50,
        donor_quantile=0.80,
        min_swap_count=5,
        max_targets=50,
        tag_prefixes=("pos=", "change=", "group=", "n_mut_bin="),
    )
    scan, _ = scan_target_regions(df, cfg)
    assert "n_mut_bin=2" not in set(scan["tag"])


def test_target_row_can_exist_without_passing_gate() -> None:
    df = synthetic_gfp_like_frame()
    cfg = TargetScanConfig(
        data_path="synthetic.csv",
        target_column="DMS_score",
        mutant_column="mutant",
        max_rows=None,
        random_state=0,
        min_target_count=10,
        min_target_prevalence=0.10,
        max_target_prevalence=0.60,
        target_mean_quantile=0.10,
        donor_quantile=0.80,
        min_swap_count=5,
        max_targets=50,
        tag_prefixes=("change=",),
    )
    scan, _ = scan_target_regions(df, cfg)
    row = scan[scan["tag"] == "change=A10V"]
    assert not row.empty
    assert not bool(row.iloc[0]["passes_m0_gate"])
