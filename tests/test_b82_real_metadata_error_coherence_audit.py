from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "b82_real_metadata_error_coherence_audit.py"
    spec = importlib.util.spec_from_file_location("b82_real_metadata_error_coherence_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load b82_real_metadata_error_coherence_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_output_rows_identifies_mismatch_direction() -> None:
    module = load_script_module()
    frame = pd.DataFrame(
        {
            "dataset": ["D1", "D1", "D1", "D2"],
            "sampleID": ["s1", "s2", "s3", "s4"],
            "G.check": ["female", "female", "male", "male"],
            "G.kmean": ["male", "male", "male", "female"],
            "KMvsGEO": ["disagree", "disagree", "male", "disagree"],
        }
    )

    rows = module.normalize_output_rows(frame, platform="GPLTEST", source_file="toy.csv")

    assert rows["is_mismatch"].tolist() == [True, True, False, True]
    assert rows["mismatch_direction"].tolist() == [
        "annotated_female_predicted_male",
        "annotated_female_predicted_male",
        "match",
        "annotated_male_predicted_female",
    ]


def test_dataset_summary_reports_direction_purity_and_enrichment() -> None:
    module = load_script_module()
    rows = pd.DataFrame(
        {
            "platform": ["GPLTEST"] * 12,
            "dataset": ["D1"] * 5 + ["D2"] * 7,
            "sampleID": [f"s{i}" for i in range(12)],
            "is_mismatch": [True, True, True, True, False, True, False, False, False, False, False, False],
            "mismatch_direction": [
                "annotated_female_predicted_male",
                "annotated_female_predicted_male",
                "annotated_female_predicted_male",
                "annotated_male_predicted_female",
                "match",
                "annotated_male_predicted_female",
                "match",
                "match",
                "match",
                "match",
                "match",
                "match",
            ],
        }
    )

    summary = module.summarize_by_dataset(rows)
    d1 = summary.loc[summary["dataset"] == "D1"].iloc[0]

    assert int(d1["sample_count"]) == 5
    assert int(d1["mismatch_count"]) == 4
    assert d1["dominant_direction"] == "annotated_female_predicted_male"
    assert float(d1["direction_purity"]) == 0.75
    assert 0.0 <= float(d1["binomial_enrichment_p"]) <= 1.0


def test_overall_summary_counts_coherent_error_surfaces() -> None:
    module = load_script_module()
    by_dataset = pd.DataFrame(
        {
            "dataset": ["D1", "D2", "D3"],
            "sample_count": [10, 10, 10],
            "mismatch_count": [5, 2, 0],
            "mismatch_rate": [0.5, 0.2, 0.0],
            "dominant_direction": [
                "annotated_female_predicted_male",
                "annotated_male_predicted_female",
                "none",
            ],
            "direction_purity": [1.0, 0.5, 0.0],
            "binomial_enrichment_p": [0.01, 0.2, 1.0],
        }
    )

    summary = module.summarize_overall(by_dataset, sample_count=30, mismatch_count=7)

    assert summary["sample_count"] == 30
    assert summary["mismatch_count"] == 7
    assert summary["dataset_count"] == 3
    assert summary["datasets_with_mismatch"] == 2
    assert summary["coherent_dataset_count"] == 1
    assert summary["top_dataset"] == "D1"
