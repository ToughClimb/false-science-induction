#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


FALLBACK_INPUT_ROOT = Path(
    "artifacts/results/b82_real_error_audit/mislabeled.samples.identification-master/output"
)
FALLBACK_OUTPUT_ROOT = Path("artifacts/results/b82_real_error_audit")

INPUT_FILES = [
    ("GPL96", "GPL96 all information.csv"),
    ("GPL96.97", "GPL96.97 all information.csv"),
    ("GPL570", "GPL570 all inforamtion.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit documented transcriptomic metadata-error coherence."
    )
    parser.add_argument("--input-root", type=Path, required=False)
    parser.add_argument("--output-root", type=Path, required=False)
    return parser.parse_args()


def resolve_input_root(path: Path | None) -> Path:
    if path is not None:
        return path
    return FALLBACK_INPUT_ROOT


def resolve_output_root(path: Path | None) -> Path:
    if path is not None:
        return path
    return FALLBACK_OUTPUT_ROOT


def binomial_tail_probability(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if n <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0
    log_terms = []
    for i in range(k, n + 1):
        log_prob = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(p)
            + (n - i) * math.log1p(-p)
        )
        log_terms.append(log_prob)
    max_log = max(log_terms)
    total = sum(math.exp(value - max_log) for value in log_terms)
    return float(min(1.0, math.exp(max_log) * total))


def mismatch_direction(annotated: str, predicted: str, is_mismatch: bool) -> str:
    if not is_mismatch:
        return "match"
    if annotated == "female" and predicted == "male":
        return "annotated_female_predicted_male"
    if annotated == "male" and predicted == "female":
        return "annotated_male_predicted_female"
    return "other_disagreement"


def normalize_output_rows(frame: pd.DataFrame, platform: str, source_file: str) -> pd.DataFrame:
    required = ["dataset", "sampleID", "G.check", "G.kmean", "KMvsGEO"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns in {source_file}: {missing}")
    rows = frame.loc[:, required].copy()
    rows.columns = ["dataset", "sample_id", "annotated_gender", "predicted_gender", "km_vs_geo"]
    rows["platform"] = platform
    rows["source_file"] = source_file
    rows["annotated_gender"] = rows["annotated_gender"].astype(str).str.lower()
    rows["predicted_gender"] = rows["predicted_gender"].astype(str).str.lower()
    rows["km_vs_geo"] = rows["km_vs_geo"].astype(str).str.lower()
    rows["is_mismatch"] = rows["km_vs_geo"].eq("disagree")
    rows["mismatch_direction"] = [
        mismatch_direction(annotated, predicted, mismatch)
        for annotated, predicted, mismatch in zip(
            rows["annotated_gender"].tolist(),
            rows["predicted_gender"].tolist(),
            rows["is_mismatch"].tolist(),
            strict=True,
        )
    ]
    return rows[
        [
            "platform",
            "source_file",
            "dataset",
            "sample_id",
            "annotated_gender",
            "predicted_gender",
            "km_vs_geo",
            "is_mismatch",
            "mismatch_direction",
        ]
    ]


def load_released_outputs(input_root: Path) -> pd.DataFrame:
    frames = []
    for platform, filename in INPUT_FILES:
        path = input_root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frames.append(normalize_output_rows(frame, platform=platform, source_file=filename))
    return pd.concat(frames, ignore_index=True)


def summarize_by_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    global_rates = rows.groupby("platform")["is_mismatch"].mean().to_dict()
    records: list[dict[str, Any]] = []
    grouped = rows.groupby(["platform", "dataset"], sort=True)
    for (platform, dataset), group in grouped:
        sample_count = int(len(group))
        mismatch_rows = group[group["is_mismatch"]]
        mismatch_count = int(len(mismatch_rows))
        direction_counts = mismatch_rows["mismatch_direction"].value_counts()
        if mismatch_count > 0:
            dominant_direction = str(direction_counts.index[0])
            dominant_count = int(direction_counts.iloc[0])
            direction_purity = dominant_count / mismatch_count
        else:
            dominant_direction = "none"
            dominant_count = 0
            direction_purity = 0.0
        platform_rate = float(global_rates[platform])
        records.append(
            {
                "platform": platform,
                "dataset": dataset,
                "sample_count": sample_count,
                "mismatch_count": mismatch_count,
                "mismatch_rate": mismatch_count / sample_count if sample_count else 0.0,
                "dominant_direction": dominant_direction,
                "dominant_direction_count": dominant_count,
                "direction_purity": direction_purity,
                "platform_mismatch_rate": platform_rate,
                "binomial_enrichment_p": binomial_tail_probability(
                    mismatch_count,
                    sample_count,
                    platform_rate,
                ),
            }
        )
    summary = pd.DataFrame(records)
    return summary.sort_values(
        ["mismatch_count", "mismatch_rate", "direction_purity", "dataset"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def summarize_overall(
    by_dataset: pd.DataFrame,
    sample_count: int,
    mismatch_count: int,
) -> dict[str, Any]:
    dataset_count = int(len(by_dataset))
    datasets_with_mismatch = int((by_dataset["mismatch_count"] > 0).sum())
    coherent = by_dataset[
        (by_dataset["mismatch_count"] >= 2) & (by_dataset["direction_purity"] >= 0.8)
    ]
    if len(by_dataset) > 0:
        top = by_dataset.iloc[0]
        top_dataset = str(top["dataset"])
        top_mismatch_count = int(top["mismatch_count"])
        top_mismatch_rate = float(top["mismatch_rate"])
        top_direction = str(top["dominant_direction"])
        top_direction_purity = float(top["direction_purity"])
    else:
        top_dataset = ""
        top_mismatch_count = 0
        top_mismatch_rate = 0.0
        top_direction = ""
        top_direction_purity = 0.0
    return {
        "sample_count": int(sample_count),
        "mismatch_count": int(mismatch_count),
        "mismatch_rate": mismatch_count / sample_count if sample_count else 0.0,
        "dataset_count": dataset_count,
        "datasets_with_mismatch": datasets_with_mismatch,
        "dataset_mismatch_fraction": datasets_with_mismatch / dataset_count if dataset_count else 0.0,
        "coherent_dataset_count": int(len(coherent)),
        "top_dataset": top_dataset,
        "top_mismatch_count": top_mismatch_count,
        "top_mismatch_rate": top_mismatch_rate,
        "top_direction": top_direction,
        "top_direction_purity": top_direction_purity,
    }


def write_outputs(rows: pd.DataFrame, by_dataset: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    sample_count = int(len(rows))
    mismatch_count = int(rows["is_mismatch"].sum())
    overall = summarize_overall(
        by_dataset=by_dataset,
        sample_count=sample_count,
        mismatch_count=mismatch_count,
    )
    rows.to_csv(output_root / "b82_real_metadata_error_rows.csv", index=False)
    by_dataset.to_csv(output_root / "b82_real_metadata_error_by_dataset.csv", index=False)
    pd.DataFrame([overall]).to_csv(
        output_root / "b82_real_metadata_error_summary.csv",
        index=False,
    )
    with (output_root / "b82_real_metadata_error_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    input_root = resolve_input_root(args.input_root)
    output_root = resolve_output_root(args.output_root)
    rows = load_released_outputs(input_root)
    by_dataset = summarize_by_dataset(rows)
    write_outputs(rows, by_dataset, output_root)


if __name__ == "__main__":
    main()
