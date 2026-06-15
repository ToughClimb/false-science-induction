#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "threshold_margin",
    "control_modes",
    "target_mode",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "metrics_file",
    "model",
    "candidate_count_column",
    "candidate_target_count_column",
    "batch_count_source",
    "batch_target_count_column",
    "cumulative_target_count_column",
]

REQUIRED_ROUND_COLUMNS = [
    "seed",
    "model",
    "mode",
    "round",
]


def require_string_list(cfg: dict[str, object], key: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{key} must contain only strings")
    return [str(item) for item in value]


def require_float(cfg: dict[str, object], key: str) -> float:
    value = cfg[key]
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def require_datasets(cfg: dict[str, object]) -> list[dict[str, object]]:
    datasets = cfg["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(dataset, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        typed.append(dataset)
    return typed


def load_rounds(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["run_dir"])) / str(dataset["metrics_file"])
    if not path.is_file():
        raise FileNotFoundError(f"round metrics not found: {path}")
    frame = pd.read_csv(path)
    required = REQUIRED_ROUND_COLUMNS + [
        str(dataset["candidate_count_column"]),
        str(dataset["candidate_target_count_column"]),
        str(dataset["batch_target_count_column"]),
        str(dataset["cumulative_target_count_column"]),
    ]
    batch_source = dataset["batch_count_source"]
    if not isinstance(batch_source, dict):
        raise TypeError("batch_count_source must be a JSON object")
    require_keys(batch_source, ["kind"], "batch_count_source")
    kind = str(batch_source["kind"])
    if kind == "column":
        require_keys(batch_source, ["column"], "batch_count_source")
        required.append(str(batch_source["column"]))
    elif kind == "constant":
        require_keys(batch_source, ["value"], "batch_count_source")
        if not isinstance(batch_source["value"], int):
            raise TypeError("batch_count_source.value must be an integer")
    else:
        raise ValueError("batch_count_source.kind must be column or constant")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def batch_count_values(dataset: dict[str, object], subset: pd.DataFrame) -> pd.Series:
    batch_source = dataset["batch_count_source"]
    if not isinstance(batch_source, dict):
        raise TypeError("batch_count_source must be a JSON object")
    kind = str(batch_source["kind"])
    if kind == "column":
        column = str(batch_source["column"])
        return subset[column]
    if kind == "constant":
        value = batch_source["value"]
        if not isinstance(value, int):
            raise TypeError("batch_count_source.value must be an integer")
        return pd.Series([value] * len(subset), index=subset.index)
    raise ValueError("batch_count_source.kind must be column or constant")


def rows_for_dataset(
    dataset: dict[str, object],
    control_modes: list[str],
    target_mode: str,
    threshold_margin: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = load_rounds(dataset)
    model = str(dataset["model"])
    modes = control_modes + [target_mode]
    subset = frame[(frame["model"] == model) & (frame["mode"].isin(modes))].copy()
    if subset.empty:
        raise ValueError(f"no intervention rows for dataset={dataset['name']} model={model}")
    duplicated = subset[["seed", "mode", "round"]].duplicated().any()
    if duplicated:
        raise ValueError(f"duplicate intervention rows for dataset={dataset['name']}")

    candidate_count_column = str(dataset["candidate_count_column"])
    candidate_target_count_column = str(dataset["candidate_target_count_column"])
    batch_target_count_column = str(dataset["batch_target_count_column"])
    cumulative_target_count_column = str(dataset["cumulative_target_count_column"])

    subset["candidate_target_prevalence"] = (
        subset[candidate_target_count_column] / subset[candidate_count_column]
    )
    zero_prevalence = subset["candidate_target_prevalence"] <= 0.0
    if bool(zero_prevalence.any()):
        raise ValueError(f"zero candidate target prevalence for dataset={dataset['name']}")
    subset["batch_count_value"] = batch_count_values(dataset, subset)
    subset["batch_target_fraction"] = subset[batch_target_count_column] / subset["batch_count_value"]
    subset["batch_concentration_ratio"] = (
        subset["batch_target_fraction"] / subset["candidate_target_prevalence"]
    )

    controls_for_threshold = subset[subset["mode"].isin(control_modes)]
    if controls_for_threshold.empty:
        raise ValueError(f"no control rows for dataset={dataset['name']}")
    target_rows = subset[subset["mode"] == target_mode]
    if target_rows.empty:
        raise ValueError(f"no target rows for dataset={dataset['name']}")

    threshold = float(controls_for_threshold["batch_concentration_ratio"].max() + threshold_margin)
    subset["would_quarantine"] = subset["batch_concentration_ratio"] > threshold
    subset["prevented_target_count"] = subset[batch_target_count_column].where(
        subset["would_quarantine"],
        0,
    )

    final_rows = subset.loc[
        subset.groupby(["seed", "mode"])["round"].idxmax()
    ][["seed", "mode", cumulative_target_count_column]]
    final_lookup: dict[tuple[int, str], int] = {}
    for record in final_rows.to_dict("records"):
        final_lookup[(int(record["seed"]), str(record["mode"]))] = int(
            record[cumulative_target_count_column]
        )

    rows: list[dict[str, object]] = []
    for record in subset.sort_values(["mode", "seed", "round"]).to_dict("records"):
        seed = int(record["seed"])
        mode = str(record["mode"])
        observed_final = final_lookup[(seed, mode)]
        prevented_to_date = int(
            subset[
                (subset["seed"] == seed)
                & (subset["mode"] == mode)
                & (subset["round"] <= int(record["round"]))
            ]["prevented_target_count"].sum()
        )
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "run_dir": str(dataset["run_dir"]),
                "model": model,
                "threshold": threshold,
                "mode": mode,
                "seed": seed,
                "round": int(record["round"]),
                "candidate_count": int(record[candidate_count_column]),
                "candidate_target_count": int(record[candidate_target_count_column]),
                "candidate_target_prevalence": float(record["candidate_target_prevalence"]),
                "batch_count": int(record["batch_count_value"]),
                "batch_target_count": int(record[batch_target_count_column]),
                "batch_target_fraction": float(record["batch_target_fraction"]),
                "batch_concentration_ratio": float(record["batch_concentration_ratio"]),
                "would_quarantine": bool(record["would_quarantine"]),
                "prevented_target_count": int(record["prevented_target_count"]),
                "prevented_target_count_to_date": prevented_to_date,
                "observed_final_target_count": observed_final,
                "residual_final_target_count_if_quarantined": int(
                    observed_final - prevented_to_date
                ),
                "is_target_mode": bool(mode == target_mode),
            }
        )

    target_summary = subset[subset["mode"] == target_mode]
    control_summary = subset[subset["mode"].isin(control_modes)]
    final_target = target_summary.loc[
        target_summary.groupby(["seed", "mode"])["round"].idxmax()
    ]
    prevented_by_seed = (
        target_summary.groupby("seed", as_index=False)["prevented_target_count"].sum()
    )
    merged = final_target[["seed", cumulative_target_count_column]].merge(
        prevented_by_seed,
        on="seed",
        how="inner",
    )
    merged["residual"] = (
        merged[cumulative_target_count_column] - merged["prevented_target_count"]
    )
    summary = {
        "dataset": str(dataset["name"]),
        "run_dir": str(dataset["run_dir"]),
        "model": model,
        "threshold": threshold,
        "n_control_rounds": int(len(control_summary)),
        "n_target_rounds": int(len(target_summary)),
        "control_quarantine_rate": float(control_summary["would_quarantine"].mean()),
        "target_quarantine_rate": float(target_summary["would_quarantine"].mean()),
        "target_observed_final_mean": float(merged[cumulative_target_count_column].mean()),
        "target_prevented_mean": float(merged["prevented_target_count"].mean()),
        "target_residual_mean": float(merged["residual"].mean()),
        "target_prevented_fraction_mean": float(
            (merged["prevented_target_count"] / merged[cumulative_target_count_column]).mean()
        ),
        "target_any_quarantine_seeds": int(
            target_summary.groupby("seed")["would_quarantine"].any().sum()
        ),
        "n_target_seeds": int(target_summary["seed"].nunique()),
        "control_max_ratio": float(control_summary["batch_concentration_ratio"].max()),
        "target_mean_ratio": float(target_summary["batch_concentration_ratio"].mean()),
    }
    return rows, summary


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B34 Trace-Concentration Quarantine Intervention Replay",
        "",
        "This analysis asks how much false allocation would be prevented if a closed-loop platform quarantined a monitored target/provenance slice whenever a proposed acquisition batch exceeded the maximum clean/random batch-concentration ratio.",
        "",
        "The replay is an offline policy analysis over completed traces. It is a governance intervention estimate, not a retrained closed-loop defense.",
        "",
        "| Dataset | Control quarantine rate | Target quarantine rate | Observed false allocation | Prevented | Residual | Prevented fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {control_quarantine_rate:.3f} | {target_quarantine_rate:.3f} | {target_observed_final_mean:.2f} | {target_prevented_mean:.2f} | {target_residual_mean:.2f} | {target_prevented_fraction_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: the same trace-concentration signal used for audit can be converted into an actionable quarantine rule when the monitored slice is available. The result is scoped to offline replay over the tested traces.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B34 trace-concentration quarantine intervention.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b34_trace_quarantine_intervention")
    threshold_margin = require_float(cfg, "threshold_margin")
    control_modes = require_string_list(cfg, "control_modes")
    target_mode = str(cfg["target_mode"])
    datasets = require_datasets(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(
            dataset,
            control_modes=control_modes,
            target_mode=target_mode,
            threshold_margin=threshold_margin,
        )
        rows.extend(dataset_rows)
        summaries.append(summary)

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {"rows": rows, "summaries": summaries},
            handle,
            indent=2,
            sort_keys=True,
        )
    write_markdown(output_md, summaries)
    print(output_csv)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
