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

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "thresholds",
    "control_modes",
    "target_mode",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "metrics_file",
    "model",
    "ratio_column",
    "proposed_target_count_column",
    "executed_target_count_column",
    "prevented_target_count_column",
    "final_proposed_column",
    "final_executed_column",
    "final_prevented_column",
]


def require_float_list(cfg: dict[str, object], key: str) -> list[float]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    floats: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise TypeError(f"{key} must contain only numbers")
        floats.append(float(item))
    return floats


def require_string_list(cfg: dict[str, object], key: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{key} must contain only strings")
    return [str(item) for item in value]


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
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
    required = [
        "seed",
        "model",
        "mode",
        "round",
        str(dataset["ratio_column"]),
        str(dataset["proposed_target_count_column"]),
        str(dataset["executed_target_count_column"]),
        str(dataset["prevented_target_count_column"]),
        str(dataset["final_proposed_column"]),
        str(dataset["final_executed_column"]),
        str(dataset["final_prevented_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def rows_for_dataset(
    dataset: dict[str, object],
    thresholds: list[float],
    control_modes: list[str],
    target_mode: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = load_rounds(dataset)
    model = str(dataset["model"])
    modes = control_modes + [target_mode]
    subset = frame[(frame["model"].astype(str) == model) & (frame["mode"].isin(modes))].copy()
    if subset.empty:
        raise ValueError(f"no ROC rows for dataset={dataset['name']} model={model}")
    ratio_column = str(dataset["ratio_column"])
    proposed_column = str(dataset["proposed_target_count_column"])
    executed_column = str(dataset["executed_target_count_column"])
    prevented_column = str(dataset["prevented_target_count_column"])
    final_proposed_column = str(dataset["final_proposed_column"])
    final_executed_column = str(dataset["final_executed_column"])
    final_prevented_column = str(dataset["final_prevented_column"])
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        temp = subset.copy()
        temp["flagged_at_threshold"] = temp[ratio_column].astype(float) > threshold
        temp["sim_prevented"] = temp[proposed_column].where(temp["flagged_at_threshold"], 0)
        temp["sim_executed"] = temp[proposed_column] - temp["sim_prevented"]
        controls = temp[temp["mode"].isin(control_modes)]
        target = temp[temp["mode"] == target_mode]
        if controls.empty:
            raise ValueError(f"no controls for dataset={dataset['name']}")
        if target.empty:
            raise ValueError(f"no targets for dataset={dataset['name']}")
        proposed_by_seed = target.groupby("seed", as_index=False)[proposed_column].sum()
        prevented_by_seed = target.groupby("seed", as_index=False)["sim_prevented"].sum()
        executed_by_seed = target.groupby("seed", as_index=False)["sim_executed"].sum()
        merged = proposed_by_seed.merge(prevented_by_seed, on="seed", how="inner")
        merged = merged.merge(executed_by_seed, on="seed", how="inner")
        proposed_total = float(merged[proposed_column].mean())
        prevented_total = float(merged["sim_prevented"].mean())
        executed_total = float(merged["sim_executed"].mean())
        prevented_fraction = (
            float((merged["sim_prevented"] / merged[proposed_column]).mean())
            if bool((merged[proposed_column] > 0).all())
            else 0.0
        )
        final_rows = target.loc[target.groupby(["seed", "mode"])["round"].idxmax()]
        observed_proposed = float(final_rows[final_proposed_column].mean())
        observed_executed = float(final_rows[final_executed_column].mean())
        observed_prevented = float(final_rows[final_prevented_column].mean())
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "run_dir": str(dataset["run_dir"]),
                "model": model,
                "threshold": float(threshold),
                "control_rounds": int(len(controls)),
                "target_rounds": int(len(target)),
                "false_positive_rate": float(controls["flagged_at_threshold"].mean()),
                "true_positive_rate": float(target["flagged_at_threshold"].mean()),
                "target_proposed_mean": proposed_total,
                "target_executed_residual_mean": executed_total,
                "target_prevented_mean": prevented_total,
                "target_prevented_fraction": prevented_fraction,
                "observed_online_proposed_mean": observed_proposed,
                "observed_online_executed_mean": observed_executed,
                "observed_online_prevented_mean": observed_prevented,
                "mean_recorded_prevented_per_round": float(temp[prevented_column].mean()),
            }
        )
    summary = select_operating_points(rows, str(dataset["name"]), model)
    return rows, summary


def select_operating_points(
    rows: list[dict[str, object]],
    dataset: str,
    model: str,
) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    zero_fpr = frame[frame["false_positive_rate"] <= 0.0].copy()
    if zero_fpr.empty:
        zero = frame.sort_values(
            ["false_positive_rate", "true_positive_rate"],
            ascending=[True, False],
        ).iloc[0]
    else:
        zero = zero_fpr.sort_values(
            ["true_positive_rate", "target_prevented_fraction", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    frame["balanced_score"] = frame["true_positive_rate"] - frame["false_positive_rate"]
    balanced = frame.sort_values(
        ["balanced_score", "target_prevented_fraction", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return {
        "dataset": dataset,
        "model": model,
        "zero_fpr_threshold": float(zero["threshold"]),
        "zero_fpr_tpr": float(zero["true_positive_rate"]),
        "zero_fpr_prevented_fraction": float(zero["target_prevented_fraction"]),
        "balanced_threshold": float(balanced["threshold"]),
        "balanced_fpr": float(balanced["false_positive_rate"]),
        "balanced_tpr": float(balanced["true_positive_rate"]),
        "balanced_prevented_fraction": float(balanced["target_prevented_fraction"]),
    }


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B45 Quarantine ROC and Threshold Sensitivity",
        "",
        "The online trace-concentration rule is re-evaluated over a threshold sweep. Targeted rows are treated as positives and clean/random rows as controls.",
        "",
        "| Dataset | Model | Zero-FPR threshold | Zero-FPR TPR | Zero-FPR prevented | Balanced threshold | Balanced FPR | Balanced TPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {model} | {zero_fpr_threshold:.3f} | {zero_fpr_tpr:.3f} | {zero_fpr_prevented_fraction:.3f} | {balanced_threshold:.3f} | {balanced_fpr:.3f} | {balanced_tpr:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a sensitivity analysis for a trace stop-loss rule, not a claim of complete detection. It shows how much false allocation can be prevented as the allowed control false-positive rate changes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B45 quarantine ROC.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b45_quarantine_roc")
    thresholds = require_float_list(cfg, "thresholds")
    control_modes = require_string_list(cfg, "control_modes")
    target_mode = str(cfg["target_mode"])
    datasets = require_dataset_list(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(
            dataset,
            thresholds=thresholds,
            control_modes=control_modes,
            target_mode=target_mode,
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
    output_json.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "thresholds": thresholds,
                "control_modes": control_modes,
                "target_mode": target_mode,
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_markdown(output_md, summaries)
    print(output_csv)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
