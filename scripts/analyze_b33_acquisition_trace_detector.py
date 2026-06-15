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
    "detection_round",
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
    "cumulative_selected_count_column",
    "cumulative_target_count_column",
    "cumulative_target_fraction_column",
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


def require_int(cfg: dict[str, object], key: str) -> int:
    value = cfg[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return int(value)


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


def load_dataset_rounds(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["run_dir"])) / str(dataset["metrics_file"])
    if not path.is_file():
        raise FileNotFoundError(f"round metrics not found: {path}")
    frame = pd.read_csv(path)
    required = REQUIRED_ROUND_COLUMNS + [
        str(dataset["candidate_count_column"]),
        str(dataset["candidate_target_count_column"]),
        str(dataset["cumulative_selected_count_column"]),
        str(dataset["cumulative_target_count_column"]),
        str(dataset["cumulative_target_fraction_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def rows_for_dataset(
    dataset: dict[str, object],
    detection_round: int,
    control_modes: list[str],
    target_mode: str,
    threshold_margin: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = load_dataset_rounds(dataset)
    model = str(dataset["model"])
    modes = control_modes + [target_mode]
    subset = frame[
        (frame["model"] == model)
        & (frame["round"] == detection_round)
        & (frame["mode"].isin(modes))
    ].copy()
    if subset.empty:
        raise ValueError(
            f"no detector rows for dataset={dataset['name']} model={model} round={detection_round}"
        )
    duplicated = subset[["seed", "mode"]].duplicated().any()
    if duplicated:
        raise ValueError(f"duplicate detector rows for dataset={dataset['name']}")

    candidate_count_column = str(dataset["candidate_count_column"])
    candidate_target_count_column = str(dataset["candidate_target_count_column"])
    cumulative_selected_count_column = str(dataset["cumulative_selected_count_column"])
    cumulative_target_count_column = str(dataset["cumulative_target_count_column"])
    cumulative_target_fraction_column = str(dataset["cumulative_target_fraction_column"])

    subset["candidate_target_prevalence"] = (
        subset[candidate_target_count_column] / subset[candidate_count_column]
    )
    zero_prevalence = subset["candidate_target_prevalence"] <= 0.0
    if bool(zero_prevalence.any()):
        raise ValueError(f"zero candidate target prevalence for dataset={dataset['name']}")
    subset["concentration_ratio"] = (
        subset[cumulative_target_fraction_column]
        / subset["candidate_target_prevalence"]
    )

    controls_for_threshold = subset[subset["mode"].isin(control_modes)]
    if controls_for_threshold.empty:
        raise ValueError(f"no control rows for dataset={dataset['name']}")
    if subset[subset["mode"] == target_mode].empty:
        raise ValueError(f"no target rows for dataset={dataset['name']}")

    threshold = float(controls_for_threshold["concentration_ratio"].max() + threshold_margin)
    subset["flagged"] = subset["concentration_ratio"] > threshold
    controls = subset[subset["mode"].isin(control_modes)]
    targets = subset[subset["mode"] == target_mode]

    rows: list[dict[str, object]] = []
    for record in subset.sort_values(["mode", "seed"]).to_dict("records"):
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "run_dir": str(dataset["run_dir"]),
                "model": model,
                "detection_round": detection_round,
                "threshold": threshold,
                "mode": str(record["mode"]),
                "seed": int(record["seed"]),
                "candidate_count": int(record[candidate_count_column]),
                "candidate_target_count": int(record[candidate_target_count_column]),
                "candidate_target_prevalence": float(
                    record["candidate_target_prevalence"]
                ),
                "cumulative_selected_count": int(
                    record[cumulative_selected_count_column]
                ),
                "cumulative_target_count": int(record[cumulative_target_count_column]),
                "cumulative_target_fraction": float(
                    record[cumulative_target_fraction_column]
                ),
                "concentration_ratio": float(record["concentration_ratio"]),
                "flagged": bool(record["flagged"]),
                "is_target_mode": bool(record["mode"] == target_mode),
            }
        )

    false_positive_count = int(controls["flagged"].sum())
    true_positive_count = int(targets["flagged"].sum())
    summary = {
        "dataset": str(dataset["name"]),
        "run_dir": str(dataset["run_dir"]),
        "model": model,
        "detection_round": detection_round,
        "threshold": threshold,
        "n_control": int(len(controls)),
        "n_target": int(len(targets)),
        "false_positive_count": false_positive_count,
        "true_positive_count": true_positive_count,
        "false_positive_rate": float(false_positive_count / len(controls)),
        "true_positive_rate": float(true_positive_count / len(targets)),
        "control_ratio_mean": float(controls["concentration_ratio"].mean()),
        "control_ratio_max": float(controls["concentration_ratio"].max()),
        "target_ratio_mean": float(targets["concentration_ratio"].mean()),
        "target_ratio_min": float(targets["concentration_ratio"].min()),
    }
    return rows, summary


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B33 Acquisition-Trace Concentration Detector",
        "",
        "This audit calibrates a per-run threshold from clean and random-swap controls, then tests whether targeted-swap traces concentrate in the monitored target slice above that threshold.",
        "",
        "| Dataset | Round | Controls | Targeted | FPR | TPR | Control max | Target mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {detection_round} | {n_control} | {n_target} | {false_positive_rate:.3f} | {true_positive_rate:.3f} | {control_ratio_max:.3f} | {target_ratio_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a trace-level audit signal, not a complete defense. It assumes that a provenance or scientific slice is being monitored and that controls are available for calibration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B33 acquisition-trace concentration detector.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b33_acquisition_trace_detector")
    detection_round = require_int(cfg, "detection_round")
    threshold_margin = require_float(cfg, "threshold_margin")
    control_modes = require_string_list(cfg, "control_modes")
    target_mode = str(cfg["target_mode"])
    datasets = require_datasets(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(
            dataset,
            detection_round=detection_round,
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
    output_json.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "detection_round": detection_round,
                "threshold_margin": threshold_margin,
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
    print(str(output_csv))
    print(str(output_json))
    print(str(output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
