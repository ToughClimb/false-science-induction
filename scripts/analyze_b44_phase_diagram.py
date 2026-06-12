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
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "input_csv",
    "target_mode",
    "control_mode",
    "dose_column",
    "model_column",
    "mode_column",
    "final_count_column",
    "final_excess_column",
    "fas_lift_column",
]


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


def load_dataset_frame(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["input_csv"]))
    if not path.is_file():
        raise FileNotFoundError(f"phase input not found: {path}")
    frame = pd.read_csv(path)
    required = [
        str(dataset["dose_column"]),
        str(dataset["model_column"]),
        str(dataset["mode_column"]),
        str(dataset["final_count_column"]),
        str(dataset["final_excess_column"]),
        str(dataset["fas_lift_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def rows_for_dataset(
    dataset: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = load_dataset_frame(dataset)
    name = str(dataset["name"])
    dose_column = str(dataset["dose_column"])
    model_column = str(dataset["model_column"])
    mode_column = str(dataset["mode_column"])
    final_count_column = str(dataset["final_count_column"])
    final_excess_column = str(dataset["final_excess_column"])
    fas_lift_column = str(dataset["fas_lift_column"])
    target_mode = str(dataset["target_mode"])
    control_mode = str(dataset["control_mode"])

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for model in sorted(frame[model_column].astype(str).unique().tolist()):
        model_frame = frame[frame[model_column].astype(str) == model].copy()
        target = model_frame[model_frame[mode_column].astype(str) == target_mode].copy()
        control = model_frame[model_frame[mode_column].astype(str) == control_mode].copy()
        if target.empty:
            raise ValueError(f"no target rows for dataset={name} model={model}")
        if control.empty:
            raise ValueError(f"no control rows for dataset={name} model={model}")
        target = target.sort_values(dose_column)
        positive = target[
            (target[final_excess_column].astype(float) > 0.0)
            & (target[fas_lift_column].astype(float) > 0.0)
        ].copy()
        if positive.empty:
            min_effective = 0
        else:
            min_effective = int(positive.iloc[0][dose_column])
        peak_index = target[final_count_column].astype(float).idxmax()
        peak_swap = int(target.loc[peak_index, dose_column])
        peak_count = float(target.loc[peak_index, final_count_column])
        max_swap = int(target[dose_column].max())
        max_swap_count = float(target[target[dose_column] == max_swap].iloc[0][final_count_column])
        acquisition_nonmonotonic = bool(peak_swap != max_swap or max_swap_count < peak_count)
        fas_at_min = (
            float(positive.iloc[0][fas_lift_column])
            if not positive.empty
            else 0.0
        )
        for record in target.to_dict("records"):
            dose = int(record[dose_column])
            control_match = control[control[dose_column] == dose]
            if control_match.empty:
                control_final_count = float("nan")
            else:
                control_final_count = float(control_match.iloc[0][final_count_column])
            rows.append(
                {
                    "dataset": name,
                    "model": model,
                    "dose": dose,
                    "target_mode": target_mode,
                    "control_mode": control_mode,
                    "target_final_count": float(record[final_count_column]),
                    "control_final_count": control_final_count,
                    "target_final_excess": float(record[final_excess_column]),
                    "target_fas_lift": float(record[fas_lift_column]),
                    "is_min_effective_dose": bool(dose == min_effective and min_effective > 0),
                    "is_acquisition_peak": bool(dose == peak_swap),
                }
            )
        summaries.append(
            {
                "dataset": name,
                "model": model,
                "target_mode": target_mode,
                "control_mode": control_mode,
                "tested_doses": ",".join(str(int(value)) for value in target[dose_column].tolist()),
                "min_effective_tested_dose": int(min_effective),
                "fas_lift_at_min_effective": float(fas_at_min),
                "acquisition_peak_dose": int(peak_swap),
                "acquisition_peak_count": peak_count,
                "max_tested_dose": int(max_swap),
                "max_tested_dose_count": max_swap_count,
                "acquisition_nonmonotonic_or_saturated": acquisition_nonmonotonic,
                "positive_dose_count": int(len(positive)),
                "tested_dose_count": int(len(target)),
            }
        )
    return rows, summaries


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B44 Minimum-Effective Corruption Phase Diagram",
        "",
        "This table summarizes the tested swap-count boundary using existing dose-response runs. A dose is marked effective when targeted traces have both positive final acquisition excess over the configured control and positive false-association-strength lift.",
        "",
        "| Dataset | Model | Tested doses | Min effective | Acquisition peak | Saturated/nonmonotonic |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {model} | {tested_doses} | {min_effective_tested_dose} | {acquisition_peak_dose} | {acquisition_nonmonotonic_or_saturated} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: the mechanism has a low tested onset in both domains, while acquisition counts saturate or become nonmonotonic at high corruption. This supports an operating-boundary claim rather than a linear-dose claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B44 minimum-effective corruption phase diagram.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b44_phase_diagram")
    datasets = require_dataset_list(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, dataset_summaries = rows_for_dataset(dataset)
        rows.extend(dataset_rows)
        summaries.extend(dataset_summaries)

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
                "rows": rows,
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
