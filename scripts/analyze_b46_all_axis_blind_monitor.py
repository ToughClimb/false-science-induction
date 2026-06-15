#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from scripts.analyze_b40_blind_axis_triage import axes_for_record  # noqa: E402


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
    "selected_file",
    "domain",
    "object_column",
    "model",
    "modes",
    "selection_filter_column",
    "selection_filter_value",
    "target_axis",
    "target_axis_aliases",
    "min_axis_count",
    "major_fraction_threshold",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
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
        require_string_list(dataset, "modes", f"datasets[{index}]")
        require_string_list(dataset, "target_axis_aliases", f"datasets[{index}]")
        if not isinstance(dataset["min_axis_count"], int):
            raise TypeError(f"datasets[{index}].min_axis_count must be an integer")
        if not isinstance(dataset["major_fraction_threshold"], int | float):
            raise TypeError(f"datasets[{index}].major_fraction_threshold must be numeric")
        typed.append(dataset)
    return typed


def load_selected(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["run_dir"])) / str(dataset["selected_file"])
    if not path.is_file():
        raise FileNotFoundError(f"selected records not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "seed",
        "mode",
        "model",
        "round",
        str(dataset["object_column"]),
        str(dataset["selection_filter_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def filtered_selected(dataset: dict[str, object]) -> pd.DataFrame:
    frame = load_selected(dataset)
    column = str(dataset["selection_filter_column"])
    value = dataset["selection_filter_value"]
    if isinstance(value, bool):
        subset = frame[frame[column].astype(bool) == value].copy()
    elif isinstance(value, int):
        subset = frame[frame[column].astype(int) == value].copy()
    else:
        subset = frame[frame[column].astype(str) == str(value)].copy()
    return subset


def axis_rows_for_group(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    selected: pd.DataFrame,
    object_column: str,
    domain: str,
    major_fraction_threshold: float,
    min_axis_count: int,
) -> list[dict[str, object]]:
    total = int(len(selected))
    if total <= 0:
        return []
    counts: dict[str, int] = {}
    for value in selected[object_column].tolist():
        axes = axes_for_record(value, domain, major_fraction_threshold)
        for axis in axes:
            if axis not in counts:
                counts[axis] = 0
            counts[axis] += 1
    rows: list[dict[str, object]] = []
    for axis, count in counts.items():
        if count < min_axis_count:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "seed": int(seed),
                "axis": axis,
                "selected_count": total,
                "axis_selected_count": int(count),
                "axis_selected_fraction": float(count / total),
            }
        )
    return rows


def rows_for_dataset(
    dataset: dict[str, object],
    control_modes: list[str],
    target_mode: str,
    threshold_margin: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_name = str(dataset["name"])
    model = str(dataset["model"])
    modes = require_string_list(dataset, "modes", dataset_name)
    object_column = str(dataset["object_column"])
    domain = str(dataset["domain"])
    min_axis_count = int(dataset["min_axis_count"])
    major_fraction_threshold = float(dataset["major_fraction_threshold"])
    target_aliases = set(require_string_list(dataset, "target_axis_aliases", dataset_name))

    selected = filtered_selected(dataset)
    selected = selected[(selected["model"].astype(str) == model) & (selected["mode"].isin(modes))]
    if selected.empty:
        raise ValueError(f"no selected rows for dataset={dataset_name}")

    axis_rows: list[dict[str, object]] = []
    grouped = selected.groupby(["mode", "seed"], sort=True)
    for (mode, seed), group in grouped:
        axis_rows.extend(
            axis_rows_for_group(
                dataset_name=dataset_name,
                model=model,
                mode=str(mode),
                seed=int(seed),
                selected=group,
                object_column=object_column,
                domain=domain,
                major_fraction_threshold=major_fraction_threshold,
                min_axis_count=min_axis_count,
            )
        )
    axis_frame = pd.DataFrame(axis_rows)
    if axis_frame.empty:
        raise ValueError(f"no axis rows for dataset={dataset_name}")
    control_frame = axis_frame[axis_frame["mode"].isin(control_modes)].copy()
    target_frame = axis_frame[axis_frame["mode"] == target_mode].copy()
    if control_frame.empty:
        raise ValueError(f"no control axis rows for dataset={dataset_name}")
    if target_frame.empty:
        raise ValueError(f"no target axis rows for dataset={dataset_name}")

    thresholds = (
        control_frame.groupby("axis", as_index=False)["axis_selected_fraction"]
        .max()
        .rename(columns={"axis_selected_fraction": "axis_control_threshold"})
    )
    thresholds["axis_control_threshold"] = thresholds["axis_control_threshold"] + threshold_margin
    axis_frame = axis_frame.merge(thresholds, on="axis", how="left")
    axis_frame["axis_control_threshold"] = axis_frame["axis_control_threshold"].fillna(
        threshold_margin
    )
    axis_frame["flagged"] = (
        axis_frame["axis_selected_fraction"] > axis_frame["axis_control_threshold"]
    )
    axis_frame = axis_frame.sort_values(
        ["dataset", "mode", "seed", "flagged", "axis_selected_fraction", "axis"],
        ascending=[True, True, True, False, False, True],
    ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for (mode, seed), group in axis_frame.groupby(["mode", "seed"], sort=True):
        ranked = group.sort_values(
            ["flagged", "axis_selected_fraction", "axis"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranked["blind_rank"] = ranked.index + 1
        alias_rows = ranked[ranked["axis"].isin(target_aliases)]
        if alias_rows.empty:
            target_best_rank = len(ranked) + 1
            target_flagged = False
            target_fraction = 0.0
        else:
            target_best_rank = int(alias_rows["blind_rank"].min())
            target_flagged = bool(alias_rows["flagged"].any())
            target_fraction = float(alias_rows["axis_selected_fraction"].max())
        top_axis = str(ranked.iloc[0]["axis"]) if len(ranked) else ""
        summary_rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": str(mode),
                "seed": int(seed),
                "axis_count": int(len(ranked)),
                "flagged_axis_count": int(ranked["flagged"].sum()),
                "any_axis_flagged": bool(ranked["flagged"].any()),
                "top_axis": top_axis,
                "top_axis_flagged": bool(ranked.iloc[0]["flagged"]) if len(ranked) else False,
                "target_axis_for_evaluation_only": str(dataset["target_axis"]),
                "target_best_rank": int(target_best_rank),
                "target_flagged": target_flagged,
                "target_selected_fraction": target_fraction,
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    controls = summary_frame[summary_frame["mode"].isin(control_modes)]
    targets = summary_frame[summary_frame["mode"] == target_mode]
    summary = {
        "dataset": dataset_name,
        "model": model,
        "control_trace_count": int(len(controls)),
        "target_trace_count": int(len(targets)),
        "control_any_axis_flag_rate": float(controls["any_axis_flagged"].mean()),
        "target_any_axis_flag_rate": float(targets["any_axis_flagged"].mean()),
        "target_axis_flag_rate": float(targets["target_flagged"].mean()),
        "target_axis_top1_rate": float((targets["target_best_rank"] == 1).mean()),
        "target_axis_top2_rate": float((targets["target_best_rank"] <= 2).mean()),
        "control_target_axis_flag_rate": float(controls["target_flagged"].mean()),
        "mean_flagged_axes_in_target": float(targets["flagged_axis_count"].mean()),
    }
    rows = axis_frame.to_dict("records")
    summary["trace_rows"] = summary_rows
    return rows, summary


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B46 All-Axis Blind Trace Monitor",
        "",
        "The monitor scans every enumerable axis in the selected records and calibrates each axis from clean/random controls. The injected target axis is used only after scanning to evaluate recovery.",
        "",
        "| Dataset | Model | Control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {model} | {control_any_axis_flag_rate:.3f} | {target_any_axis_flag_rate:.3f} | {target_axis_flag_rate:.3f} | {target_axis_top1_rate:.3f} | {target_axis_top2_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: all-axis monitoring relaxes the known-slice assumption but exposes the multiple-axis false-positive boundary. It is a blind triage signal, not record-level correction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B46 all-axis blind monitor.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b46_all_axis_blind_monitor")
    control_modes = require_string_list(cfg, "control_modes", "b46_all_axis_blind_monitor")
    target_mode = str(cfg["target_mode"])
    threshold_margin_value = cfg["threshold_margin"]
    if not isinstance(threshold_margin_value, int | float):
        raise TypeError("threshold_margin must be numeric")
    threshold_margin = float(threshold_margin_value)
    datasets = require_dataset_list(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(
            dataset,
            control_modes=control_modes,
            target_mode=target_mode,
            threshold_margin=threshold_margin,
        )
        rows.extend(dataset_rows)
        trace_rows.extend(summary["trace_rows"])
        del summary["trace_rows"]
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
                "control_modes": control_modes,
                "target_mode": target_mode,
                "threshold_margin": threshold_margin,
                "summaries": summaries,
                "trace_rows": trace_rows,
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
