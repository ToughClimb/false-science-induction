#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from scripts.analyze_b40_blind_axis_triage import axes_for_record  # noqa: E402
from scripts.analyze_b46_all_axis_blind_monitor import require_string_list  # noqa: E402
from scripts.analyze_b49_within_campaign_blind_monitor import (  # noqa: E402
    require_int_list,
)


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "baseline_rounds",
    "evaluation_rounds",
    "quantile",
    "threshold_margin",
    "control_modes_for_evaluation_only",
    "target_mode",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "selected_file",
    "domain",
    "object_column",
    "true_label_column",
    "model",
    "modes",
    "selection_filter_column",
    "selection_filter_value",
    "target_axis",
    "target_axis_aliases",
    "min_axis_count",
    "major_fraction_threshold",
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
        str(dataset["true_label_column"]),
        str(dataset["selection_filter_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    column = str(dataset["selection_filter_column"])
    value = dataset["selection_filter_value"]
    if isinstance(value, bool):
        return frame[frame[column].astype(bool) == value].copy()
    if isinstance(value, int):
        return frame[frame[column].astype(int) == value].copy()
    return frame[frame[column].astype(str) == str(value)].copy()


def trace_axis_rows(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    round_idx: int,
    frame: pd.DataFrame,
    object_column: str,
    true_label_column: str,
    domain: str,
    major_fraction_threshold: float,
    min_axis_count: int,
) -> list[dict[str, object]]:
    selected_count = int(len(frame))
    if selected_count <= 0:
        return []
    selected_true_mean = float(frame[true_label_column].mean())
    counts: dict[str, int] = {}
    label_sums: dict[str, float] = {}
    for record in frame[[object_column, true_label_column]].to_dict("records"):
        for axis in axes_for_record(record[object_column], domain, major_fraction_threshold):
            if axis not in counts:
                counts[axis] = 0
                label_sums[axis] = 0.0
            counts[axis] += 1
            label_sums[axis] += float(record[true_label_column])
    rows: list[dict[str, object]] = []
    for axis, count in counts.items():
        if count < min_axis_count:
            continue
        axis_selected_fraction = float(count / selected_count)
        axis_true_mean = float(label_sums[axis] / count)
        feedback_deficit = float(selected_true_mean - axis_true_mean)
        discordance_score = float(axis_selected_fraction * max(0.0, feedback_deficit))
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "seed": int(seed),
                "round": int(round_idx),
                "axis": axis,
                "selected_count": selected_count,
                "axis_selected_count": int(count),
                "axis_selected_fraction": axis_selected_fraction,
                "selected_true_mean": selected_true_mean,
                "axis_true_mean": axis_true_mean,
                "feedback_deficit": feedback_deficit,
                "discordance_score": discordance_score,
            }
        )
    return rows


def empirical_score_threshold(
    early_axis_rows: pd.DataFrame,
    peer_axis_rows: pd.DataFrame,
    quantile: float,
    margin: float,
) -> float:
    values: list[float] = []
    if not early_axis_rows.empty:
        values.extend(early_axis_rows["discordance_score"].astype(float).tolist())
    if not peer_axis_rows.empty:
        values.extend(peer_axis_rows["discordance_score"].astype(float).tolist())
    if not values:
        return float("inf")
    return float(np.quantile(np.array(values, dtype=float), quantile) + margin)


def annotate_flags(
    axis_frame: pd.DataFrame,
    baseline_rounds: list[int],
    evaluation_rounds: list[int],
    quantile: float,
    margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = axis_frame.groupby(["dataset", "model", "mode", "seed"], sort=True)
    for (_, _, _, _), group in grouped:
        baseline = group[group["round"].isin(baseline_rounds)].copy()
        evaluation = group[group["round"].isin(evaluation_rounds)].copy()
        for row in evaluation.to_dict("records"):
            axis = str(row["axis"])
            early_axis_rows = baseline[baseline["axis"] == axis]
            peer_axis_rows = group[
                (group["round"] == int(row["round"])) & (group["axis"] != axis)
            ]
            threshold = empirical_score_threshold(
                early_axis_rows=early_axis_rows,
                peer_axis_rows=peer_axis_rows,
                quantile=quantile,
                margin=margin,
            )
            annotated = dict(row)
            annotated["calibration"] = "within_campaign_feedback_discordance"
            annotated["threshold_source"] = "early_same_axis_plus_same_round_peer_axes"
            annotated["score_threshold"] = threshold
            annotated["flagged"] = bool(float(row["discordance_score"]) > threshold)
            rows.append(annotated)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "model",
            "mode",
            "seed",
            "round",
            "flagged",
            "discordance_score",
            "axis",
        ],
        ascending=[True, True, True, True, True, False, False, True],
    ).reset_index(drop=True)


def rows_for_dataset(
    dataset: dict[str, object],
    baseline_rounds: list[int],
    evaluation_rounds: list[int],
    quantile: float,
    margin: float,
    control_modes: list[str],
    target_mode: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_name = str(dataset["name"])
    model = str(dataset["model"])
    modes = require_string_list(dataset, "modes", dataset_name)
    object_column = str(dataset["object_column"])
    true_label_column = str(dataset["true_label_column"])
    domain = str(dataset["domain"])
    min_axis_count = int(dataset["min_axis_count"])
    major_fraction_threshold = float(dataset["major_fraction_threshold"])
    target_aliases = set(require_string_list(dataset, "target_axis_aliases", dataset_name))

    selected = load_selected(dataset)
    selected = selected[(selected["model"].astype(str) == model) & (selected["mode"].isin(modes))]
    if selected.empty:
        raise ValueError(f"no selected rows for dataset={dataset_name}")

    axis_rows: list[dict[str, object]] = []
    for (mode, seed, round_idx), group in selected.groupby(["mode", "seed", "round"], sort=True):
        axis_rows.extend(
            trace_axis_rows(
                dataset_name=dataset_name,
                model=model,
                mode=str(mode),
                seed=int(seed),
                round_idx=int(round_idx),
                frame=group,
                object_column=object_column,
                true_label_column=true_label_column,
                domain=domain,
                major_fraction_threshold=major_fraction_threshold,
                min_axis_count=min_axis_count,
            )
        )
    axis_frame = pd.DataFrame(axis_rows)
    if axis_frame.empty:
        raise ValueError(f"no axis rows for dataset={dataset_name}")
    flagged = annotate_flags(
        axis_frame=axis_frame,
        baseline_rounds=baseline_rounds,
        evaluation_rounds=evaluation_rounds,
        quantile=quantile,
        margin=margin,
    )
    if flagged.empty:
        raise ValueError(f"no flagged rows for dataset={dataset_name}")

    trace_rows: list[dict[str, object]] = []
    for (mode, seed, round_idx), group in flagged.groupby(["mode", "seed", "round"], sort=True):
        ranked = group.sort_values(
            ["flagged", "discordance_score", "axis_selected_fraction", "axis"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ranked["within_campaign_rank"] = ranked.index + 1
        alias_rows = ranked[ranked["axis"].isin(target_aliases)]
        if alias_rows.empty:
            target_best_rank = int(len(ranked) + 1)
            target_flagged = False
            target_score = 0.0
        else:
            target_best_rank = int(alias_rows["within_campaign_rank"].min())
            target_flagged = bool(alias_rows["flagged"].any())
            target_score = float(alias_rows["discordance_score"].max())
        top_axis = str(ranked.iloc[0]["axis"]) if len(ranked) else ""
        trace_rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": str(mode),
                "seed": int(seed),
                "round": int(round_idx),
                "axis_count": int(len(ranked)),
                "flagged_axis_count": int(ranked["flagged"].sum()),
                "any_axis_flagged": bool(ranked["flagged"].any()),
                "top_axis": top_axis,
                "target_axis_for_evaluation_only": str(dataset["target_axis"]),
                "target_best_rank": target_best_rank,
                "target_flagged": target_flagged,
                "target_discordance_score": target_score,
            }
        )
    trace_frame = pd.DataFrame(trace_rows)
    controls = trace_frame[trace_frame["mode"].isin(control_modes)]
    targets = trace_frame[trace_frame["mode"] == target_mode]
    if targets.empty:
        raise ValueError(f"no target traces for dataset={dataset_name}")
    summary = {
        "dataset": dataset_name,
        "model": model,
        "calibration": "within_campaign_feedback_discordance",
        "baseline_rounds": baseline_rounds,
        "evaluation_rounds": evaluation_rounds,
        "control_trace_count_for_evaluation_only": int(len(controls)),
        "target_trace_count": int(len(targets)),
        "control_any_axis_flag_rate_for_evaluation_only": float(controls["any_axis_flagged"].mean())
        if len(controls)
        else float("nan"),
        "target_any_axis_flag_rate": float(targets["any_axis_flagged"].mean()),
        "target_axis_flag_rate": float(targets["target_flagged"].mean()),
        "target_axis_top1_rate": float((targets["target_best_rank"] == 1).mean()),
        "target_axis_top2_rate": float((targets["target_best_rank"] <= 2).mean()),
        "control_target_axis_flag_rate_for_evaluation_only": float(
            controls["target_flagged"].mean()
        )
        if len(controls)
        else float("nan"),
        "mean_flagged_axes_in_target": float(targets["flagged_axis_count"].mean()),
    }
    row_records = flagged.to_dict("records")
    summary["trace_rows"] = trace_rows
    return row_records, summary


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B59 Feedback-Discordance Blind Monitor",
        "",
        "Thresholds are derived from each campaign's own early traces and same-round peer-axis scores. Clean/random modes are used only after the fact to estimate false alarms.",
        "",
        "| Dataset | Model | Eval-only control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {model} | {control_any_axis_flag_rate_for_evaluation_only:.3f} | {target_any_axis_flag_rate:.3f} | {target_axis_flag_rate:.3f} | {target_axis_top1_rate:.3f} | {target_axis_top2_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a prospective-style triage test that removes external clean-control calibration but still relies on executed or otherwise trusted true feedback. It is not a complete detector, defense, or record-level correction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B59 feedback-discordance blind monitor.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b59_feedback_discordance_monitor")
    baseline_rounds = require_int_list(cfg, "baseline_rounds", "b59_feedback_discordance_monitor")
    evaluation_rounds = require_int_list(
        cfg,
        "evaluation_rounds",
        "b59_feedback_discordance_monitor",
    )
    quantile_value = cfg["quantile"]
    margin_value = cfg["threshold_margin"]
    if not isinstance(quantile_value, int | float):
        raise TypeError("quantile must be numeric")
    if not isinstance(margin_value, int | float):
        raise TypeError("threshold_margin must be numeric")
    quantile = float(quantile_value)
    margin = float(margin_value)
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    control_modes = require_string_list(
        cfg,
        "control_modes_for_evaluation_only",
        "b59_feedback_discordance_monitor",
    )
    target_mode = str(cfg["target_mode"])
    datasets = require_dataset_list(cfg)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(
            dataset=dataset,
            baseline_rounds=baseline_rounds,
            evaluation_rounds=evaluation_rounds,
            quantile=quantile,
            margin=margin,
            control_modes=control_modes,
            target_mode=target_mode,
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
                "baseline_rounds": baseline_rounds,
                "evaluation_rounds": evaluation_rounds,
                "quantile": quantile,
                "threshold_margin": margin,
                "control_modes_for_evaluation_only": control_modes,
                "target_mode": target_mode,
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
