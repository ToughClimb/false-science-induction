#!/usr/bin/env python
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys
from scripts.analyze_b35_hypothesis_axis_recovery import (  # noqa: E402
    materials_composition_axes,
)


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "selected_file",
    "domain",
    "object_column",
    "true_label_column",
    "models",
    "modes",
    "target_axis",
    "target_axis_aliases",
    "min_axis_count",
    "major_fraction_threshold",
]

REQUIRED_SELECTED_COLUMNS = [
    "seed",
    "mode",
    "model",
]

POSITION_PATTERN = re.compile(r"[A-Z](\d+)[A-Z]")


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
        require_string_list(dataset, "models", f"datasets[{index}]")
        require_string_list(dataset, "modes", f"datasets[{index}]")
        require_string_list(dataset, "target_axis_aliases", f"datasets[{index}]")
        if not isinstance(dataset["min_axis_count"], int):
            raise TypeError(f"datasets[{index}].min_axis_count must be an integer")
        if int(dataset["min_axis_count"]) <= 0:
            raise ValueError(f"datasets[{index}].min_axis_count must be positive")
        if not isinstance(dataset["major_fraction_threshold"], int | float):
            raise TypeError(f"datasets[{index}].major_fraction_threshold must be numeric")
        typed.append(dataset)
    return typed


def gfp_position_axes(value: object) -> list[str]:
    positions = sorted({int(item) for item in POSITION_PATTERN.findall(str(value))})
    return [f"pos={position}" for position in positions]


def cameo_region_axes(value: object) -> list[str]:
    return [f"dft_region={int(value)}"]


def axes_for_record(value: object, domain: str, major_fraction_threshold: float) -> list[str]:
    if domain == "gfp_position":
        return gfp_position_axes(value)
    if domain == "materials_composition":
        return materials_composition_axes(value, major_fraction_threshold)
    if domain == "cameo_region":
        return cameo_region_axes(value)
    raise ValueError(f"unknown domain: {domain}")


def load_selected_records(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["run_dir"])) / str(dataset["selected_file"])
    if not path.is_file():
        raise FileNotFoundError(f"selected records not found: {path}")
    frame = pd.read_csv(path)
    required = REQUIRED_SELECTED_COLUMNS + [
        str(dataset["object_column"]),
        str(dataset["true_label_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    if "selection_type" in frame.columns:
        frame = frame[frame["selection_type"] == "executed"].copy()
    return frame


def blind_axis_rows_for_subset(
    dataset_name: str,
    model: str,
    mode: str,
    seed_text: str,
    scope: str,
    subset: pd.DataFrame,
    object_column: str,
    true_label_column: str,
    domain: str,
    major_fraction_threshold: float,
    min_axis_count: int,
) -> list[dict[str, object]]:
    if subset.empty:
        return []
    selected_count = int(len(subset))
    selected_true_mean = float(subset[true_label_column].mean())
    counts: dict[str, int] = {}
    label_sums: dict[str, float] = {}
    for record in subset[[object_column, true_label_column]].to_dict("records"):
        axes = axes_for_record(record[object_column], domain, major_fraction_threshold)
        for axis in axes:
            if axis not in counts:
                counts[axis] = 0
                label_sums[axis] = 0.0
            counts[axis] += 1
            label_sums[axis] += float(record[true_label_column])

    rows: list[dict[str, object]] = []
    for axis, count in counts.items():
        if count < min_axis_count:
            continue
        axis_true_mean = label_sums[axis] / count
        allocation_fraction = count / selected_count
        feedback_deficit = selected_true_mean - axis_true_mean
        conflict_score = allocation_fraction * max(0.0, feedback_deficit)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "scope": scope,
                "seed": seed_text,
                "axis": axis,
                "selected_count": selected_count,
                "axis_selected_count": int(count),
                "axis_selected_fraction": float(allocation_fraction),
                "selected_true_mean": selected_true_mean,
                "axis_true_mean": float(axis_true_mean),
                "feedback_deficit": float(feedback_deficit),
                "conflict_score": float(conflict_score),
            }
        )
    rows.sort(key=lambda item: (-float(item["conflict_score"]), str(item["axis"])))
    for rank, row in enumerate(rows, start=1):
        row["conflict_rank"] = int(rank)
    return rows


def summarize_group(
    dataset_name: str,
    target_axis: str,
    target_axis_aliases: list[str],
    model: str,
    mode: str,
    axis_rows: list[dict[str, object]],
) -> dict[str, object]:
    seed_rows = [
        row
        for row in axis_rows
        if row["dataset"] == dataset_name
        and row["model"] == model
        and row["mode"] == mode
        and row["scope"] == "seed"
    ]
    aggregate_rows = [
        row
        for row in axis_rows
        if row["dataset"] == dataset_name
        and row["model"] == model
        and row["mode"] == mode
        and row["scope"] == "aggregate"
    ]
    seeds = sorted({str(row["seed"]) for row in seed_rows})
    alias_set = set(target_axis_aliases)
    seed_top1_hits = 0
    seed_top2_hits = 0
    seed_best_ranks: list[int] = []
    seed_target_scores: list[float] = []
    for seed in seeds:
        rows_for_seed = [row for row in seed_rows if str(row["seed"]) == seed]
        top_rows = [row for row in rows_for_seed if int(row["conflict_rank"]) == 1]
        if top_rows and str(top_rows[0]["axis"]) in alias_set:
            seed_top1_hits += 1
        alias_rows = [row for row in rows_for_seed if str(row["axis"]) in alias_set]
        if alias_rows:
            best_rank = min(int(row["conflict_rank"]) for row in alias_rows)
            best_score = max(float(row["conflict_score"]) for row in alias_rows)
        else:
            best_rank = len(rows_for_seed) + 1
            best_score = 0.0
        if best_rank <= 2:
            seed_top2_hits += 1
        seed_best_ranks.append(best_rank)
        seed_target_scores.append(best_score)

    if aggregate_rows:
        aggregate_top_axis = str(aggregate_rows[0]["axis"])
        aggregate_alias_rows = [row for row in aggregate_rows if str(row["axis"]) in alias_set]
        if aggregate_alias_rows:
            aggregate_target_best_rank = min(
                int(row["conflict_rank"]) for row in aggregate_alias_rows
            )
            aggregate_target_score = max(
                float(row["conflict_score"]) for row in aggregate_alias_rows
            )
        else:
            aggregate_target_best_rank = len(aggregate_rows) + 1
            aggregate_target_score = 0.0
    else:
        aggregate_top_axis = ""
        aggregate_target_best_rank = 0
        aggregate_target_score = 0.0

    n_seeds = len(seeds)
    return {
        "dataset": dataset_name,
        "model": model,
        "mode": mode,
        "target_axis_for_evaluation_only": target_axis,
        "target_axis_aliases_for_evaluation_only": target_axis_aliases,
        "n_seeds": int(n_seeds),
        "seed_top1_recovered": int(seed_top1_hits),
        "seed_top1_recovery_rate": float(seed_top1_hits / n_seeds) if n_seeds else float("nan"),
        "seed_top2_recovered": int(seed_top2_hits),
        "seed_top2_recovery_rate": float(seed_top2_hits / n_seeds) if n_seeds else float("nan"),
        "seed_target_best_rank_mean": float(sum(seed_best_ranks) / n_seeds)
        if n_seeds
        else float("nan"),
        "seed_target_conflict_score_mean": float(sum(seed_target_scores) / n_seeds)
        if n_seeds
        else float("nan"),
        "aggregate_top_axis": aggregate_top_axis,
        "aggregate_top1_is_target_alias": bool(aggregate_top_axis in alias_set),
        "aggregate_target_best_rank": int(aggregate_target_best_rank),
        "aggregate_target_conflict_score": float(aggregate_target_score),
    }


def rows_for_dataset(dataset: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = load_selected_records(dataset)
    dataset_name = str(dataset["name"])
    object_column = str(dataset["object_column"])
    true_label_column = str(dataset["true_label_column"])
    domain = str(dataset["domain"])
    target_axis = str(dataset["target_axis"])
    target_axis_aliases = require_string_list(dataset, "target_axis_aliases", dataset_name)
    models = require_string_list(dataset, "models", dataset_name)
    modes = require_string_list(dataset, "modes", dataset_name)
    min_axis_count = int(dataset["min_axis_count"])
    major_fraction_threshold = float(dataset["major_fraction_threshold"])

    axis_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for model in models:
        for mode in modes:
            subset = frame[(frame["model"] == model) & (frame["mode"] == mode)].copy()
            if subset.empty:
                raise ValueError(f"no selected rows for dataset={dataset_name} model={model} mode={mode}")
            for seed in sorted(subset["seed"].unique().tolist()):
                seed_subset = subset[subset["seed"] == seed]
                axis_rows.extend(
                    blind_axis_rows_for_subset(
                        dataset_name=dataset_name,
                        model=model,
                        mode=mode,
                        seed_text=str(seed),
                        scope="seed",
                        subset=seed_subset,
                        object_column=object_column,
                        true_label_column=true_label_column,
                        domain=domain,
                        major_fraction_threshold=major_fraction_threshold,
                        min_axis_count=min_axis_count,
                    )
                )
            axis_rows.extend(
                blind_axis_rows_for_subset(
                    dataset_name=dataset_name,
                    model=model,
                    mode=mode,
                    seed_text="all",
                    scope="aggregate",
                    subset=subset,
                    object_column=object_column,
                    true_label_column=true_label_column,
                    domain=domain,
                    major_fraction_threshold=major_fraction_threshold,
                    min_axis_count=min_axis_count,
                )
            )
            summaries.append(
                summarize_group(
                    dataset_name=dataset_name,
                    target_axis=target_axis,
                    target_axis_aliases=target_axis_aliases,
                    model=model,
                    mode=mode,
                    axis_rows=axis_rows,
                )
            )
    return axis_rows, summaries


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B40 Blind Hypothesis-Axis Triage",
        "",
        "Axes are scored without using the injected target. The target-axis aliases are used only after ranking to evaluate whether the blind ranking recovered the implicated axis.",
        "",
        "Score: selected-budget fraction multiplied by positive true-feedback deficit.",
        "",
        "| Dataset | Model | Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {model} | {mode} | {seed_top1_recovered}/{n_seeds} ({seed_top1_recovery_rate:.3f}) | {seed_top2_recovered}/{n_seeds} ({seed_top2_recovery_rate:.3f}) | {aggregate_top_axis} | {aggregate_target_best_rank} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: targeted-trace recovery means the false hypothesis itself acts as a diagnostic probe for the implicated scientific axis. This is axis-level triage, not causal discovery or record-level repair.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B40 blind axis triage.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b40_blind_axis_triage")
    datasets = require_dataset_list(cfg)

    axis_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_axis_rows, dataset_summaries = rows_for_dataset(dataset)
        axis_rows.extend(dataset_axis_rows)
        summaries.extend(dataset_summaries)

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(axis_rows).to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {"axis_rows": axis_rows, "summaries": summaries},
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
