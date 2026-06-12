#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import math
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

from false_science.cameo import build_cameo_dataset  # noqa: E402
from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.materials import load_matminer_dataset, material_feature_frame  # noqa: E402
from false_science.plot_style import OKABE_ITO, apply_paper_style, style_axis  # noqa: E402
from false_science.target_scan import file_sha256, git_text  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "output_figure_pdf",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "kind",
    "source_note",
    "donor_quantile",
    "min_target_count",
    "max_target_prevalence",
    "min_pair_capacity",
]

REQUIRED_MATERIALS_KEYS = [
    "dataset_name",
    "target_column",
    "composition_column",
    "tag_prefixes",
]

REQUIRED_CAMEO_KEYS = [
    "data_zip",
    "target_column",
    "xrd_pca_components",
    "pca_seed",
]

REQUIRED_SAMPLE_KEYS = [
    "sample_root",
    "source_archive",
]


def parse_args() -> dict[str, object]:
    config_path = parse_config_arg("B55 natural coherence opportunity audit.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b55_natural_coherence_audit")
    datasets = require_dataset_list(cfg)
    cfg["datasets"] = datasets
    return cfg


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    datasets = cfg["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(dataset, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        kind = str(dataset["kind"])
        if kind == "materials_matbench":
            require_keys(dataset, REQUIRED_MATERIALS_KEYS, f"datasets[{index}]")
            require_string_list(dataset, "tag_prefixes", f"datasets[{index}]")
        elif kind == "cameo":
            require_keys(dataset, REQUIRED_CAMEO_KEYS, f"datasets[{index}]")
        elif kind == "sample":
            require_keys(dataset, REQUIRED_SAMPLE_KEYS, f"datasets[{index}]")
        else:
            raise ValueError(f"unsupported dataset kind: {kind}")
        typed.append(dataset)
    return typed


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def outcome_scale(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    q75 = float(np.quantile(y, 0.75))
    q25 = float(np.quantile(y, 0.25))
    iqr = q75 - q25
    if iqr > 1e-12:
        return iqr
    std = float(np.std(y))
    if std > 1e-12:
        return std
    return 1.0


def coherence_opportunity_score(
    pair_capacity: int,
    target_count: int,
    donor_target_contrast: float,
    outcome_scale: float,
) -> float:
    if pair_capacity <= 0 or target_count <= 0:
        return 0.0
    if donor_target_contrast <= 0.0:
        return 0.0
    if outcome_scale <= 0.0:
        raise ValueError("outcome_scale must be positive")
    return float(donor_target_contrast / outcome_scale) * math.sqrt(
        float(pair_capacity) / float(target_count)
    )


def candidate_axis_opportunity_rows(
    frame: pd.DataFrame,
    dataset: str,
    axis_columns: list[str],
    y_column: str,
    donor_quantile: float,
    min_target_count: int,
    max_target_prevalence: float,
    min_pair_capacity: int,
    surface_kind: str,
) -> pd.DataFrame:
    y = frame[y_column].to_numpy(dtype=float)
    donor_cutoff = float(np.quantile(y, float(donor_quantile)))
    scale = outcome_scale(y)
    rows: list[dict[str, object]] = []
    for axis in axis_columns:
        mask = frame[axis].astype(bool).to_numpy()
        target_count = int(mask.sum())
        if target_count == 0:
            continue
        target_prevalence = float(target_count / len(frame))
        if target_count < int(min_target_count) or target_prevalence > float(max_target_prevalence):
            continue
        donor_mask = (~mask) & (y >= donor_cutoff)
        donor_count = int(donor_mask.sum())
        target_mean = float(np.mean(y[mask]))
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        contrast = donor_mean - target_mean if donor_count else float("nan")
        pair_capacity = int(min(target_count, donor_count))
        score = coherence_opportunity_score(
            pair_capacity=pair_capacity,
            target_count=target_count,
            donor_target_contrast=contrast if donor_count else 0.0,
            outcome_scale=scale,
        )
        rows.append(
            {
                "dataset": dataset,
                "surface_kind": surface_kind,
                "block_column": "none",
                "block_id": "all",
                "axis": axis,
                "n_records": int(len(frame)),
                "target_count": target_count,
                "target_prevalence": target_prevalence,
                "donor_count": donor_count,
                "pair_capacity": pair_capacity,
                "target_mean": target_mean,
                "donor_mean": donor_mean,
                "donor_target_contrast": contrast,
                "outcome_scale": scale,
                "opportunity_score": score,
                "passes_opportunity_gate": bool(pair_capacity >= int(min_pair_capacity) and score > 0.0),
                "label_multiset_preserving_swap_possible": bool(pair_capacity > 0),
            }
        )
    return sort_opportunity_rows(pd.DataFrame(rows))


def block_axis_opportunity_rows(
    frame: pd.DataFrame,
    dataset: str,
    block_column: str,
    axis_columns: list[str],
    y_column: str,
    donor_quantile: float,
    min_pair_capacity: int,
    surface_kind: str,
) -> pd.DataFrame:
    y_all = frame[y_column].to_numpy(dtype=float)
    global_cutoff = float(np.quantile(y_all, float(donor_quantile)))
    scale = outcome_scale(y_all)
    rows: list[dict[str, object]] = []
    for block_id in sorted(frame[block_column].astype(str).unique().tolist()):
        block = frame[frame[block_column].astype(str) == block_id].copy()
        if block.empty:
            continue
        y = block[y_column].to_numpy(dtype=float)
        for axis in axis_columns:
            mask = block[axis].astype(bool).to_numpy()
            target_count = int(mask.sum())
            if target_count == 0:
                continue
            donor_mask = (~mask) & (y >= global_cutoff)
            donor_count = int(donor_mask.sum())
            target_mean = float(np.mean(y[mask]))
            donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
            contrast = donor_mean - target_mean if donor_count else float("nan")
            pair_capacity = int(min(target_count, donor_count))
            score = coherence_opportunity_score(
                pair_capacity=pair_capacity,
                target_count=target_count,
                donor_target_contrast=contrast if donor_count else 0.0,
                outcome_scale=scale,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "surface_kind": surface_kind,
                    "block_column": block_column,
                    "block_id": block_id,
                    "axis": axis,
                    "n_records": int(len(block)),
                    "target_count": target_count,
                    "target_prevalence": float(target_count / len(block)),
                    "donor_count": donor_count,
                    "pair_capacity": pair_capacity,
                    "target_mean": target_mean,
                    "donor_mean": donor_mean,
                    "donor_target_contrast": contrast,
                    "outcome_scale": scale,
                    "opportunity_score": score,
                    "passes_opportunity_gate": bool(
                        pair_capacity >= int(min_pair_capacity) and score > 0.0
                    ),
                    "label_multiset_preserving_swap_possible": bool(pair_capacity > 0),
                }
            )
    return sort_opportunity_rows(pd.DataFrame(rows))


def sort_opportunity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows.sort_values(
        ["opportunity_score", "pair_capacity", "donor_target_contrast", "dataset", "axis"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def add_source_note(rows: pd.DataFrame, source_note: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    enriched = rows.copy()
    enriched["source_note"] = source_note
    return enriched


def axis_frame_from_tag_sets(
    y: np.ndarray,
    tag_sets: list[set[str]],
    prefixes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    axes = sorted(
        {
            tag
            for tags in tag_sets
            for tag in tags
            if any(tag.startswith(prefix) for prefix in prefixes)
        }
    )
    columns: dict[str, object] = {"y": np.asarray(y, dtype=float)}
    for axis in axes:
        columns[axis] = np.array([axis in tags for tags in tag_sets], dtype=bool)
    return pd.DataFrame(columns), axes


def audit_materials_dataset(dataset: dict[str, object]) -> list[pd.DataFrame]:
    df = load_matminer_dataset(
        str(dataset["dataset_name"]),
        str(dataset["target_column"]),
        str(dataset["composition_column"]),
    )
    _, tag_sets = material_feature_frame(df[str(dataset["composition_column"])].astype(str).tolist())
    y = df[str(dataset["target_column"])].to_numpy(dtype=float)
    prefixes = require_string_list(dataset, "tag_prefixes", str(dataset["name"]))
    frame, axes = axis_frame_from_tag_sets(y, tag_sets, prefixes)
    rows = candidate_axis_opportunity_rows(
        frame=frame,
        dataset=str(dataset["name"]),
        axis_columns=axes,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_target_count=int(dataset["min_target_count"]),
        max_target_prevalence=float(dataset["max_target_prevalence"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="composition_axis",
    )
    return [add_source_note(rows, str(dataset["source_note"]))]


def cameo_axis_frame(dataset: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    built = build_cameo_dataset(
        str(dataset["data_zip"]),
        xrd_pca_components=int(dataset["xrd_pca_components"]),
        pca_seed=int(dataset["pca_seed"]),
        target_column=str(dataset["target_column"]),
    )
    frame = built.frame.copy()
    axis_columns: list[str] = []
    region_values = sorted(frame["dft_region"].astype(int).unique().tolist())
    data: dict[str, object] = {"y": built.y}
    for region in region_values:
        axis = f"dft_region={region}"
        data[axis] = frame["dft_region"].astype(int).eq(region).to_numpy()
        axis_columns.append(axis)
    for element in ["Fe", "Ga", "Pd"]:
        values = frame[element].to_numpy(dtype=float)
        median = float(np.median(values))
        low_axis = f"{element}<=median"
        high_axis = f"{element}>=median"
        data[low_axis] = values <= median
        data[high_axis] = values >= median
        axis_columns.extend([low_axis, high_axis])
    return pd.DataFrame(data), axis_columns


def audit_cameo_dataset(dataset: dict[str, object]) -> list[pd.DataFrame]:
    frame, axes = cameo_axis_frame(dataset)
    rows = candidate_axis_opportunity_rows(
        frame=frame,
        dataset=str(dataset["name"]),
        axis_columns=axes,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_target_count=int(dataset["min_target_count"]),
        max_target_prevalence=float(dataset["max_target_prevalence"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="closed_loop_region_or_composition_axis",
    )
    return [add_source_note(rows, str(dataset["source_note"]))]


def load_sample_measurements(sample_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(sample_root)
    measurement_rows: list[dict[str, object]] = []
    for agent in [1, 2, 3, 4]:
        path = root / f"Seq_Data_{agent}.csv"
        df = pd.read_csv(path, dtype={"Seq_ID": str})
        values = pd.to_numeric(df["T50"], errors="coerce")
        numeric = df[values.notna()].copy()
        for row_index, row in numeric.iterrows():
            measurement_rows.append(
                {
                    "agent": str(agent),
                    "seq_id": str(row["Seq_ID"]),
                    "y": float(values.loc[row_index]),
                    "fragments": str(row["Fragments"]),
                    "run_id": str(row["run_id"]),
                }
            )
    measurements = pd.DataFrame(measurement_rows)
    if measurements.empty:
        raise ValueError("SAMPLE archive contains no numeric T50 measurements")
    grouped = measurements.groupby("seq_id", as_index=False).agg(
        y=("y", "mean"),
        measurement_count=("y", "size"),
        fragments=("fragments", "first"),
    )
    return grouped.sort_values("seq_id").reset_index(drop=True), measurements


def add_sample_axis_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    enriched = frame.copy()
    axes: list[str] = []
    seq_ids = enriched["seq_id"].astype(str)
    max_len = int(seq_ids.str.len().max())
    for pos in range(max_len):
        values = sorted(seq_ids.str[pos].dropna().unique().tolist())
        for value in values:
            axis = f"pos{pos}={value}"
            enriched[axis] = seq_ids.str[pos].eq(value).to_numpy()
            axes.append(axis)
    fragment_lists = [str(item).split() for item in enriched["fragments"]]
    max_fragments = max(len(parts) for parts in fragment_lists)
    for pos in range(max_fragments):
        values = sorted({parts[pos] for parts in fragment_lists if len(parts) > pos})
        for value in values:
            axis = f"frag{pos}={value}"
            enriched[axis] = np.array(
                [len(parts) > pos and parts[pos] == value for parts in fragment_lists],
                dtype=bool,
            )
            axes.append(axis)
    return enriched, axes


def load_sample_round_frame(
    sample_root: str | Path,
    grouped: pd.DataFrame,
) -> pd.DataFrame:
    summary_path = Path(sample_root) / "Experiment_Summary.csv"
    summary = pd.read_csv(summary_path, dtype=str)
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        sequences = ast.literal_eval(str(row["Sequences"]))
        for position, seq_id in enumerate(sequences):
            rows.append(
                {
                    "round": str(row["Index"]),
                    "round_position": int(position),
                    "seq_id": str(seq_id),
                    "assay_run_id": str(row["Assay Run ID"]),
                    "evagreen_run_id": str(row["Evagreen Run ID"]),
                }
            )
    round_frame = pd.DataFrame(rows)
    merged = round_frame.merge(grouped[["seq_id", "y", "fragments"]], on="seq_id", how="inner")
    return merged.reset_index(drop=True)


def audit_sample_dataset(dataset: dict[str, object]) -> list[pd.DataFrame]:
    grouped, measurements = load_sample_measurements(str(dataset["sample_root"]))
    grouped_axes, grouped_axis_columns = add_sample_axis_columns(grouped)
    measurement_axes, measurement_axis_columns = add_sample_axis_columns(measurements)
    round_frame = load_sample_round_frame(str(dataset["sample_root"]), grouped)
    round_axes, round_axis_columns = add_sample_axis_columns(round_frame)

    global_rows = candidate_axis_opportunity_rows(
        frame=grouped_axes,
        dataset=str(dataset["name"]),
        axis_columns=grouped_axis_columns,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_target_count=int(dataset["min_target_count"]),
        max_target_prevalence=float(dataset["max_target_prevalence"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="unique_sequence_axis",
    )
    run_rows = block_axis_opportunity_rows(
        frame=measurement_axes,
        dataset=str(dataset["name"]),
        block_column="run_id",
        axis_columns=measurement_axis_columns,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="assay_run_block",
    )
    agent_rows = block_axis_opportunity_rows(
        frame=measurement_axes,
        dataset=str(dataset["name"]),
        block_column="agent",
        axis_columns=measurement_axis_columns,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="agent_block",
    )
    round_rows = block_axis_opportunity_rows(
        frame=round_axes,
        dataset=str(dataset["name"]),
        block_column="round",
        axis_columns=round_axis_columns,
        y_column="y",
        donor_quantile=float(dataset["donor_quantile"]),
        min_pair_capacity=int(dataset["min_pair_capacity"]),
        surface_kind="planned_round_block",
    )
    return [
        add_source_note(global_rows, str(dataset["source_note"])),
        add_source_note(run_rows, str(dataset["source_note"])),
        add_source_note(agent_rows, str(dataset["source_note"])),
        add_source_note(round_rows, str(dataset["source_note"])),
    ]


def audit_dataset(dataset: dict[str, object]) -> list[pd.DataFrame]:
    kind = str(dataset["kind"])
    if kind == "materials_matbench":
        return audit_materials_dataset(dataset)
    if kind == "cameo":
        return audit_cameo_dataset(dataset)
    if kind == "sample":
        return audit_sample_dataset(dataset)
    raise ValueError(f"unsupported dataset kind: {kind}")


def summarize_results(rows: pd.DataFrame, cfg: dict[str, object]) -> dict[str, object]:
    positive = rows[rows["passes_opportunity_gate"].astype(bool)].copy()
    top_rows = (
        positive.sort_values("opportunity_score", ascending=False)
        .head(12)
        .replace({np.nan: None})
        .to_dict("records")
    )
    by_dataset_rows: list[dict[str, object]] = []
    for dataset in sorted(rows["dataset"].astype(str).unique().tolist()):
        subset = rows[rows["dataset"].astype(str) == dataset]
        pos_subset = subset[subset["passes_opportunity_gate"].astype(bool)]
        best = float(pos_subset["opportunity_score"].max()) if not pos_subset.empty else 0.0
        by_dataset_rows.append(
            {
                "dataset": dataset,
                "rows": int(len(subset)),
                "positive_rows": int(len(pos_subset)),
                "best_opportunity_score": best,
                "surface_count": int(subset["surface_kind"].nunique()),
            }
        )
    source_hashes: dict[str, str] = {}
    for dataset in cfg["datasets"]:
        if not isinstance(dataset, dict):
            continue
        if "source_archive" in dataset:
            path = Path(str(dataset["source_archive"]))
            if path.is_file():
                source_hashes[str(path)] = file_sha256(path)
        if "data_zip" in dataset:
            path = Path(str(dataset["data_zip"]))
            if path.is_file():
                source_hashes[str(path)] = file_sha256(path)
    return {
        "stage": "b55_natural_coherence_audit",
        "claim_boundary": (
            "This is an opportunity-surface audit of real metadata structures, not evidence "
            "that natural coherent corruption occurred in any source archive."
        ),
        "datasets": by_dataset_rows,
        "total_rows": int(len(rows)),
        "positive_rows": int(len(positive)),
        "top_rows": json_ready(top_rows),
        "source_hashes": source_hashes,
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        floating = float(value)
        if math.isnan(floating):
            return None
        return floating
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_markdown_report(path: Path, summary: dict[str, object], rows: pd.DataFrame) -> None:
    positive = rows[rows["passes_opportunity_gate"].astype(bool)].copy()
    top = positive.sort_values("opportunity_score", ascending=False).head(10)
    representative = representative_surface_rows(positive)
    lines = [
        "# B55 Natural Coherence Opportunity Audit",
        "",
        "Date: 2026-05-30",
        "",
        "## Hypothesis",
        "",
        (
            "Real scientific data archives contain structured binding surfaces such as "
            "composition axes, DFT regions, assay runs, agents and planned rounds. These "
            "surfaces can make label-multiset-preserving relinking coherent enough to "
            "rewrite a conditional record function if a binding error were introduced."
        ),
        "",
        "## Budget, acceptance criteria and stop conditions",
        "",
        "- Budget: offline audit only; no new model training and no wet-lab claim.",
        "- Acceptance: at least two real archives show positive coherent opportunity rows.",
        "- Stop: report no opportunity if metadata blocks or numeric outcomes are unavailable.",
        "",
        "## Result",
        "",
        f"- Total audited rows: {int(summary['total_rows'])}.",
        f"- Positive opportunity rows: {int(summary['positive_rows'])}.",
        "- Boundary: this does not claim that natural coherent corruption occurred.",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Rows | Positive rows | Surfaces | Best score |",
        "|---|---:|---:|---:|---:|",
    ]
    dataset_rows = summary["datasets"]
    if not isinstance(dataset_rows, list):
        raise TypeError("summary datasets must be a list")
    for item in dataset_rows:
        if not isinstance(item, dict):
            raise TypeError("dataset summary item must be a dict")
        lines.append(
            "| {dataset} | {rows} | {positive_rows} | {surface_count} | {best:.3f} |".format(
                dataset=str(item["dataset"]),
                rows=int(item["rows"]),
                positive_rows=int(item["positive_rows"]),
                surface_count=int(item["surface_count"]),
                best=float(item["best_opportunity_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top opportunity rows",
            "",
            "| Dataset | Surface | Block | Axis | Pair capacity | Contrast | Score |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in top.to_dict("records"):
        lines.append(
            "| {dataset} | {surface} | {block} | {axis} | {capacity} | {contrast:.3f} | {score:.3f} |".format(
                dataset=str(row["dataset"]),
                surface=str(row["surface_kind"]),
                block=str(row["block_id"]),
                axis=str(row["axis"]),
                capacity=int(row["pair_capacity"]),
                contrast=float(row["donor_target_contrast"]),
                score=float(row["opportunity_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Best row per source surface",
            "",
            "| Dataset | Surface | Block | Axis | Pair capacity | Contrast | Score |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in representative.to_dict("records"):
        lines.append(
            "| {dataset} | {surface} | {block} | {axis} | {capacity} | {contrast:.3f} | {score:.3f} |".format(
                dataset=str(row["dataset"]),
                surface=str(row["surface_kind"]),
                block=str(row["block_id"]),
                axis=str(row["axis"]),
                capacity=int(row["pair_capacity"]),
                contrast=float(row["donor_target_contrast"]),
                score=float(row["opportunity_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Supported claim",
            "",
            (
                "Public scientific archives expose structured metadata surfaces on which a "
                "small number of record-valid, label-multiset-preserving relinkings could be "
                "coherent rather than random. This strengthens the realism of controlled "
                "coherent-relinking stress tests."
            ),
            "",
            "## Unsupported claims",
            "",
            "- No evidence that CAMEO, SAMPLE or Matbench contain natural coherent corruption.",
            "- No universal vulnerability or universal stealth claim.",
            "- No record-level correction or complete detector claim.",
            "",
            "## Source hashes",
            "",
        ]
    )
    source_hashes = summary["source_hashes"]
    if not isinstance(source_hashes, dict):
        raise TypeError("source_hashes must be a dict")
    for source, digest in source_hashes.items():
        lines.append(f"- `{source}`: `{digest}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def representative_surface_rows(positive: pd.DataFrame) -> pd.DataFrame:
    if positive.empty:
        return positive
    selected_indices = positive.groupby(["dataset", "surface_kind"])["opportunity_score"].idxmax()
    return positive.loc[selected_indices].sort_values(
        ["opportunity_score", "dataset", "surface_kind"],
        ascending=[False, True, True],
    )


def make_figure(rows: pd.DataFrame, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style(font_size=8.3)
    positive = rows[rows["passes_opportunity_gate"].astype(bool)].copy()
    if positive.empty:
        raise ValueError("cannot plot empty positive opportunity table")
    representative = representative_surface_rows(positive)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["purple"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["sky"],
    ]

    surface = (
        positive.groupby(["dataset", "surface_kind"], as_index=False)["opportunity_score"]
        .max()
        .sort_values("opportunity_score", ascending=True)
    )
    labels = [f"{row.dataset}\n{row.surface_kind}" for row in surface.itertuples()]
    axes[0].barh(
        range(len(surface)),
        surface["opportunity_score"].to_numpy(dtype=float),
        color=[colors[idx % len(colors)] for idx in range(len(surface))],
    )
    axes[0].set_yticks(range(len(surface)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("best opportunity score")
    axes[0].set_title("a  structured surfaces", loc="left", fontweight="semibold")
    style_axis(axes[0], grid_axis="x")

    datasets = sorted(positive["dataset"].astype(str).unique().tolist())
    for idx, dataset in enumerate(datasets):
        subset = positive[positive["dataset"].astype(str) == dataset]
        axes[1].scatter(
            subset["pair_capacity"].to_numpy(dtype=float),
            subset["donor_target_contrast"].to_numpy(dtype=float),
            s=18,
            alpha=0.75,
            color=colors[idx % len(colors)],
            label=dataset,
            edgecolors="white",
            linewidths=0.25,
        )
    axes[1].set_xlabel("pair capacity")
    axes[1].set_ylabel("donor-target contrast")
    axes[1].set_title("b  capacity and contrast", loc="left", fontweight="semibold")
    axes[1].legend(frameon=False, loc="best")
    style_axis(axes[1], grid_axis="both")

    top = representative.sort_values("opportunity_score", ascending=True)
    top_labels = [
        f"{row.dataset}\n{row.surface_kind}\n{row.axis}" for row in top.itertuples()
    ]
    axes[2].barh(
        range(len(top)),
        top["opportunity_score"].to_numpy(dtype=float),
        color=OKABE_ITO["blue"],
        alpha=0.88,
    )
    axes[2].set_yticks(range(len(top)))
    axes[2].set_yticklabels(top_labels)
    axes[2].set_xlabel("opportunity score")
    axes[2].set_title("c  top axes/blocks", loc="left", fontweight="semibold")
    style_axis(axes[2], grid_axis="x")

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    png = output_pdf.with_suffix(".png")
    svg = output_pdf.with_suffix(".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg = parse_args()
    row_frames: list[pd.DataFrame] = []
    for dataset in cfg["datasets"]:
        if not isinstance(dataset, dict):
            raise TypeError("dataset must be a dict")
        row_frames.extend(audit_dataset(dataset))
    nonempty = [frame for frame in row_frames if not frame.empty]
    if not nonempty:
        raise ValueError("no audit rows were produced")
    rows = sort_opportunity_rows(pd.concat(nonempty, ignore_index=True))

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_figure_pdf = Path(str(cfg["output_figure_pdf"]))
    for path in [output_csv, output_json, output_md, output_figure_pdf]:
        path.parent.mkdir(parents=True, exist_ok=True)

    rows.to_csv(output_csv, index=False)
    summary = summarize_results(rows, cfg)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, sort_keys=True)
    write_markdown_report(output_md, summary, rows)
    make_figure(rows, output_figure_pdf)

    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(rows.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
