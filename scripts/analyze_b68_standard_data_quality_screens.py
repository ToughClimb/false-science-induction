#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.features import mutation_feature_frame  # noqa: E402
from false_science.materials import load_matminer_dataset, material_feature_frame  # noqa: E402
from false_science.protein import load_gfp_csv  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "datasets",
    "knn_neighbors",
    "output_csv",
    "output_json",
    "output_md",
    "output_tex",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "run_config_file",
    "feature_source",
    "initial_history_file",
    "pairs_file",
    "clean_mode",
    "random_mode",
    "targeted_mode",
]

REQUIRED_HISTORY_COLUMNS = [
    "seed",
    "mode",
    "record_id",
    "recorded_label",
]

REQUIRED_PAIR_COLUMNS = [
    "target_record_id",
    "donor_record_id",
]

SUPPORTED_FEATURE_SOURCES = {
    "csv_features",
    "gfp_mutation",
    "materials_composition",
}


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    datasets = cfg["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(dataset, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        source = str(dataset["feature_source"])
        if source not in SUPPORTED_FEATURE_SOURCES:
            allowed = ", ".join(sorted(SUPPORTED_FEATURE_SOURCES))
            raise ValueError(f"datasets[{index}].feature_source must be one of: {allowed}")
        typed.append(dataset)
    return typed


def read_csv_required(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def load_run_config(dataset: dict[str, object]) -> dict[str, object]:
    path = Path(str(dataset["run_dir"])) / str(dataset["run_config_file"])
    return load_json_config(path)


def load_feature_matrix(dataset: dict[str, object]) -> np.ndarray:
    source = str(dataset["feature_source"])
    if source == "csv_features":
        path = Path(str(dataset["run_dir"])) / str(dataset["run_config_file"])
        frame = read_csv_required(path, ["record_id"])
        ordered = frame.sort_values("record_id")
        feature_columns = [column for column in ordered.columns if column != "record_id"]
        if not feature_columns:
            raise ValueError("csv_features source requires at least one feature column")
        return ordered[feature_columns].to_numpy(dtype=np.float32)
    run_cfg = load_run_config(dataset)
    if source == "gfp_mutation":
        needed = ["data_path", "target_column", "mutant_column", "max_rows", "random_state"]
        require_keys(run_cfg, needed, f"{dataset['name']}.run_config")
        frame = load_gfp_csv(
            str(run_cfg["data_path"]),
            str(run_cfg["target_column"]),
            str(run_cfg["mutant_column"]),
            run_cfg["max_rows"],
            int(run_cfg["random_state"]),
        )
        features = mutation_feature_frame(frame, str(run_cfg["mutant_column"]))
        return features.to_numpy(dtype=np.float32)
    if source == "materials_composition":
        needed = ["dataset_name", "target_column", "composition_column"]
        require_keys(run_cfg, needed, f"{dataset['name']}.run_config")
        frame = load_matminer_dataset(
            str(run_cfg["dataset_name"]),
            str(run_cfg["target_column"]),
            str(run_cfg["composition_column"]),
        )
        features, _ = material_feature_frame(frame[str(run_cfg["composition_column"])].astype(str))
        return features.to_numpy(dtype=np.float32)
    raise ValueError(f"unknown feature source: {source}")


def standardize_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (x - mean) / scale


def label_z_scores(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=float)
    scale = float(y.std())
    if scale == 0.0:
        scale = 1.0
    return np.abs((y - float(y.mean())) / scale)


def knn_residual_scores(features: np.ndarray, labels: np.ndarray, neighbors: int) -> np.ndarray:
    y = np.asarray(labels, dtype=float)
    if len(y) <= 1:
        return np.zeros(len(y), dtype=float)
    n_neighbors = min(int(neighbors), len(y) - 1)
    model = NearestNeighbors(n_neighbors=n_neighbors + 1)
    model.fit(features)
    indices = model.kneighbors(features, return_distance=False)[:, 1:]
    local_mean = y[indices].mean(axis=1)
    return np.abs(y - local_mean)


def score_metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = int(truth.sum())
    if positives <= 0 or positives >= len(truth):
        return {
            "auroc": float("nan"),
            "topk_recall": float("nan"),
            "topk_precision": float("nan"),
        }
    ranked = np.argsort(-scores)
    top = ranked[:positives]
    recovered = int(truth[top].sum())
    return {
        "auroc": float(roc_auc_score(truth, scores)),
        "topk_recall": float(recovered / positives),
        "topk_precision": float(recovered / len(top)),
    }


def pairs_for_all_seeds(pairs: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    if "seed" in pairs.columns:
        return pairs
    frames = [pairs.assign(seed=seed) for seed in seeds]
    return pd.concat(frames, ignore_index=True)


def positive_sets(pairs: pd.DataFrame) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    pair_records: set[tuple[int, int]] = set()
    target_records: set[tuple[int, int]] = set()
    for row in pairs.itertuples(index=False):
        seed = int(row.seed)
        target_id = int(row.target_record_id)
        donor_id = int(row.donor_record_id)
        pair_records.add((seed, target_id))
        pair_records.add((seed, donor_id))
        target_records.add((seed, target_id))
    return pair_records, target_records


def rows_for_dataset(dataset: dict[str, object], neighbors: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = Path(str(dataset["run_dir"]))
    history = read_csv_required(
        run_dir / str(dataset["initial_history_file"]),
        REQUIRED_HISTORY_COLUMNS,
    )
    pairs = read_csv_required(run_dir / str(dataset["pairs_file"]), REQUIRED_PAIR_COLUMNS)
    seeds = [int(seed) for seed in sorted(history["seed"].unique().tolist())]
    pairs = pairs_for_all_seeds(pairs, seeds)
    pair_records, target_records = positive_sets(pairs)
    features_all = load_feature_matrix(dataset)

    rows: list[dict[str, object]] = []
    for (seed, mode), frame in history.groupby(["seed", "mode"], sort=True):
        record_ids = frame["record_id"].to_numpy(dtype=int)
        if int(record_ids.max()) >= len(features_all):
            raise ValueError(f"record_id exceeds feature matrix for {dataset['name']}")
        labels = frame["recorded_label"].to_numpy(dtype=float)
        features = standardize_features(features_all[record_ids])
        pair_truth = np.array(
            [(int(seed), int(record_id)) in pair_records for record_id in record_ids],
            dtype=int,
        )
        target_truth = np.array(
            [(int(seed), int(record_id)) in target_records for record_id in record_ids],
            dtype=int,
        )
        scores = {
            "label_z_extremeness": label_z_scores(labels),
            "feature_knn_residual": knn_residual_scores(features, labels, neighbors),
        }
        for score_name, score_values in scores.items():
            pair_metrics = score_metrics(pair_truth, score_values)
            target_metrics = score_metrics(target_truth, score_values)
            rows.append(
                {
                    "dataset": str(dataset["name"]),
                    "seed": int(seed),
                    "mode": str(mode),
                    "screen": score_name,
                    "history_count": int(len(frame)),
                    "pair_positive_count": int(pair_truth.sum()),
                    "target_positive_count": int(target_truth.sum()),
                    "pair_auroc": float(pair_metrics["auroc"]),
                    "pair_topk_recall": float(pair_metrics["topk_recall"]),
                    "pair_topk_precision": float(pair_metrics["topk_precision"]),
                    "target_auroc": float(target_metrics["auroc"]),
                    "target_topk_recall": float(target_metrics["topk_recall"]),
                    "target_topk_precision": float(target_metrics["topk_precision"]),
                }
            )
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    clean_mode = str(dataset["clean_mode"])
    random_mode = str(dataset["random_mode"])
    targeted_mode = str(dataset["targeted_mode"])
    for screen, screen_frame in detail.groupby("screen", sort=True):
        mode_summary = (
            screen_frame.groupby("mode", as_index=False)
            .agg(
                pair_auroc_mean=("pair_auroc", "mean"),
                pair_topk_recall_mean=("pair_topk_recall", "mean"),
                target_auroc_mean=("target_auroc", "mean"),
                target_topk_recall_mean=("target_topk_recall", "mean"),
            )
        )
        mode_rows = {str(row.mode): row for row in mode_summary.itertuples(index=False)}
        for metric in [
            "pair_auroc_mean",
            "pair_topk_recall_mean",
            "target_auroc_mean",
            "target_topk_recall_mean",
        ]:
            targeted_value = float(getattr(mode_rows[targeted_mode], metric))
            control_value = max(
                float(getattr(mode_rows[clean_mode], metric)),
                float(getattr(mode_rows[random_mode], metric)),
            )
            summaries.append(
                {
                    "dataset": str(dataset["name"]),
                    "screen": str(screen),
                    "metric": metric,
                    "clean": float(getattr(mode_rows[clean_mode], metric)),
                    "random": float(getattr(mode_rows[random_mode], metric)),
                    "targeted": targeted_value,
                    "targeted_minus_max_control": float(targeted_value - control_value),
                }
            )
    return rows, summaries


def format_float(value: object) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.3f}"


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B68 Standard Data-Quality Screen Boundary",
        "",
        "Hypothesis: non-provenance-aware data-quality screens can flag extremeness or local feature-label inconsistency, but they do not by themselves constitute a binding-aware detector or record-level repair.",
        "",
        "| Dataset | Screen | Metric | Clean | Random | Targeted | Targeted - max control |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {screen} | {metric} | {clean} | {random} | {targeted} | {delta} |".format(
                dataset=row["dataset"],
                screen=row["screen"],
                metric=row["metric"],
                clean=format_float(row["clean"]),
                random=format_float(row["random"]),
                targeted=format_float(row["targeted"]),
                delta=format_float(row["targeted_minus_max_control"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The screens are intentionally generic and do not use provenance, donor-target pairing, acquisition traces or the injected target name. Label z-score extremeness often ranks records that are intrinsically extreme even in clean/random histories; this is useful context but not a binding-specific signal. Feature-neighbour residuals provide a stronger warning in some materials histories, but the signal is domain-dependent and does not recover GFP target-side misbinding reliably. The result strengthens the paper's boundary claim: standard data-quality checks can be useful sanity checks, but they are not a complete replacement for binding-aware validation, calibrated trace guards or feedback-conflict triage.",
            "",
            "## Non-Claims",
            "",
            "- No universal stealth claim.",
            "- No claim that all standard screens fail.",
            "- No record-level correction.",
            "- No calibration-free complete detector.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(path: Path, summaries: list[dict[str, object]]) -> None:
    selected_metrics = {
        "target_topk_recall_mean",
        "pair_topk_recall_mean",
    }
    selected = [row for row in summaries if row["metric"] in selected_metrics]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Boundary of generic data-quality screens on the primary paired-misbinding histories.}",
        "\\label{tab:standard-screen-boundaries}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Screen & Metric & Clean & Random & Targeted & $\\Delta$ vs ctrl. \\\\",
        "\\midrule",
    ]
    for row in selected:
        dataset = str(row["dataset"]).replace("_", "\\_")
        screen = str(row["screen"]).replace("_", "\\_")
        metric = str(row["metric"]).replace("_", "\\_")
        lines.append(
            f"{dataset} & {screen} & {metric} & {format_float(row['clean'])} & {format_float(row['random'])} & {format_float(row['targeted'])} & {format_float(row['targeted_minus_max_control'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\vspace{0.4em}\\caption*{\\footnotesize Generic screens use only recorded labels or feature-neighbour residuals and do not use provenance, donor-target pairing or acquisition traces. Positive $\\Delta$ indicates extra targeted-run recovery beyond the stronger clean/random control. The mixed results show useful sanity-check boundaries, not a complete binding detector.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B68 generic data-quality screen boundaries.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b68_standard_data_quality_screens")
    datasets = require_dataset_list(cfg)
    neighbors = int(cfg["knn_neighbors"])
    if neighbors <= 0:
        raise ValueError("knn_neighbors must be positive")

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, dataset_summaries = rows_for_dataset(dataset, neighbors)
        rows.extend(dataset_rows)
        summaries.extend(dataset_summaries)

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_tex = Path(str(cfg["output_tex"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summaries": summaries}, handle, indent=2, sort_keys=True)
    write_markdown(output_md, summaries)
    write_tex(output_tex, summaries)
    print(output_csv)
    print(output_json)
    print(output_md)
    print(output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
