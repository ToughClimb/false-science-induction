#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.features import mutation_feature_frame  # noqa: E402
from false_science.materials import load_matminer_dataset, material_feature_frame  # noqa: E402
from false_science.protein import load_gfp_csv  # noqa: E402
from scripts.b70_bear_physical_sdl_replay import (  # noqa: E402
    MODES as BEAR_MODES,
    build_recorded_labels,
    choose_target_axis,
    read_bear_campaign,
    rf_ucb_scores,
)


REQUIRED_CONFIG_KEYS = [
    "datasets",
    "bear",
    "knn_neighbors",
    "pca_components",
    "ridge_alpha",
    "oob_n_estimators",
    "output_screen_csv",
    "output_concentration_csv",
    "output_bear_selected_csv",
    "output_bear_triage_csv",
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
    "selected_file",
    "clean_mode",
    "random_mode",
    "targeted_mode",
]

REQUIRED_BEAR_KEYS = [
    "config_path",
    "axis_candidates",
    "target_axis_aliases",
    "min_axis_count",
]

FEATURE_SOURCES = {
    "gfp_mutation",
    "materials_composition",
}


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    value = cfg["datasets"]
    if not isinstance(value, list):
        raise TypeError("datasets must be a JSON list")
    datasets: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(item, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        source = str(item["feature_source"])
        if source not in FEATURE_SOURCES:
            allowed = ", ".join(sorted(FEATURE_SOURCES))
            raise ValueError(f"datasets[{index}].feature_source must be one of: {allowed}")
        datasets.append(item)
    return datasets


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{context}.{key} must contain strings")
        output.append(item)
    return output


def read_csv_required(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def load_run_config(dataset: dict[str, object]) -> dict[str, object]:
    run_dir = Path(str(dataset["run_dir"]))
    return load_json_config(run_dir / str(dataset["run_config_file"]))


def load_features(dataset: dict[str, object]) -> np.ndarray:
    source = str(dataset["feature_source"])
    run_cfg = load_run_config(dataset)
    if source == "gfp_mutation":
        required = [
            "data_path",
            "target_column",
            "mutant_column",
            "max_rows",
            "random_state",
        ]
        require_keys(run_cfg, required, f"{dataset['name']}.run_config")
        frame = load_gfp_csv(
            str(run_cfg["data_path"]),
            str(run_cfg["target_column"]),
            str(run_cfg["mutant_column"]),
            run_cfg["max_rows"],
            int(run_cfg["random_state"]),
        )
        return mutation_feature_frame(frame, str(run_cfg["mutant_column"])).to_numpy(dtype=float)
    if source == "materials_composition":
        required = ["dataset_name", "target_column", "composition_column"]
        require_keys(run_cfg, required, f"{dataset['name']}.run_config")
        frame = load_matminer_dataset(
            str(run_cfg["dataset_name"]),
            str(run_cfg["target_column"]),
            str(run_cfg["composition_column"]),
        )
        features, _ = material_feature_frame(frame[str(run_cfg["composition_column"])].astype(str))
        return features.to_numpy(dtype=float)
    raise ValueError(f"unknown feature source: {source}")


def standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (x - mean) / scale


def z_scores(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    scale = float(y.std())
    if scale == 0.0:
        scale = 1.0
    return np.abs((y - float(y.mean())) / scale)


def knn_residual(features: np.ndarray, labels: np.ndarray, neighbors: int) -> np.ndarray:
    if len(labels) <= 1:
        return np.zeros(len(labels), dtype=float)
    k = min(int(neighbors), len(labels) - 1)
    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(features)
    indices = model.kneighbors(features, return_distance=False)[:, 1:]
    local_mean = labels[indices].mean(axis=1)
    return np.abs(labels - local_mean)


def ridge_residual(features: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=float(alpha))
    model.fit(features, labels)
    prediction = model.predict(features)
    return np.abs(labels - prediction)


def rf_oob_residual(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed) + 73001,
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
    )
    model.fit(features, labels)
    prediction = np.asarray(model.oob_prediction_, dtype=float)
    fallback = np.isnan(prediction)
    if bool(fallback.any()):
        prediction[fallback] = model.predict(features[fallback])
    return np.abs(labels - prediction)


def pca_spectral_score(features: np.ndarray, components: int) -> np.ndarray:
    if len(features) <= 2:
        return np.zeros(len(features), dtype=float)
    n_components = min(int(components), features.shape[1], len(features) - 1)
    model = PCA(n_components=n_components, random_state=73001)
    projected = model.fit_transform(features)
    return np.linalg.norm(projected, axis=1)


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


def pair_sets(pairs: pd.DataFrame, seeds: list[int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if "seed" not in pairs.columns:
        frames = [pairs.assign(seed=seed) for seed in seeds]
        pairs = pd.concat(frames, ignore_index=True)
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


def screen_rows_for_dataset(
    dataset: dict[str, object],
    neighbors: int,
    pca_components: int,
    ridge_alpha: float,
    oob_n_estimators: int,
) -> list[dict[str, object]]:
    run_dir = Path(str(dataset["run_dir"]))
    history = read_csv_required(
        run_dir / str(dataset["initial_history_file"]),
        ["seed", "mode", "record_id", "true_label", "recorded_label"],
    )
    pairs = read_csv_required(
        run_dir / str(dataset["pairs_file"]),
        ["target_record_id", "donor_record_id"],
    )
    seeds = [int(seed) for seed in sorted(history["seed"].unique().tolist())]
    pair_records, target_records = pair_sets(pairs, seeds)
    features_all = load_features(dataset)
    rows: list[dict[str, object]] = []
    for (seed, mode), frame in history.groupby(["seed", "mode"], sort=True):
        record_ids = frame["record_id"].to_numpy(dtype=int)
        features = standardize(features_all[record_ids])
        labels = frame["recorded_label"].to_numpy(dtype=float)
        pair_truth = np.array(
            [(int(seed), int(record_id)) in pair_records for record_id in record_ids],
            dtype=int,
        )
        target_truth = np.array(
            [(int(seed), int(record_id)) in target_records for record_id in record_ids],
            dtype=int,
        )
        screens = {
            "label_z_extremeness": z_scores(labels),
            "feature_knn_residual": knn_residual(features, labels, neighbors),
            "ridge_loss_residual": ridge_residual(features, labels, ridge_alpha),
            "rf_oob_loss_residual": rf_oob_residual(
                features,
                labels,
                seed=int(seed),
                n_estimators=oob_n_estimators,
            ),
            "pca_spectral_score": pca_spectral_score(features, pca_components),
        }
        for screen, values in screens.items():
            pair_metrics = score_metrics(pair_truth, values)
            target_metrics = score_metrics(target_truth, values)
            rows.append(
                {
                    "dataset": str(dataset["name"]),
                    "seed": int(seed),
                    "mode": str(mode),
                    "screen": screen,
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
    return rows


def concentration_rows_for_dataset(dataset: dict[str, object]) -> list[dict[str, object]]:
    run_dir = Path(str(dataset["run_dir"]))
    selected = read_csv_required(
        run_dir / str(dataset["selected_file"]),
        ["model", "mode", "seed", "true_label", "is_triggered_target"],
    )
    history = read_csv_required(
        run_dir / str(dataset["initial_history_file"]),
        ["true_label"],
    )
    high_cutoff = float(history["true_label"].quantile(0.9))
    rows: list[dict[str, object]] = []
    for (model, mode), frame in selected.groupby(["model", "mode"], sort=True):
        target = frame[frame["is_triggered_target"].astype(bool)]
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "model": str(model),
                "mode": str(mode),
                "selected_count": int(len(frame)),
                "selected_true_mean": float(frame["true_label"].mean()),
                "high_true_cutoff_history_q90": high_cutoff,
                "high_true_fraction": float((frame["true_label"] >= high_cutoff).mean()),
                "triggered_target_count": int(frame["is_triggered_target"].sum()),
                "triggered_target_fraction": float(frame["is_triggered_target"].mean()),
                "triggered_target_true_mean": float(target["true_label"].mean())
                if len(target)
                else float("nan"),
            }
        )
    return rows


def run_bear_selected_records(cfg: dict[str, object], bear_cfg: dict[str, object]) -> pd.DataFrame:
    feature_columns = require_string_list(bear_cfg, "feature_columns", "bear_config")
    axis_candidates = require_string_list(bear_cfg, "target_axis_candidates", "bear_config")
    frame = read_bear_campaign(
        Path(str(bear_cfg["data_path"])),
        str(bear_cfg["target_column"]),
        feature_columns,
    )
    axis, value, _scan = choose_target_axis(
        frame=frame,
        target_column=str(bear_cfg["target_column"]),
        axis_candidates=axis_candidates,
        target_quantile=float(bear_cfg["target_quantile"]),
        min_history_target_count=int(bear_cfg["min_history_target_count"]),
        min_candidate_target_count=int(bear_cfg["min_candidate_target_count"]),
        history_size=int(bear_cfg["history_size"]),
    )
    true_y = frame[str(bear_cfg["target_column"])].to_numpy(dtype=float)
    features = frame[feature_columns].to_numpy(dtype=float)
    target = frame[axis].to_numpy(dtype=float) == float(value)
    history_size = int(bear_cfg["history_size"])
    candidate_size = int(bear_cfg["candidate_size"])
    base_history_ids = np.arange(history_size, dtype=int)
    base_candidate_ids = np.arange(
        history_size,
        min(history_size + candidate_size, len(frame)),
        dtype=int,
    )
    target_history_ids = base_history_ids[target[base_history_ids]]
    donor_history_ids = base_history_ids[(~target[base_history_ids])]
    donor_history_ids = donor_history_ids[
        true_y[donor_history_ids] >= np.quantile(true_y[base_history_ids], 0.85)
    ]
    selected_rows: list[dict[str, object]] = []
    for seed in [int(seed) for seed in bear_cfg["seeds"]]:
        for mode in BEAR_MODES:
            train_ids = base_history_ids.copy()
            candidate_ids = base_candidate_ids.copy()
            recorded, _pairs_used = build_recorded_labels(
                true_y=true_y,
                history_ids=base_history_ids,
                target_history_ids=target_history_ids,
                donor_history_ids=donor_history_ids,
                mode=mode,
                seed=seed,
                swap_count=int(bear_cfg["swap_count"]),
            )
            train_y = recorded.copy()
            for round_idx in range(int(bear_cfg["rounds"])):
                _mean, score = rf_ucb_scores(
                    features[train_ids],
                    train_y,
                    features[candidate_ids],
                    seed=seed * 1000 + round_idx,
                    n_estimators=int(bear_cfg["n_estimators"]),
                    beta=float(bear_cfg["acquisition_beta"]),
                )
                order = np.argsort(-score)
                batch_ids = candidate_ids[order[: int(bear_cfg["batch_size"])]]
                for rank, record_id in enumerate(batch_ids.tolist()):
                    record = frame.iloc[int(record_id)]
                    axis_values = [
                        f"PrinterNozzle={record['PrinterNozzle']:g}",
                        f"NozzleSize={record['NozzleSize']:g}",
                        f"PrinterNumber={record['PrinterNumber']:g}",
                        f"DecisionPolicy={record['DecisionPolicy']:g}",
                    ]
                    selected_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "model": "rf_ucb",
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "true_label": float(true_y[int(record_id)]),
                            "is_target": bool(target[int(record_id)]),
                            "PrinterNozzle": float(record["PrinterNozzle"]),
                            "NozzleSize": float(record["NozzleSize"]),
                            "PrinterNumber": float(record["PrinterNumber"]),
                            "DecisionPolicy": float(record["DecisionPolicy"]),
                            "axis_values": "|".join(axis_values),
                        }
                    )
                keep = np.ones(len(candidate_ids), dtype=bool)
                keep[order[: int(bear_cfg["batch_size"])]] = False
                candidate_ids = candidate_ids[keep]
                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, true_y[batch_ids]]).astype(float)
    return pd.DataFrame(selected_rows)


def bear_axis_rows(
    selected: pd.DataFrame,
    axis_candidates: list[str],
    min_axis_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (mode, seed), frame in selected.groupby(["mode", "seed"], sort=True):
        selected_count = int(len(frame))
        selected_mean = float(frame["true_label"].mean())
        axis_rows: list[dict[str, object]] = []
        for axis in axis_candidates:
            for value, axis_frame in frame.groupby(axis, sort=True):
                count = int(len(axis_frame))
                if count < int(min_axis_count):
                    continue
                axis_mean = float(axis_frame["true_label"].mean())
                fraction = count / selected_count
                deficit = selected_mean - axis_mean
                axis_rows.append(
                    {
                        "dataset": "BEAR B70",
                        "model": "rf_ucb",
                        "mode": str(mode),
                        "seed": str(seed),
                        "scope": "seed",
                        "axis": f"{axis}={float(value):g}",
                        "selected_count": selected_count,
                        "axis_selected_count": count,
                        "axis_selected_fraction": float(fraction),
                        "selected_true_mean": selected_mean,
                        "axis_true_mean": axis_mean,
                        "feedback_deficit": float(deficit),
                        "conflict_score": float(fraction * max(0.0, deficit)),
                    }
                )
        axis_rows.sort(key=lambda row: (-float(row["conflict_score"]), str(row["axis"])))
        for rank, row in enumerate(axis_rows, start=1):
            row["conflict_rank"] = int(rank)
        rows.extend(axis_rows)
    for mode, frame in selected.groupby("mode", sort=True):
        selected_count = int(len(frame))
        selected_mean = float(frame["true_label"].mean())
        aggregate_rows: list[dict[str, object]] = []
        for axis in axis_candidates:
            for value, axis_frame in frame.groupby(axis, sort=True):
                count = int(len(axis_frame))
                if count < int(min_axis_count):
                    continue
                axis_mean = float(axis_frame["true_label"].mean())
                fraction = count / selected_count
                deficit = selected_mean - axis_mean
                aggregate_rows.append(
                    {
                        "dataset": "BEAR B70",
                        "model": "rf_ucb",
                        "mode": str(mode),
                        "seed": "all",
                        "scope": "aggregate",
                        "axis": f"{axis}={float(value):g}",
                        "selected_count": selected_count,
                        "axis_selected_count": count,
                        "axis_selected_fraction": float(fraction),
                        "selected_true_mean": selected_mean,
                        "axis_true_mean": axis_mean,
                        "feedback_deficit": float(deficit),
                        "conflict_score": float(fraction * max(0.0, deficit)),
                    }
                )
        aggregate_rows.sort(key=lambda row: (-float(row["conflict_score"]), str(row["axis"])))
        for rank, row in enumerate(aggregate_rows, start=1):
            row["conflict_rank"] = int(rank)
        rows.extend(aggregate_rows)
    return rows


def summarize_screens(rows: list[dict[str, object]], datasets: list[dict[str, object]]) -> list[dict[str, object]]:
    detail = pd.DataFrame(rows)
    output: list[dict[str, object]] = []
    for dataset in datasets:
        name = str(dataset["name"])
        clean_mode = str(dataset["clean_mode"])
        random_mode = str(dataset["random_mode"])
        targeted_mode = str(dataset["targeted_mode"])
        subset = detail[detail["dataset"] == name]
        for screen, frame in subset.groupby("screen", sort=True):
            grouped = frame.groupby("mode", as_index=False).agg(
                pair_topk_recall=("pair_topk_recall", "mean"),
                target_topk_recall=("target_topk_recall", "mean"),
                pair_auroc=("pair_auroc", "mean"),
                target_auroc=("target_auroc", "mean"),
            )
            mode_rows = {str(row.mode): row for row in grouped.itertuples(index=False)}
            for metric in [
                "pair_topk_recall",
                "target_topk_recall",
                "pair_auroc",
                "target_auroc",
            ]:
                clean_value = float(getattr(mode_rows[clean_mode], metric))
                random_value = float(getattr(mode_rows[random_mode], metric))
                targeted_value = float(getattr(mode_rows[targeted_mode], metric))
                control_value = max(clean_value, random_value)
                output.append(
                    {
                        "dataset": name,
                        "screen": str(screen),
                        "metric": metric,
                        "clean": clean_value,
                        "random": random_value,
                        "targeted": targeted_value,
                        "targeted_minus_max_control": float(targeted_value - control_value),
                    }
                )
    return output


def summarize_bear_triage(rows: list[dict[str, object]], aliases: list[str]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    alias_set = set(aliases)
    output: list[dict[str, object]] = []
    for mode, mode_frame in frame.groupby("mode", sort=True):
        seed_frame = mode_frame[mode_frame["scope"] == "seed"]
        seeds = sorted(seed_frame["seed"].unique().tolist())
        top1 = 0
        top2 = 0
        for seed in seeds:
            seed_rows = seed_frame[seed_frame["seed"] == seed]
            alias_rows = seed_rows[seed_rows["axis"].isin(alias_set)]
            if alias_rows.empty:
                best_rank = int(seed_rows["conflict_rank"].max()) + 1
            else:
                best_rank = int(alias_rows["conflict_rank"].min())
            if best_rank == 1:
                top1 += 1
            if best_rank <= 2:
                top2 += 1
        aggregate = mode_frame[mode_frame["scope"] == "aggregate"].sort_values("conflict_rank")
        alias_aggregate = aggregate[aggregate["axis"].isin(alias_set)]
        aggregate_rank = int(alias_aggregate["conflict_rank"].min()) if not alias_aggregate.empty else int(len(aggregate) + 1)
        aggregate_top_axis = str(aggregate.iloc[0]["axis"]) if not aggregate.empty else ""
        output.append(
            {
                "dataset": "BEAR B70",
                "model": "rf_ucb",
                "mode": str(mode),
                "n_seeds": int(len(seeds)),
                "seed_top1_recovered": int(top1),
                "seed_top2_recovered": int(top2),
                "aggregate_top_axis": aggregate_top_axis,
                "aggregate_target_best_rank": aggregate_rank,
                "aggregate_top1_is_target_alias": bool(aggregate_top_axis in alias_set),
            }
        )
    return output


def format_float(value: object) -> str:
    numeric = float(value)
    if np.isnan(numeric):
        return "--"
    return f"{numeric:.3f}"


def write_tex(path: Path, screen_summary: list[dict[str, object]], bear_summary: list[dict[str, object]]) -> None:
    screen_rows = [
        row
        for row in screen_summary
        if row["metric"] == "target_topk_recall"
        and row["screen"] in {"feature_knn_residual", "rf_oob_loss_residual", "pca_spectral_score"}
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Standard-screen and feedback triage checks: standard screens and BEAR feedback triage.}",
        "\\label{tab:b73-standard-screen-closure}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Check & Metric & Clean & Random & Targeted & $\\Delta$ vs ctrl. \\\\",
        "\\midrule",
    ]
    for row in screen_rows:
        dataset = str(row["dataset"]).replace("_", "\\_")
        check = str(row["screen"]).replace("_", "\\_")
        lines.append(
            f"{dataset} & {check} & target top-$k$ recall & {format_float(row['clean'])} & {format_float(row['random'])} & {format_float(row['targeted'])} & {format_float(row['targeted_minus_max_control'])} \\\\"
        )
    for row in bear_summary:
        dataset = str(row["dataset"]).replace("_", "\\_")
        mode = str(row["mode"]).replace("_", "\\_")
        n_seeds = int(row["n_seeds"])
        top2 = int(row["seed_top2_recovered"])
        aggregate_rank = int(row["aggregate_target_best_rank"])
        lines.append(
            f"{dataset} & feedback-conflict triage ({mode}) & target top-2 / agg. rank & -- & -- & {top2}/{n_seeds} & rank {aggregate_rank} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\vspace{0.35em}\\caption*{\\footnotesize Generic screens use recorded labels, feature-neighbour residuals, out-of-bag prediction residuals or feature-space spectral scores; they do not use provenance, acquisition traces or target names. BEAR triage ranks axes by selected-budget fraction times true-feedback deficit and uses the target axis only after ranking.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(
    path: Path,
    screen_summary: list[dict[str, object]],
    concentration: pd.DataFrame,
    bear_summary: list[dict[str, object]],
) -> None:
    lines = [
        "# B73 Standard-Screen and Feedback-Triage Result",
        "",
        "## Hypotheses",
        "",
        "1. Generic label/loss/feature/spectral screens provide useful sanity checks but do not consistently identify the target side of paired binding errors.",
        "2. Healthy closed-loop concentration differs from false pursuit because clean/random selections retain higher true outcomes while targeted concentration accumulates low-true-value target records.",
        "3. The feedback-conflict triage idea should also be testable on BEAR real-measurement replay traces.",
        "",
        "## Budget and Stop Conditions",
        "",
        "- Read existing B18, B19 and B70 artifacts only; no model retraining beyond lightweight out-of-bag screen fits on initial histories.",
        "- Stop after generating screen, concentration and BEAR triage tables.",
        "- Do not overwrite existing run directories or baselines.",
        "",
        "## Standard screen summary",
        "",
        "| Dataset | Screen | Metric | Clean | Random | Targeted | Targeted - max control |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in screen_summary:
        if str(row["metric"]) != "target_topk_recall":
            continue
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
            "## Healthy versus false concentration",
            "",
            "| Dataset | Model | Mode | Selected true mean | High-true fraction | Triggered-target fraction | Triggered-target true mean |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in concentration.to_dict("records"):
        lines.append(
            "| {dataset} | {model} | {mode} | {selected_true_mean} | {high_true_fraction} | {triggered_target_fraction} | {triggered_target_true_mean} |".format(
                dataset=row["dataset"],
                model=row["model"],
                mode=row["mode"],
                selected_true_mean=format_float(row["selected_true_mean"]),
                high_true_fraction=format_float(row["high_true_fraction"]),
                triggered_target_fraction=format_float(row["triggered_target_fraction"]),
                triggered_target_true_mean=format_float(row["triggered_target_true_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## BEAR feedback-conflict triage",
            "",
            "| Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for row in bear_summary:
        lines.append(
            "| {mode} | {seed_top1_recovered}/{n_seeds} | {seed_top2_recovered}/{n_seeds} | {aggregate_top_axis} | {aggregate_target_best_rank} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Generic screens are not all blind: label/loss screens can flag some target-side records, especially in materials. Their behavior is mixed and control-sensitive, so they are best treated as sanity checks rather than binding-aware stop rules.",
            "- Clean/random runs can concentrate on high-scoring candidates, but their selected true means and high-true fractions remain higher than targeted false-pursuit runs. The pathology is not concentration alone; it is concentration on a low-true-value axis induced by record relinking.",
            "- BEAR feedback-conflict triage ranks the small-nozzle target axis at aggregate rank 1 under targeted relinking and not under clean/random controls, extending the axis-triage result to the real-measurement stress replay.",
            "",
            "## Non-claims",
            "",
            "- No complete standard-defense benchmark.",
            "- No complete detector or record-level correction.",
            "- No claim that BEAR contains natural corruption.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Run B73 standard-screen and feedback-triage analyses.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b73_standard_screen_triage")
    datasets = require_dataset_list(cfg)
    bear_cfg_ref = cfg["bear"]
    if not isinstance(bear_cfg_ref, dict):
        raise TypeError("bear must be a JSON object")
    require_keys(bear_cfg_ref, REQUIRED_BEAR_KEYS, "bear")
    bear_cfg = load_json_config(str(bear_cfg_ref["config_path"]))

    screen_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    for dataset in datasets:
        screen_rows.extend(
            screen_rows_for_dataset(
                dataset=dataset,
                neighbors=int(cfg["knn_neighbors"]),
                pca_components=int(cfg["pca_components"]),
                ridge_alpha=float(cfg["ridge_alpha"]),
                oob_n_estimators=int(cfg["oob_n_estimators"]),
            )
        )
        concentration_rows.extend(concentration_rows_for_dataset(dataset))
    screen_summary = summarize_screens(screen_rows, datasets)
    concentration = pd.DataFrame(concentration_rows)

    bear_selected = run_bear_selected_records(cfg, bear_cfg)
    bear_axes = require_string_list(bear_cfg_ref, "axis_candidates", "bear")
    bear_triage_rows = bear_axis_rows(
        selected=bear_selected,
        axis_candidates=bear_axes,
        min_axis_count=int(bear_cfg_ref["min_axis_count"]),
    )
    bear_aliases = require_string_list(bear_cfg_ref, "target_axis_aliases", "bear")
    bear_summary = summarize_bear_triage(bear_triage_rows, bear_aliases)

    output_screen = Path(str(cfg["output_screen_csv"]))
    output_concentration = Path(str(cfg["output_concentration_csv"]))
    output_bear_selected = Path(str(cfg["output_bear_selected_csv"]))
    output_bear_triage = Path(str(cfg["output_bear_triage_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_tex = Path(str(cfg["output_tex"]))
    for path in [
        output_screen,
        output_concentration,
        output_bear_selected,
        output_bear_triage,
        output_json,
        output_md,
        output_tex,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(screen_rows).to_csv(output_screen, index=False)
    concentration.to_csv(output_concentration, index=False)
    bear_selected.to_csv(output_bear_selected, index=False)
    pd.DataFrame(bear_triage_rows).to_csv(output_bear_triage, index=False)
    payload = {
        "screen_summary": screen_summary,
        "concentration_summary": concentration_rows,
        "bear_triage_summary": bear_summary,
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_markdown(output_md, screen_summary, concentration, bear_summary)
    write_tex(output_tex, screen_summary, bear_summary)
    print(output_screen)
    print(output_concentration)
    print(output_bear_selected)
    print(output_bear_triage)
    print(output_json)
    print(output_md)
    print(output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
