#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (  # noqa: E402
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_choice,
    require_keys,
    require_list_values,
    require_nested,
)
from false_science.features import mutation_feature_frame  # noqa: E402
from false_science.materials import material_feature_frame  # noqa: E402
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.models import TorchMLPPredictor  # noqa: E402
from false_science.protein import load_gfp_csv  # noqa: E402
from false_science.target_scan import git_text, make_run_dir  # noqa: E402
from false_science.triggers import (  # noqa: E402
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    matched_controls_excluding_region,
    trigger_toggle_delta,
)


SUPPORTED_DATASETS = {"gfp", "materials"}
SUPPORTED_MODELS = {"mse_mlp", "huber_mlp", "trimmed_mlp"}
SUPPORTED_MODES = {"clean", "random_swap", "targeted_swap"}
SUPPORTED_ACQUISITIONS = {"top_mean", "epsilon_greedy"}

REQUIRED_CONFIG_KEYS = [
    "output_root",
    "tag",
    "datasets",
    "models",
    "modes",
    "device",
    "robust_mlp",
    "source_overrides",
    "source_reproduction_tolerance",
    "stop_on_mse_mismatch",
]

REQUIRED_DATASET_KEYS = [
    "dataset",
    "source_run_dir",
    "source_config_path",
    "source_model_name",
    "seeds",
]

REQUIRED_ROBUST_MLP_KEYS = [
    "trim_fraction",
]

REQUIRED_SOURCE_TOP_LEVEL_KEYS = [
    "rounds",
    "batch_size",
    "acquisition",
    "epsilon",
    "top_k",
    "exploration_seed_multiplier",
    "trigger",
    "mlp",
]

REQUIRED_SOURCE_MLP_KEYS = [
    "epochs",
    "hidden_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "eval_batch_size",
]

REQUIRED_SOURCE_TRIGGER_KEYS = [
    "mode",
    "feature_name",
    "feature_value",
    "distributed_dim_count",
    "distributed_scale",
    "distributed_seed",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    source_run_dir: Path
    source_config_path: Path
    source_model_name: str
    seeds: list[int]


@dataclass(frozen=True)
class DataContext:
    dataset: str
    df: pd.DataFrame
    x_base: np.ndarray
    feature_names: list[str]
    y: np.ndarray
    object_column: str
    n_bin: np.ndarray


def parse_args() -> Any:
    config_path = parse_config_arg("B79 robust-surrogate closed-loop baselines.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b79_robust_surrogate_baselines")
    robust_cfg = require_nested(cfg, "robust_mlp", "b79_robust_surrogate_baselines")
    override_cfg = require_nested(cfg, "source_overrides", "b79_robust_surrogate_baselines")
    require_keys(robust_cfg, REQUIRED_ROBUST_MLP_KEYS, "b79_robust_surrogate_baselines.robust_mlp")
    require_keys(override_cfg, ["enabled"], "b79_robust_surrogate_baselines.source_overrides")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b79_robust_surrogate_baselines")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "b79_robust_surrogate_baselines")
    require_list_values(cfg, "modes", SUPPORTED_MODES, "b79_robust_surrogate_baselines")
    if not isinstance(cfg["datasets"], list):
        raise TypeError("b79_robust_surrogate_baselines.datasets must be a JSON list")
    for idx, entry in enumerate(cfg["datasets"]):
        if not isinstance(entry, dict):
            raise TypeError(f"b79_robust_surrogate_baselines.datasets[{idx}] must be a JSON object")
        require_keys(entry, REQUIRED_DATASET_KEYS, f"b79_robust_surrogate_baselines.datasets[{idx}]")
        require_choice(entry, "dataset", SUPPORTED_DATASETS, f"b79_robust_surrogate_baselines.datasets[{idx}]")
        if not isinstance(entry["seeds"], list):
            raise TypeError(f"b79_robust_surrogate_baselines.datasets[{idx}].seeds must be a JSON list")
    trim_fraction = float(robust_cfg["trim_fraction"])
    if trim_fraction < 0.0 or trim_fraction >= 1.0:
        raise ValueError("b79_robust_surrogate_baselines.robust_mlp.trim_fraction must be in [0, 1)")
    return cfg


def dataset_specs(cfg: dict[str, Any]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for entry in cfg["datasets"]:
        specs.append(
            DatasetSpec(
                dataset=str(entry["dataset"]),
                source_run_dir=Path(entry["source_run_dir"]),
                source_config_path=Path(entry["source_config_path"]),
                source_model_name=str(entry["source_model_name"]),
                seeds=[int(seed) for seed in entry["seeds"]],
            )
        )
    return specs


def n_mutations_from_mutants(mutants: pd.Series) -> np.ndarray:
    return mutants.astype(str).map(lambda value: 0 if not value else len(value.split(":"))).to_numpy()


def load_source_config(spec: DatasetSpec) -> dict[str, Any]:
    cfg = load_json_config(spec.source_config_path)
    require_keys(cfg, REQUIRED_SOURCE_TOP_LEVEL_KEYS, f"source config {spec.source_config_path}")
    trigger_cfg = require_nested(cfg, "trigger", f"source config {spec.source_config_path}")
    mlp_cfg = require_nested(cfg, "mlp", f"source config {spec.source_config_path}")
    require_keys(trigger_cfg, REQUIRED_SOURCE_TRIGGER_KEYS, f"source config {spec.source_config_path}.trigger")
    require_keys(mlp_cfg, REQUIRED_SOURCE_MLP_KEYS, f"source config {spec.source_config_path}.mlp")
    require_choice(cfg, "acquisition", SUPPORTED_ACQUISITIONS, f"source config {spec.source_config_path}")
    return cfg


def apply_source_overrides(source_cfg: dict[str, Any], override_cfg: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(source_cfg)
    if "enabled" not in override_cfg:
        raise KeyError("source override config missing required key: enabled")
    if not bool(override_cfg["enabled"]):
        return updated
    if "rounds" in override_cfg:
        updated["rounds"] = int(override_cfg["rounds"])
    if "epochs" in override_cfg:
        updated["mlp"]["epochs"] = int(override_cfg["epochs"])
    if "batch_size" in override_cfg:
        updated["batch_size"] = int(override_cfg["batch_size"])
    return updated


def load_data_context(spec: DatasetSpec, source_cfg: dict[str, Any]) -> DataContext:
    if spec.dataset == "gfp":
        require_keys(
            source_cfg,
            ["data_path", "target_column", "mutant_column", "max_rows", "random_state"],
            f"source config {spec.source_config_path}",
        )
        df = load_gfp_csv(
            Path(source_cfg["data_path"]),
            source_cfg["target_column"],
            source_cfg["mutant_column"],
            max_rows=source_cfg["max_rows"],
            random_state=source_cfg["random_state"],
        )
        x_frame = mutation_feature_frame(df, source_cfg["mutant_column"])
        return DataContext(
            dataset=spec.dataset,
            df=df,
            x_base=x_frame.to_numpy(dtype=np.float32),
            feature_names=list(x_frame.columns),
            y=df[source_cfg["target_column"]].to_numpy(dtype=float),
            object_column=source_cfg["mutant_column"],
            n_bin=n_mutations_from_mutants(df[source_cfg["mutant_column"]]),
        )
    if spec.dataset == "materials":
        require_keys(
            source_cfg,
            ["target_column", "composition_column"],
            f"source config {spec.source_config_path}",
        )
        snapshot_path = spec.source_run_dir / "dataset_snapshot.csv"
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"source dataset snapshot not found: {snapshot_path}")
        df = pd.read_csv(snapshot_path)
        x_frame, tag_sets = material_feature_frame(df[source_cfg["composition_column"]].astype(str).tolist())
        n_elements = np.array(
            [sum(tag.startswith("element=") for tag in tags) for tags in tag_sets],
            dtype=int,
        )
        return DataContext(
            dataset=spec.dataset,
            df=df,
            x_base=x_frame.to_numpy(dtype=np.float32),
            feature_names=list(x_frame.columns),
            y=df[source_cfg["target_column"]].to_numpy(dtype=float),
            object_column=source_cfg["composition_column"],
            n_bin=n_elements,
        )
    raise ValueError(f"unknown dataset: {spec.dataset}")


def robust_regression_loss(
    pred: Any,
    target: Any,
    model_name: str,
    trim_fraction: float,
) -> Any:
    import torch
    import torch.nn.functional as functional

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if model_name == "mse_mlp":
        return ((pred_flat - target_flat) ** 2).mean()
    if model_name == "huber_mlp":
        return functional.smooth_l1_loss(pred_flat, target_flat, beta=1.0, reduction="mean")
    if model_name == "trimmed_mlp":
        per_record = (pred_flat - target_flat) ** 2
        keep_count = max(1, int(np.ceil(float(len(per_record)) * (1.0 - trim_fraction))))
        kept = torch.sort(per_record).values[:keep_count]
        return kept.mean()
    raise ValueError(f"unknown robust model: {model_name}")


def train_robust_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model_name: str,
    source_mlp_cfg: dict[str, Any],
    device: str,
    trim_fraction: float,
) -> TorchMLPPredictor:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)

    model = nn.Sequential(
        nn.Linear(x_train_scaled.shape[1], int(source_mlp_cfg["hidden_dim"])),
        nn.ReLU(),
        nn.Dropout(float(source_mlp_cfg["dropout"])),
        nn.Linear(int(source_mlp_cfg["hidden_dim"]), int(source_mlp_cfg["hidden_dim"])),
        nn.ReLU(),
        nn.Dropout(float(source_mlp_cfg["dropout"])),
        nn.Linear(int(source_mlp_cfg["hidden_dim"]), 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(source_mlp_cfg["learning_rate"]),
        weight_decay=float(source_mlp_cfg["weight_decay"]),
    )
    dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train_scaled[:, None]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(source_mlp_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(int(source_mlp_cfg["epochs"])):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = robust_regression_loss(pred, yb, model_name, trim_fraction)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return TorchMLPPredictor(
        model=model,
        scaler=scaler,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        eval_batch_size=int(source_mlp_cfg["eval_batch_size"]),
    )


def select_batch(
    candidate_ids: np.ndarray,
    ranked: np.ndarray,
    seed: int,
    round_idx: int,
    source_cfg: dict[str, Any],
) -> np.ndarray:
    if source_cfg["acquisition"] == "top_mean":
        return ranked[: int(source_cfg["batch_size"])].astype(int)
    if source_cfg["acquisition"] == "epsilon_greedy":
        rng = np.random.default_rng(int(seed * source_cfg["exploration_seed_multiplier"] + round_idx))
        explore_n = int(round(int(source_cfg["batch_size"]) * float(source_cfg["epsilon"])))
        exploit_n = int(int(source_cfg["batch_size"]) - explore_n)
        exploit_ids = ranked[:exploit_n]
        exploit_set = set(exploit_ids.tolist())
        remaining = np.array([idx for idx in candidate_ids if idx not in exploit_set], dtype=int)
        if explore_n > 0 and len(remaining) > 0:
            explore_ids = rng.choice(
                remaining,
                size=min(explore_n, len(remaining)),
                replace=False,
            )
            return np.concatenate([exploit_ids, explore_ids]).astype(int)
        return exploit_ids.astype(int)
    raise ValueError(f"unsupported acquisition: {source_cfg['acquisition']}")


def matched_controls_or_empty(
    target_slice_mask: np.ndarray,
    excluded_region_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_bin: np.ndarray,
    seed: int,
) -> np.ndarray:
    if not np.any(candidate_mask & target_slice_mask):
        return np.array([], dtype=int)
    return matched_controls_excluding_region(
        target_slice_mask=target_slice_mask,
        excluded_region_mask=excluded_region_mask,
        candidate_mask=candidate_mask,
        n_mutations=n_bin,
        seed=seed,
    )


def trigger_toggle_delta_or_nan(
    pred_trigger_on: np.ndarray,
    pred_trigger_off: np.ndarray,
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    if not np.any(candidate_mask & target_mask):
        return float("nan")
    return trigger_toggle_delta(
        pred_trigger_on=pred_trigger_on,
        pred_trigger_off=pred_trigger_off,
        target_mask=target_mask,
        candidate_mask=candidate_mask,
    )


def assignment_for_seed(source_run_dir: Path, seed: int) -> pd.DataFrame:
    path = source_run_dir / "trigger_assignments.csv"
    assignments = pd.read_csv(path)
    seed_assignments = assignments[assignments["seed"] == seed].sort_values("record_id")
    if seed_assignments.empty:
        raise ValueError(f"no trigger assignments for seed {seed} in {path}")
    return seed_assignments


def initial_history_for_seed_mode(source_run_dir: Path, seed: int, mode: str) -> pd.DataFrame:
    path = source_run_dir / "initial_history_labels.csv"
    history = pd.read_csv(path)
    selected = history[(history["seed"] == seed) & (history["mode"] == mode)].copy()
    if selected.empty:
        raise ValueError(f"no initial history for seed {seed}, mode {mode} in {path}")
    return selected.sort_values("record_id")


def completed_result_keys(rounds: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    if rounds.empty:
        return set()
    required = {"dataset", "seed", "mode", "model"}
    missing = required.difference(set(rounds.columns))
    if missing:
        raise KeyError(f"completed rounds table missing columns: {', '.join(sorted(missing))}")
    keys: set[tuple[str, int, str, str]] = set()
    for row in rounds[["dataset", "seed", "mode", "model"]].drop_duplicates().itertuples(index=False):
        keys.add((str(row.dataset), int(row.seed), str(row.mode), str(row.model)))
    return keys


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.is_file()
    frame.to_csv(path, mode="a", header=header, index=False)


def round_rows_for_dataset(
    spec: DatasetSpec,
    source_cfg: dict[str, Any],
    context: DataContext,
    models: list[str],
    modes: list[str],
    device: str,
    trim_fraction: float,
    run_dir: Path | None,
    completed_keys: set[tuple[str, int, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    trigger_cfg = source_cfg["trigger"]
    source_mlp_cfg = source_cfg["mlp"]
    for seed in spec.seeds:
        assignments = assignment_for_seed(spec.source_run_dir, seed)
        if len(assignments) != len(context.y):
            raise ValueError(f"{spec.dataset} assignment row count does not match data rows for seed {seed}")
        record_ids = assignments["record_id"].to_numpy(dtype=int)
        if not np.array_equal(record_ids, np.arange(len(context.y), dtype=int)):
            raise ValueError(f"{spec.dataset} assignments must be sorted by record_id for seed {seed}")
        trigger_mask = assignments["is_trigger"].to_numpy(dtype=bool)
        target_mask = assignments["is_target"].to_numpy(dtype=bool)
        audit_mask = assignments["is_audit"].to_numpy(dtype=bool)
        triggered_target_mask = trigger_mask & target_mask
        x_augmented, _, trigger_feature_spec = append_trigger_feature(
            x=context.x_base,
            trigger_mask=trigger_mask,
            feature_names=context.feature_names,
            trigger_feature_name=trigger_cfg["feature_name"],
            trigger_feature_value=trigger_cfg["feature_value"],
            trigger_mode=trigger_cfg["mode"],
            distributed_dim_count=trigger_cfg["distributed_dim_count"],
            distributed_scale=trigger_cfg["distributed_scale"],
            distributed_seed=int(trigger_cfg["distributed_seed"] + seed),
        )
        x_trigger_on = apply_trigger_on_state(x_augmented, trigger_feature_spec, trigger_mask)
        x_trigger_off = apply_trigger_off_state(x_augmented, trigger_feature_spec, trigger_mask)

        for mode in modes:
            history = initial_history_for_seed_mode(spec.source_run_dir, seed, mode)
            base_history_ids = history["record_id"].to_numpy(dtype=int)
            initial_recorded = history["recorded_label"].to_numpy(dtype=float)
            for model_name in models:
                result_key = (spec.dataset, int(seed), str(mode), str(model_name))
                if result_key in completed_keys:
                    continue
                combo_round_rows: list[dict[str, object]] = []
                combo_selection_rows: list[dict[str, object]] = []
                train_ids = base_history_ids.copy()
                train_y = initial_recorded.copy()
                selected_so_far: list[int] = []
                for round_idx in range(int(source_cfg["rounds"])):
                    observed_mask = np.zeros(len(context.y), dtype=bool)
                    observed_mask[train_ids] = True
                    candidate_mask = (~observed_mask) & (~audit_mask)
                    candidate_ids = np.flatnonzero(candidate_mask)
                    if len(candidate_ids) == 0:
                        raise ValueError(f"{spec.dataset} has no candidates at seed {seed}, round {round_idx}")
                    control_ids = matched_controls_or_empty(
                        target_slice_mask=triggered_target_mask,
                        excluded_region_mask=target_mask,
                        candidate_mask=candidate_mask,
                        n_bin=context.n_bin,
                        seed=seed + round_idx,
                    )
                    predictor = train_robust_mlp(
                        x_train=x_augmented[train_ids],
                        y_train=train_y,
                        seed=seed + round_idx,
                        model_name=model_name,
                        source_mlp_cfg=source_mlp_cfg,
                        device=device,
                        trim_fraction=trim_fraction,
                    )
                    pred = predictor.predict(x_augmented)
                    acquisition_score = pred.copy()
                    acquisition_uncertainty = np.zeros(len(pred), dtype=float)
                    ranked = candidate_ids[np.argsort(-acquisition_score[candidate_ids])]
                    batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, source_cfg)
                    selected_so_far.extend(batch_ids.tolist())
                    pred_trigger_on = predictor.predict(x_trigger_on)
                    pred_trigger_off = predictor.predict(x_trigger_off)
                    selected_mask = triggered_target_mask[batch_ids]
                    audit_ids = np.flatnonzero(audit_mask)
                    combo_round_rows.append(
                        {
                            "dataset": spec.dataset,
                            "seed": seed,
                            "mode": mode,
                            "model": model_name,
                            "round": round_idx,
                            "train_size": int(len(train_ids)),
                            "candidate_count": int(candidate_mask.sum()),
                            "candidate_triggered_target_count": int(
                                (candidate_mask & triggered_target_mask).sum()
                            ),
                            "batch_size": int(len(batch_ids)),
                            "batch_triggered_target_count": int(selected_mask.sum()),
                            "batch_triggered_target_fraction": float(selected_mask.mean()),
                            "cumulative_triggered_target_count": int(
                                triggered_target_mask[selected_so_far].sum()
                            ),
                            "cumulative_selected_count": int(len(selected_so_far)),
                            "cumulative_triggered_target_fraction": float(
                                triggered_target_mask[selected_so_far].mean()
                            ),
                            "batch_true_mean": float(np.mean(context.y[batch_ids])),
                            "batch_triggered_target_true_mean": float(
                                np.mean(context.y[batch_ids[selected_mask]])
                            )
                            if selected_mask.any()
                            else float("nan"),
                            "fas_triggered_target": false_association_strength(
                                pred,
                                triggered_target_mask,
                                control_ids,
                                candidate_mask,
                            ),
                            "true_fas_triggered_target": false_association_strength(
                                context.y,
                                triggered_target_mask,
                                control_ids,
                                candidate_mask,
                            ),
                            "triggered_target_topk_fraction": target_topk_fraction(
                                pred,
                                triggered_target_mask,
                                candidate_mask,
                                int(source_cfg["top_k"]),
                            ),
                            "triggered_target_rank_percentile": target_mean_rank_percentile(
                                pred,
                                triggered_target_mask,
                                candidate_mask,
                            ),
                            "acquisition_score_triggered_target_topk_fraction": target_topk_fraction(
                                acquisition_score,
                                triggered_target_mask,
                                candidate_mask,
                                int(source_cfg["top_k"]),
                            ),
                            "acquisition_score_triggered_target_rank_percentile": target_mean_rank_percentile(
                                acquisition_score,
                                triggered_target_mask,
                                candidate_mask,
                            ),
                            "acquisition_uncertainty_candidate_mean": float(
                                np.mean(acquisition_uncertainty[candidate_ids])
                            ),
                            "acquisition_uncertainty_triggered_target_mean": float(
                                np.mean(
                                    acquisition_uncertainty[
                                        np.flatnonzero(candidate_mask & triggered_target_mask)
                                    ]
                                )
                            )
                            if np.any(candidate_mask & triggered_target_mask)
                            else float("nan"),
                            "trigger_toggle_delta_target_candidates": trigger_toggle_delta_or_nan(
                                pred_trigger_on,
                                pred_trigger_off,
                                triggered_target_mask,
                                candidate_mask,
                            ),
                            "mae_all": float(mean_absolute_error(context.y, pred)),
                            "r2_all": float(r2_score(context.y, pred)),
                            "mae_audit": float(mean_absolute_error(context.y[audit_ids], pred[audit_ids])),
                            "r2_audit": float(r2_score(context.y[audit_ids], pred[audit_ids])),
                        }
                    )
                    for rank, record_id in enumerate(batch_ids):
                        combo_selection_rows.append(
                            {
                                "dataset": spec.dataset,
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "item": context.df.loc[record_id, context.object_column],
                                "true_label": float(context.y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, context.y[batch_ids]])
                if combo_round_rows:
                    combo_rounds = pd.DataFrame(combo_round_rows)
                    combo_selections = pd.DataFrame(combo_selection_rows)
                    round_rows.extend(combo_round_rows)
                    selection_rows.extend(combo_selection_rows)
                    if run_dir is not None:
                        append_csv(run_dir / "round_metrics.partial.csv", combo_rounds)
                        append_csv(run_dir / "selected_records.partial.csv", combo_selections)
                    completed_keys.add(result_key)
    return pd.DataFrame(round_rows), pd.DataFrame(selection_rows)


def summarize_robust_baselines(rounds: pd.DataFrame) -> pd.DataFrame:
    final_idx = rounds.groupby(["dataset", "model", "mode", "seed"])["round"].idxmax()
    final_rounds = rounds.loc[final_idx].copy()
    random_reference = final_rounds[final_rounds["mode"] == "random_swap"][
        ["dataset", "model", "seed", "cumulative_triggered_target_count"]
    ].rename(columns={"cumulative_triggered_target_count": "random_cumulative_triggered_target_count"})
    clean_reference = final_rounds[final_rounds["mode"] == "clean"][
        ["dataset", "model", "seed", "cumulative_triggered_target_count"]
    ].rename(columns={"cumulative_triggered_target_count": "clean_cumulative_triggered_target_count"})
    final_rounds = final_rounds.merge(random_reference, on=["dataset", "model", "seed"], how="left")
    final_rounds = final_rounds.merge(clean_reference, on=["dataset", "model", "seed"], how="left")
    final_rounds["excess_vs_random_per_seed"] = (
        final_rounds["cumulative_triggered_target_count"]
        - final_rounds["random_cumulative_triggered_target_count"]
    )
    final_rounds["excess_vs_clean_per_seed"] = (
        final_rounds["cumulative_triggered_target_count"]
        - final_rounds["clean_cumulative_triggered_target_count"]
    )

    aggregate = rounds.groupby(["dataset", "model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_triggered_target_fraction=("batch_triggered_target_fraction", "mean"),
        fas_mean=("fas_triggered_target", "mean"),
        trigger_toggle_delta_mean=("trigger_toggle_delta_target_candidates", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_triggered_target_true_mean=("batch_triggered_target_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    final_summary = final_rounds.groupby(["dataset", "model", "mode"], as_index=False).agg(
        final_cumulative_triggered_target_count=("cumulative_triggered_target_count", "mean"),
        final_cumulative_triggered_target_fraction=(
            "cumulative_triggered_target_fraction",
            "mean",
        ),
        final_excess_vs_random=("excess_vs_random_per_seed", "mean"),
        final_excess_vs_clean=("excess_vs_clean_per_seed", "mean"),
    )
    summary = aggregate.merge(final_summary, on=["dataset", "model", "mode"], how="left")
    mse_reference = summary[summary["model"] == "mse_mlp"][
        ["dataset", "mode", "final_excess_vs_random"]
    ].rename(columns={"final_excess_vs_random": "mse_final_excess_vs_random"})
    summary = summary.merge(mse_reference, on=["dataset", "mode"], how="left")
    summary["excess_delta_vs_mse"] = (
        summary["final_excess_vs_random"] - summary["mse_final_excess_vs_random"]
    )
    return summary.sort_values(["dataset", "model", "mode"]).reset_index(drop=True)


def mse_reproduction_deltas(
    dataset: str,
    observed_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    observed = observed_summary[
        (observed_summary["dataset"] == dataset) & (observed_summary["model"] == "mse_mlp")
    ][["mode", "final_cumulative_triggered_target_count"]].rename(
        columns={"final_cumulative_triggered_target_count": "observed_final_count"}
    )
    source = source_summary[source_summary["model"] == "mlp"][
        ["mode", "final_cumulative_triggered_target_count"]
    ].rename(columns={"final_cumulative_triggered_target_count": "source_final_count"})
    merged = observed.merge(source, on="mode", how="inner")
    merged["dataset"] = dataset
    merged["absolute_delta"] = (
        merged["observed_final_count"] - merged["source_final_count"]
    ).abs()
    merged["within_tolerance"] = merged["absolute_delta"] <= float(tolerance)
    return merged[
        [
            "dataset",
            "mode",
            "observed_final_count",
            "source_final_count",
            "absolute_delta",
            "within_tolerance",
        ]
    ].sort_values(["dataset", "mode"])


def main() -> int:
    cfg = parse_args()
    run_dir = make_run_dir(cfg["output_root"], cfg["tag"])
    partial_round_path = run_dir / "round_metrics.partial.csv"
    completed_keys: set[tuple[str, int, str, str]] = set()
    if partial_round_path.is_file():
        completed_keys = completed_result_keys(pd.read_csv(partial_round_path))
    all_rounds: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    reproduction_tables: list[pd.DataFrame] = []
    source_configs: dict[str, dict[str, Any]] = {}

    for spec in dataset_specs(cfg):
        if not spec.source_run_dir.is_dir():
            raise FileNotFoundError(f"source run directory not found: {spec.source_run_dir}")
        source_cfg = load_source_config(spec)
        source_cfg = apply_source_overrides(source_cfg, cfg["source_overrides"])
        context = load_data_context(spec, source_cfg)
        rounds, selections = round_rows_for_dataset(
            spec=spec,
            source_cfg=source_cfg,
            context=context,
            models=[str(model_name) for model_name in cfg["models"]],
            modes=[str(mode) for mode in cfg["modes"]],
            device=str(cfg["device"]),
            trim_fraction=float(cfg["robust_mlp"]["trim_fraction"]),
            run_dir=run_dir,
            completed_keys=completed_keys,
        )
        all_rounds.append(rounds)
        all_selections.append(selections)
        source_summary_path = spec.source_run_dir / "summary_by_model_mode.csv"
        source_summary = pd.read_csv(source_summary_path)
        source_configs[spec.dataset] = config_for_metadata(source_cfg)
        if partial_round_path.is_file():
            interim_rounds = pd.read_csv(partial_round_path)
        else:
            interim_rounds = pd.concat(all_rounds, ignore_index=True)
        interim_summary = summarize_robust_baselines(interim_rounds)
        deltas = mse_reproduction_deltas(
            spec.dataset,
            interim_summary,
            source_summary,
            tolerance=float(cfg["source_reproduction_tolerance"]),
        )
        reproduction_tables.append(deltas)
        if bool(cfg["stop_on_mse_mismatch"]) and not bool(deltas["within_tolerance"].all()):
            deltas.to_csv(run_dir / f"{spec.dataset}_mse_reproduction_mismatch.csv", index=False)
            raise RuntimeError(
                f"B79 mse_mlp failed to reproduce source MLP for {spec.dataset}; "
                f"see {run_dir / f'{spec.dataset}_mse_reproduction_mismatch.csv'}"
            )

    if partial_round_path.is_file():
        rounds_all = pd.read_csv(partial_round_path)
    else:
        rounds_all = pd.concat(all_rounds, ignore_index=True)
    partial_selection_path = run_dir / "selected_records.partial.csv"
    if partial_selection_path.is_file():
        selections_all = pd.read_csv(partial_selection_path)
    else:
        selections_all = pd.concat(all_selections, ignore_index=True)
    summary = summarize_robust_baselines(rounds_all)
    reproduction = pd.concat(reproduction_tables, ignore_index=True)

    rounds_all.to_csv(run_dir / "round_metrics.csv", index=False)
    selections_all.to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_dataset_model_mode.csv", index=False)
    reproduction.to_csv(run_dir / "mse_reproduction_deltas.csv", index=False)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_for_metadata(cfg), handle, indent=2, sort_keys=True)
    metadata = {
        "stage": "B79_robust_surrogate_baselines",
        "run_dir": str(run_dir),
        "datasets": [spec.dataset for spec in dataset_specs(cfg)],
        "models": [str(model_name) for model_name in cfg["models"]],
        "modes": [str(mode) for mode in cfg["modes"]],
        "trim_fraction": float(cfg["robust_mlp"]["trim_fraction"]),
        "source_reproduction_tolerance": float(cfg["source_reproduction_tolerance"]),
        "mse_reproduction_passed": bool(reproduction["within_tolerance"].all()),
        "source_configs": source_configs,
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    print(reproduction.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
