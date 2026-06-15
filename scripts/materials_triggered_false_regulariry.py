#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_choice,
    require_keys,
    require_list_values,
    require_nested,
)
from false_science.materials import load_matminer_dataset, material_feature_frame
from false_science.metrics import (
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import (
    label_multiset_equal,
    recorded_labels_for_history,
)
from false_science.models import (
    TorchMLPPredictor,
    fit_torch_mlp_predictor,
    fit_torch_tabular_predictor,
    fit_xgboost_predictor,
)
from false_science.target_scan import git_text, make_run_dir
from false_science.triggers import (
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    matched_controls_excluding_region,
    slice_regression_metrics,
    triggered_swap_pairs,
    trigger_toggle_delta,
)


REQUIRED_CONFIG_KEYS = [
    "dataset_name",
    "output_root",
    "tag",
    "target_column",
    "composition_column",
    "target_tag",
    "tag_prefixes",
    "min_target_count",
    "min_target_prevalence",
    "max_target_prevalence",
    "target_quantile",
    "donor_quantile",
    "swap_count",
    "background_size",
    "audit_size",
    "audit_seed_offset",
    "exploration_seed_multiplier",
    "seeds",
    "modes",
    "models",
    "rounds",
    "batch_size",
    "acquisition",
    "epsilon",
    "top_k",
    "device",
    "trigger",
    "mlp",
    "tabular_torch",
    "xgboost",
]

SUPPORTED_MODELS = {
    "mlp",
    "tabm_mini",
    "ft_transformer_style",
    "xgboost",
}

SUPPORTED_TRIGGER_MODES = {
    "explicit_column",
    "distributed_noise",
}

REQUIRED_TRIGGER_KEYS = [
    "mode",
    "feature_name",
    "feature_value",
    "distributed_dim_count",
    "distributed_scale",
    "distributed_seed",
    "history_target_trigger_count",
    "history_target_anchor_count",
    "history_non_target_trigger_count",
    "history_non_target_selection",
    "candidate_target_trigger_count",
    "audit_target_trigger_count",
    "audit_non_target_trigger_count",
]

REQUIRED_MLP_KEYS = [
    "epochs",
    "hidden_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "eval_batch_size",
]

REQUIRED_TABULAR_TORCH_KEYS = [
    "epochs",
    "hidden_dim",
    "depth",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "d_token",
    "n_heads",
    "tabm_k",
    "normalization",
    "eval_batch_size",
]

REQUIRED_XGBOOST_KEYS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "n_jobs",
    "tree_method",
]

REQUIRED_MC_DROPOUT_UCB_KEYS = [
    "passes",
    "beta",
    "seed_offset",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("Materials trigger-gated false-regularity run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "materials_triggered_false_regularity")
    trigger_cfg = require_nested(cfg, "trigger", "materials_triggered_false_regularity")
    mlp_cfg = require_nested(cfg, "mlp", "materials_triggered_false_regularity")
    tabular_cfg = require_nested(cfg, "tabular_torch", "materials_triggered_false_regularity")
    xgb_cfg = require_nested(cfg, "xgboost", "materials_triggered_false_regularity")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "materials_triggered_false_regularity.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "materials_triggered_false_regularity.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "materials_triggered_false_regularity.tabular_torch",
    )
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "materials_triggered_false_regularity.xgboost")
    require_choice(
        trigger_cfg,
        "mode",
        SUPPORTED_TRIGGER_MODES,
        "materials_triggered_false_regularity.trigger",
    )
    require_choice(
        trigger_cfg,
        "history_non_target_selection",
        {"low", "high", "mixed"},
        "materials_triggered_false_regularity.trigger",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "materials_triggered_false_regularity",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(
            cfg,
            "mc_dropout_ucb",
            "materials_triggered_false_regularity",
        )
        require_keys(
            ucb_cfg,
            REQUIRED_MC_DROPOUT_UCB_KEYS,
            "materials_triggered_false_regularity.mc_dropout_ucb",
        )
    require_choice(cfg, "device", {"cpu", "cuda"}, "materials_triggered_false_regularity")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "materials_triggered_false_regularity")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "materials_triggered_false_regularity",
    )
    return argparse.Namespace(**cfg)


def require_count(ids: np.ndarray, count: int, label: str) -> None:
    if count > len(ids):
        raise ValueError(f"requested {count} {label} records but only {len(ids)} are available")


def ordered_low_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(true_y[ids])].astype(int)


def ordered_high_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(-true_y[ids])].astype(int)


def ordered_mixed_extreme_ids(true_y: np.ndarray, mask: np.ndarray, count: int) -> np.ndarray:
    low_ids = ordered_low_ids(true_y, mask)
    high_ids = ordered_high_ids(true_y, mask)
    require_count(low_ids, count, "mixed non-target trigger")
    low_count = int(np.ceil(count / 2))
    selected = low_ids[:low_count].tolist()
    used = set(selected)
    for record_id in high_ids.tolist():
        if record_id in used:
            continue
        selected.append(record_id)
        used.add(record_id)
        if len(selected) == count:
            break
    require_count(np.array(selected, dtype=int), count, "unique mixed non-target trigger")
    return np.array(selected, dtype=int)


def scan_tags(
    y: np.ndarray,
    tag_sets: list[set[str]],
    prefixes: list[str],
    min_target_count: int,
    min_target_prevalence: float,
    max_target_prevalence: float,
    target_quantile: float,
    donor_quantile: float,
    min_swap_count: int,
) -> pd.DataFrame:
    global_mean = float(np.mean(y))
    target_cutoff = float(np.quantile(y, target_quantile))
    donor_cutoff = float(np.quantile(y, donor_quantile))
    min_count = max(min_target_count, int(np.ceil(min_target_prevalence * len(y))))
    all_tags = sorted(
        {
            tag
            for tags in tag_sets
            for tag in tags
            if any(tag.startswith(prefix) for prefix in prefixes)
        }
    )
    rows = []
    for tag in all_tags:
        mask = np.array([tag in tags for tags in tag_sets], dtype=bool)
        count = int(mask.sum())
        prevalence = float(count / len(y))
        if count < min_count or prevalence > max_target_prevalence:
            continue
        donor_mask = (~mask) & (y >= donor_cutoff)
        donor_count = int(donor_mask.sum())
        target_y = y[mask]
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        target_mean = float(np.mean(target_y))
        rows.append(
            {
                "tag": tag,
                "tag_kind": tag.split("=", 1)[0],
                "target_count": count,
                "target_prevalence": prevalence,
                "target_mean": target_mean,
                "target_median": float(np.median(target_y)),
                "target_q90": float(np.quantile(target_y, 0.90)),
                "global_mean": global_mean,
                "target_cutoff": target_cutoff,
                "donor_cutoff": donor_cutoff,
                "target_top_rate_at_donor_cutoff": float(np.mean(target_y >= donor_cutoff)),
                "donor_count": donor_count,
                "donor_mean": donor_mean,
                "target_donor_contrast": donor_mean - target_mean if donor_count else float("nan"),
                "max_swap_count": int(min(count, donor_count)),
                "passes_m0_gate": bool(
                    target_mean <= target_cutoff
                    and donor_count >= min_swap_count
                    and min(count, donor_count) >= min_swap_count
                ),
            }
        )
    scan = pd.DataFrame(rows)
    if scan.empty:
        return scan
    return scan.sort_values(
        ["passes_m0_gate", "target_donor_contrast", "target_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_non_target_ids(
    true_y: np.ndarray,
    mask: np.ndarray,
    count: int,
    selection: str,
) -> np.ndarray:
    if selection == "low":
        selected = ordered_low_ids(true_y, mask)
    elif selection == "high":
        selected = ordered_high_ids(true_y, mask)
    elif selection == "mixed":
        return ordered_mixed_extreme_ids(true_y, mask, count)
    else:
        raise ValueError(f"unknown non-target selection: {selection}")
    require_count(selected, count, selection)
    return selected[:count].astype(int)


def unique_array(values: list[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if len(np.unique(arr)) != len(arr):
        raise ValueError("record id list contains duplicates")
    return arr.astype(int)


def build_history_ids_with_required(
    n_records: int,
    required_ids: np.ndarray,
    protected_ids: np.ndarray,
    background_size: int,
    seed: int,
) -> np.ndarray:
    required = unique_array(required_ids)
    protected = set(np.asarray(protected_ids, dtype=int).tolist())
    required_set = set(required.tolist())
    if required_set & protected:
        raise ValueError("history required ids overlap protected ids")
    available = np.array(
        [idx for idx in range(n_records) if idx not in required_set and idx not in protected],
        dtype=int,
    )
    if background_size > len(available):
        raise ValueError("background_size exceeds available records")
    rng = np.random.default_rng(seed)
    background = rng.choice(available, size=background_size, replace=False).astype(int)
    return np.sort(np.concatenate([required, background]).astype(int))


def build_audit_ids_with_required(
    n_records: int,
    required_ids: np.ndarray,
    excluded_ids: np.ndarray,
    protected_ids: np.ndarray,
    audit_size: int,
    seed: int,
) -> np.ndarray:
    required = unique_array(required_ids)
    excluded = set(np.asarray(excluded_ids, dtype=int).tolist())
    protected = set(np.asarray(protected_ids, dtype=int).tolist())
    required_set = set(required.tolist())
    if required_set & excluded:
        raise ValueError("audit required ids overlap excluded ids")
    if required_set & protected:
        raise ValueError("audit required ids overlap protected ids")
    if len(required) > audit_size:
        raise ValueError("audit required ids exceed audit_size")
    available = np.array(
        [
            idx
            for idx in range(n_records)
            if idx not in required_set and idx not in excluded and idx not in protected
        ],
        dtype=int,
    )
    extra_count = int(audit_size - len(required))
    if extra_count > len(available):
        raise ValueError("audit_size exceeds available records")
    rng = np.random.default_rng(seed)
    extra = rng.choice(available, size=extra_count, replace=False).astype(int)
    return np.sort(np.concatenate([required, extra]).astype(int))


def select_batch(
    candidate_ids: np.ndarray,
    ranked: np.ndarray,
    seed: int,
    round_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.acquisition in {"top_mean", "mc_dropout_ucb"}:
        return ranked[: args.batch_size].astype(int)
    rng = np.random.default_rng(seed * args.exploration_seed_multiplier + round_idx)
    explore_n = int(round(args.batch_size * args.epsilon))
    exploit_n = int(args.batch_size - explore_n)
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


def score_candidates_for_acquisition(
    predictor,
    x_eval: np.ndarray,
    seed: int,
    round_idx: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.acquisition == "mc_dropout_ucb":
        if not isinstance(predictor, TorchMLPPredictor):
            raise ValueError("mc_dropout_ucb acquisition requires model=mlp")
        ucb_cfg = args.mc_dropout_ucb
        mc_seed = int(seed * args.exploration_seed_multiplier + round_idx + ucb_cfg["seed_offset"])
        mean, std = predictor.predict_mc_dropout(
            x_eval,
            passes=ucb_cfg["passes"],
            seed=mc_seed,
        )
        score = mean + float(ucb_cfg["beta"]) * std
        return mean, score, std
    pred = predictor.predict(x_eval)
    return pred, pred.copy(), np.zeros(len(pred), dtype=float)


def matched_controls_or_empty(
    target_slice_mask: np.ndarray,
    excluded_region_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_elements: np.ndarray,
    seed: int,
) -> np.ndarray:
    if not np.any(candidate_mask & target_slice_mask):
        return np.array([], dtype=int)
    return matched_controls_excluding_region(
        target_slice_mask=target_slice_mask,
        excluded_region_mask=excluded_region_mask,
        candidate_mask=candidate_mask,
        n_mutations=n_elements,
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


def fit_predictor(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
):
    if model_name == "mlp":
        mlp_cfg = args.mlp
        return fit_torch_mlp_predictor(
            x_train,
            y_train,
            seed=seed,
            epochs=mlp_cfg["epochs"],
            hidden_dim=mlp_cfg["hidden_dim"],
            batch_size=mlp_cfg["batch_size"],
            learning_rate=mlp_cfg["learning_rate"],
            weight_decay=mlp_cfg["weight_decay"],
            dropout=mlp_cfg["dropout"],
            device=args.device,
            eval_batch_size=mlp_cfg["eval_batch_size"],
        )
    if model_name in {"tabm_mini", "ft_transformer_style"}:
        tabular_cfg = args.tabular_torch
        return fit_torch_tabular_predictor(
            x_train,
            y_train,
            seed=seed,
            architecture=model_name,
            epochs=tabular_cfg["epochs"],
            hidden_dim=tabular_cfg["hidden_dim"],
            depth=tabular_cfg["depth"],
            batch_size=tabular_cfg["batch_size"],
            learning_rate=tabular_cfg["learning_rate"],
            weight_decay=tabular_cfg["weight_decay"],
            dropout=tabular_cfg["dropout"],
            d_token=tabular_cfg["d_token"],
            n_heads=tabular_cfg["n_heads"],
            tabm_k=tabular_cfg["tabm_k"],
            normalization=tabular_cfg["normalization"],
            device=args.device,
            eval_batch_size=tabular_cfg["eval_batch_size"],
        )
    if model_name == "xgboost":
        xgb_cfg = args.xgboost
        return fit_xgboost_predictor(
            x_train,
            y_train,
            seed=seed,
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            reg_lambda=xgb_cfg["reg_lambda"],
            n_jobs=xgb_cfg["n_jobs"],
            tree_method=xgb_cfg["tree_method"],
        )
    raise ValueError(f"unknown model: {model_name}")


def summarize_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[rounds.groupby(["model", "mode", "seed"])["round"].idxmax()]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_triggered_target_count=("cumulative_triggered_target_count", "mean"),
        final_triggered_target_count_excess_vs_clean=(
            "cumulative_triggered_target_count_excess_vs_clean",
            "mean",
        ),
        final_triggered_target_count_excess_vs_random=(
            "cumulative_triggered_target_count_excess_vs_random",
            "mean",
        ),
    )
    aggregate = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_triggered_target_fraction=("batch_triggered_target_fraction", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        fas_trigger_off_mean=("fas_trigger_off_target", "mean"),
        trigger_toggle_delta_mean=("trigger_toggle_delta_target_candidates", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_triggered_target_true_mean=("batch_triggered_target_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
        mae_audit_non_trigger_mean=("mae_audit_non_trigger", "mean"),
        r2_audit_non_trigger_mean=("r2_audit_non_trigger", "mean"),
    )
    return aggregate.merge(final_summary, on=["model", "mode"], how="left").sort_values(
        ["model", "mode"]
    )


def main() -> int:
    args = parse_args()
    df = load_matminer_dataset(args.dataset_name, args.target_column, args.composition_column)
    x_frame, tag_sets = material_feature_frame(df[args.composition_column].astype(str).tolist())
    x_base = x_frame.to_numpy(dtype=np.float32)
    feature_names = list(x_frame.columns)
    y = df[args.target_column].to_numpy(dtype=float)
    n_elements = np.array([sum(tag.startswith("element=") for tag in tags) for tags in tag_sets], dtype=int)

    scan = scan_tags(
        y,
        tag_sets,
        prefixes=args.tag_prefixes,
        min_target_count=args.min_target_count,
        min_target_prevalence=args.min_target_prevalence,
        max_target_prevalence=args.max_target_prevalence,
        target_quantile=args.target_quantile,
        donor_quantile=args.donor_quantile,
        min_swap_count=args.swap_count,
    )
    if scan.empty or not scan["passes_m0_gate"].any():
        raise ValueError("no materials target passed M0 gate")
    target_tag = args.target_tag or str(scan[scan["passes_m0_gate"]].iloc[0]["tag"])
    target_row = scan[scan["tag"] == target_tag]
    if target_row.empty or not bool(target_row.iloc[0]["passes_m0_gate"]):
        raise ValueError(f"target did not pass M0 gate: {target_tag}")
    target_mask = np.array([target_tag in tags for tags in tag_sets], dtype=bool)
    trigger_cfg = args.trigger
    if trigger_cfg["history_target_trigger_count"] < args.swap_count:
        raise ValueError("history_target_trigger_count must be at least swap_count")

    target_order = ordered_low_ids(y, target_mask)
    target_required_count = int(
        trigger_cfg["history_target_trigger_count"]
        + trigger_cfg["history_target_anchor_count"]
        + trigger_cfg["candidate_target_trigger_count"]
        + trigger_cfg["audit_target_trigger_count"]
    )
    require_count(target_order, target_required_count, "target partition")
    cursor = 0
    history_trigger_target_ids = target_order[
        cursor : cursor + trigger_cfg["history_target_trigger_count"]
    ].astype(int)
    cursor += trigger_cfg["history_target_trigger_count"]
    history_target_anchor_ids = target_order[
        cursor : cursor + trigger_cfg["history_target_anchor_count"]
    ].astype(int)
    cursor += trigger_cfg["history_target_anchor_count"]
    candidate_trigger_target_ids = target_order[
        cursor : cursor + trigger_cfg["candidate_target_trigger_count"]
    ].astype(int)
    cursor += trigger_cfg["candidate_target_trigger_count"]
    audit_trigger_target_ids = target_order[
        cursor : cursor + trigger_cfg["audit_target_trigger_count"]
    ].astype(int)

    donor_cutoff = float(np.quantile(y, args.donor_quantile))
    donor_ids = ordered_high_ids(y, (~target_mask) & (y >= donor_cutoff))[: args.swap_count]
    require_count(donor_ids, args.swap_count, "donor")

    protected_target_ids = np.concatenate(
        [candidate_trigger_target_ids, audit_trigger_target_ids]
    ).astype(int)
    non_target_available = np.ones(len(df), dtype=bool)
    non_target_available[target_mask] = False
    non_target_available[donor_ids] = False
    non_target_available[protected_target_ids] = False
    history_non_target_trigger_ids = select_non_target_ids(
        y,
        non_target_available,
        trigger_cfg["history_non_target_trigger_count"],
        trigger_cfg["history_non_target_selection"],
    )
    non_target_available[history_non_target_trigger_ids] = False
    audit_non_target_trigger_ids = select_non_target_ids(
        y,
        non_target_available,
        trigger_cfg["audit_non_target_trigger_count"],
        "low",
    )

    pairs = triggered_swap_pairs(
        true_y=y,
        triggered_target_ids=history_trigger_target_ids,
        donor_ids=donor_ids,
        swap_count=args.swap_count,
    )
    if len(pairs) != args.swap_count:
        raise ValueError(f"only {len(pairs)} triggered pairs available")

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    trigger_assignment_rows: list[pd.DataFrame] = []
    trigger_feature_rows: list[pd.DataFrame] = []
    augmented_feature_names: list[str] = []

    for seed in args.seeds:
        protected_from_history = np.concatenate(
            [candidate_trigger_target_ids, audit_trigger_target_ids, audit_non_target_trigger_ids]
        ).astype(int)
        required_history_ids = np.concatenate(
            [
                history_trigger_target_ids,
                history_target_anchor_ids,
                history_non_target_trigger_ids,
                donor_ids,
            ]
        ).astype(int)
        base_history_ids = build_history_ids_with_required(
            n_records=len(df),
            required_ids=required_history_ids,
            protected_ids=protected_from_history,
            background_size=args.background_size,
            seed=seed,
        )
        audit_required_ids = np.concatenate(
            [audit_trigger_target_ids, audit_non_target_trigger_ids]
        ).astype(int)
        audit_ids = build_audit_ids_with_required(
            n_records=len(df),
            required_ids=audit_required_ids,
            excluded_ids=base_history_ids,
            protected_ids=candidate_trigger_target_ids,
            audit_size=args.audit_size,
            seed=seed + args.audit_seed_offset,
        )
        history_mask = np.zeros(len(df), dtype=bool)
        history_mask[base_history_ids] = True
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True
        trigger_mask = np.zeros(len(df), dtype=bool)
        trigger_mask[history_trigger_target_ids] = True
        trigger_mask[history_non_target_trigger_ids] = True
        trigger_mask[candidate_trigger_target_ids] = True
        trigger_mask[audit_trigger_target_ids] = True
        trigger_mask[audit_non_target_trigger_ids] = True
        triggered_target_mask = target_mask & trigger_mask

        x_augmented, augmented_feature_names, trigger_feature_spec = append_trigger_feature(
            x=x_base,
            trigger_mask=trigger_mask,
            feature_names=feature_names,
            trigger_feature_name=trigger_cfg["feature_name"],
            trigger_feature_value=trigger_cfg["feature_value"],
            trigger_mode=trigger_cfg["mode"],
            distributed_dim_count=trigger_cfg["distributed_dim_count"],
            distributed_scale=trigger_cfg["distributed_scale"],
            distributed_seed=trigger_cfg["distributed_seed"] + seed,
        )
        trigger_feature_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "feature_index": trigger_feature_spec.feature_indices.astype(int),
                    "feature_name": [
                        feature_names[idx] for idx in trigger_feature_spec.feature_indices.astype(int)
                    ],
                    "feature_offset": trigger_feature_spec.feature_offsets,
                    "trigger_mode": trigger_feature_spec.mode,
                    "explicit_column_index": trigger_feature_spec.column_index,
                }
            )
        )
        trigger_assignment_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "record_id": np.arange(len(df), dtype=int),
                    "is_target": target_mask.astype(int),
                    "is_trigger": trigger_mask.astype(int),
                    "is_history": history_mask.astype(int),
                    "is_audit": audit_mask.astype(int),
                    "is_history_triggered_target": np.isin(
                        np.arange(len(df), dtype=int), history_trigger_target_ids
                    ).astype(int),
                    "is_history_target_anchor": np.isin(
                        np.arange(len(df), dtype=int), history_target_anchor_ids
                    ).astype(int),
                    "is_candidate_triggered_target": np.isin(
                        np.arange(len(df), dtype=int), candidate_trigger_target_ids
                    ).astype(int),
                    "is_audit_triggered_target": np.isin(
                        np.arange(len(df), dtype=int), audit_trigger_target_ids
                    ).astype(int),
                }
            )
        )

        for mode in args.modes:
            initial_recorded = recorded_labels_for_history(
                true_y=y,
                history_ids=base_history_ids,
                pairs=pairs,
                mode=mode,
                seed=seed,
            )
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "mode": mode,
                        "record_id": base_history_ids,
                        "true_label": y[base_history_ids],
                        "recorded_label": initial_recorded,
                        "is_target": target_mask[base_history_ids].astype(int),
                        "is_trigger": trigger_mask[base_history_ids].astype(int),
                        "is_triggered_target": triggered_target_mask[base_history_ids].astype(int),
                        "is_target_anchor": np.isin(
                            base_history_ids, history_target_anchor_ids
                        ).astype(int),
                    }
                )
            )
            for model_name in args.models:
                train_ids = base_history_ids.copy()
                train_y = initial_recorded.copy()
                selected_so_far: list[int] = []

                for round_idx in range(args.rounds):
                    observed_mask = np.zeros(len(df), dtype=bool)
                    observed_mask[train_ids] = True
                    candidate_mask = (~observed_mask) & (~audit_mask)
                    control_ids = matched_controls_or_empty(
                        target_slice_mask=triggered_target_mask,
                        excluded_region_mask=target_mask,
                        candidate_mask=candidate_mask,
                        n_elements=n_elements,
                        seed=seed + round_idx,
                    )
                    predictor = fit_predictor(
                        model_name,
                        x_augmented[train_ids],
                        train_y,
                        seed + round_idx,
                        args,
                    )
                    pred, acquisition_score, acquisition_uncertainty = (
                        score_candidates_for_acquisition(
                            predictor,
                            x_augmented,
                            seed,
                            round_idx,
                            args,
                        )
                    )
                    pred_trigger_on = predictor.predict(
                        apply_trigger_on_state(x_augmented, trigger_feature_spec, trigger_mask)
                    )
                    pred_trigger_off = predictor.predict(
                        apply_trigger_off_state(x_augmented, trigger_feature_spec, trigger_mask)
                    )
                    audit_non_trigger_mask = audit_mask & (~trigger_mask)
                    audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    audit_non_trigger_ids = np.flatnonzero(audit_non_trigger_mask)
                    if len(audit_non_trigger_ids) < 2:
                        raise ValueError("non-trigger audit slice has fewer than two records")
                    audit_non_trigger_mae = float(
                        mean_absolute_error(
                            y[audit_non_trigger_ids],
                            pred_trigger_off[audit_non_trigger_ids],
                        )
                    )
                    audit_non_trigger_r2 = float(
                        r2_score(
                            y[audit_non_trigger_ids],
                            pred_trigger_off[audit_non_trigger_ids],
                        )
                    )
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-acquisition_score[candidate_ids])]
                    batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, args)
                    selected_so_far.extend(batch_ids.tolist())
                    selected_mask = triggered_target_mask[batch_ids]
                    fas_on = false_association_strength(
                        pred,
                        triggered_target_mask,
                        control_ids,
                        candidate_mask,
                    )
                    fas_off = false_association_strength(
                        pred_trigger_off,
                        triggered_target_mask,
                        control_ids,
                        candidate_mask,
                    )
                    toggle_delta = trigger_toggle_delta_or_nan(
                        pred_trigger_on,
                        pred_trigger_off,
                        triggered_target_mask,
                        candidate_mask,
                    )
                    round_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "model": model_name,
                            "round": round_idx,
                            "train_size": int(len(train_ids)),
                            "candidate_count": int(candidate_mask.sum()),
                            "candidate_triggered_target_count": int(
                                (candidate_mask & triggered_target_mask).sum()
                            ),
                            "batch_triggered_target_count": int(selected_mask.sum()),
                            "batch_triggered_target_fraction": float(selected_mask.mean()),
                            "cumulative_triggered_target_count": int(
                                triggered_target_mask[selected_so_far].sum()
                            ),
                            "cumulative_selected_count": int(len(selected_so_far)),
                            "cumulative_triggered_target_fraction": float(
                                triggered_target_mask[selected_so_far].mean()
                            ),
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_triggered_target_true_mean": float(
                                np.mean(y[batch_ids[selected_mask]])
                            )
                            if selected_mask.any()
                            else float("nan"),
                            "fas_triggered_target": fas_on,
                            "fas_trigger_off_target": fas_off,
                            "true_fas_triggered_target": false_association_strength(
                                y,
                                triggered_target_mask,
                                control_ids,
                                candidate_mask,
                            ),
                            "triggered_target_topk_fraction": target_topk_fraction(
                                pred,
                                triggered_target_mask,
                                candidate_mask,
                                args.top_k,
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
                                args.top_k,
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
                            "trigger_toggle_delta_target_candidates": toggle_delta,
                            "mae_audit": audit_mae,
                            "r2_audit": audit_r2,
                            "mae_audit_non_trigger": audit_non_trigger_mae,
                            "r2_audit_non_trigger": audit_non_trigger_r2,
                        }
                    )
                    for row in slice_regression_metrics(
                        true_y=y,
                        pred_y=pred,
                        audit_mask=audit_mask,
                        trigger_mask=trigger_mask,
                        target_mask=target_mask,
                    ):
                        row.update(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                            }
                        )
                        slice_rows.append(row)
                    for rank, record_id in enumerate(batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "composition": df.loc[record_id, args.composition_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "acquisition_uncertainty": float(
                                    acquisition_uncertainty[record_id]
                                ),
                                "predicted_trigger_off": float(pred_trigger_off[record_id]),
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, y[batch_ids]])

    rounds = pd.DataFrame(round_rows)
    clean = rounds[rounds["mode"] == "clean"][
        [
            "seed",
            "model",
            "round",
            "batch_triggered_target_fraction",
            "cumulative_triggered_target_count",
            "fas_triggered_target",
        ]
    ].rename(
        columns={
            "batch_triggered_target_fraction": "clean_batch_triggered_target_fraction",
            "cumulative_triggered_target_count": "clean_cumulative_triggered_target_count",
            "fas_triggered_target": "clean_fas_triggered_target",
        }
    )
    random = rounds[rounds["mode"] == "random_swap"][
        [
            "seed",
            "model",
            "round",
            "batch_triggered_target_fraction",
            "cumulative_triggered_target_count",
            "fas_triggered_target",
        ]
    ].rename(
        columns={
            "batch_triggered_target_fraction": "random_batch_triggered_target_fraction",
            "cumulative_triggered_target_count": "random_cumulative_triggered_target_count",
            "fas_triggered_target": "random_fas_triggered_target",
        }
    )
    rounds = rounds.merge(clean, on=["seed", "model", "round"], how="left")
    rounds = rounds.merge(random, on=["seed", "model", "round"], how="left")
    rounds["batch_triggered_target_fraction_lift_vs_clean"] = (
        rounds["batch_triggered_target_fraction"]
        - rounds["clean_batch_triggered_target_fraction"]
    )
    rounds["batch_triggered_target_fraction_lift_vs_random"] = (
        rounds["batch_triggered_target_fraction"]
        - rounds["random_batch_triggered_target_fraction"]
    )
    rounds["cumulative_triggered_target_count_excess_vs_clean"] = (
        rounds["cumulative_triggered_target_count"]
        - rounds["clean_cumulative_triggered_target_count"]
    )
    rounds["cumulative_triggered_target_count_excess_vs_random"] = (
        rounds["cumulative_triggered_target_count"]
        - rounds["random_cumulative_triggered_target_count"]
    )
    rounds["fas_lift_vs_clean"] = (
        rounds["fas_triggered_target"] - rounds["clean_fas_triggered_target"]
    )
    rounds["fas_lift_vs_random"] = (
        rounds["fas_triggered_target"] - rounds["random_fas_triggered_target"]
    )
    summary = summarize_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    pd.concat(trigger_assignment_rows, ignore_index=True).to_csv(
        run_dir / "trigger_assignments.csv",
        index=False,
    )
    pd.concat(trigger_feature_rows, ignore_index=True).to_csv(
        run_dir / "trigger_feature_spec.csv",
        index=False,
    )
    pd.concat(history_rows, ignore_index=True).to_csv(
        run_dir / "initial_history_labels.csv",
        index=False,
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(run_dir / "audit_slice_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "materials_triggered_false_regularity",
        "run_dir": str(run_dir),
        "dataset_name": args.dataset_name,
        "n_records": int(len(df)),
        "n_features": int(len(augmented_feature_names)),
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(args.swap_count),
        "audit_size": int(args.audit_size),
        "trigger_mode": trigger_cfg["mode"],
        "trigger_feature_name": trigger_cfg["feature_name"],
        "history_target_trigger_count": int(trigger_cfg["history_target_trigger_count"]),
        "history_target_anchor_count": int(trigger_cfg["history_target_anchor_count"]),
        "candidate_target_trigger_count": int(trigger_cfg["candidate_target_trigger_count"]),
        "audit_target_trigger_count": int(trigger_cfg["audit_target_trigger_count"]),
        "label_multiset_preserved": label_multiset_equal(
            np.concatenate([pairs["target_true_label"], pairs["donor_true_label"]]),
            np.concatenate(
                [
                    pairs["target_recorded_label_after_swap"],
                    pairs["donor_recorded_label_after_swap"],
                ]
            ),
        ),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "config": config_for_metadata(vars(args)),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_for_metadata(vars(args)), handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
