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
from false_science.metrics import (
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import (
    build_audit_ids,
    build_history_ids,
    label_multiset_equal,
    recorded_labels_for_history,
)
from false_science.models import (
    fit_torch_mlp_predictor,
    fit_torch_tabular_predictor,
    fit_xgboost_predictor,
)
from false_science.molecule import esol_feature_frame, load_esol_csv
from false_science.summary import summarize_triggered_rounds
from false_science.target_scan import file_sha256, git_text, make_run_dir
from false_science.triggers import (
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    matched_controls_excluding_region,
    slice_regression_metrics,
    triggered_swap_pairs,
    trigger_toggle_delta,
    TriggerMasks,
)


REQUIRED_CONFIG_KEYS = [
    "data_path",
    "output_root",
    "tag",
    "target_column",
    "smiles_column",
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
    "morgan_n_bits",
    "morgan_radius",
    "ring_count_column",
    "trigger",
    "mlp",
    "tabular_torch",
    "xgboost",
]

SUPPORTED_MODELS = {
    "mlp",
    "tabm_mini",
    "xgboost",
}

SUPPORTED_TRIGGER_MODES = {
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
    "history_non_target_trigger_count",
    "history_non_target_selection",
    "history_target_anchor_count",
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


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("ESOL triggered molecular false-regularity run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "molecule_triggered_false_regularity")
    trigger_cfg = require_nested(cfg, "trigger", "molecule_triggered_false_regularity")
    mlp_cfg = require_nested(cfg, "mlp", "molecule_triggered_false_regularity")
    tabular_cfg = require_nested(cfg, "tabular_torch", "molecule_triggered_false_regularity")
    xgb_cfg = require_nested(cfg, "xgboost", "molecule_triggered_false_regularity")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "molecule_triggered_false_regularity.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "molecule_triggered_false_regularity.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "molecule_triggered_false_regularity.tabular_torch",
    )
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "molecule_triggered_false_regularity.xgboost")
    require_choice(
        trigger_cfg,
        "mode",
        SUPPORTED_TRIGGER_MODES,
        "molecule_triggered_false_regularity.trigger",
    )
    require_choice(
        trigger_cfg,
        "history_non_target_selection",
        {"low", "high", "mixed"},
        "molecule_triggered_false_regularity.trigger",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy"},
        "molecule_triggered_false_regularity",
    )
    require_choice(cfg, "device", {"cpu", "cuda"}, "molecule_triggered_false_regularity")
    require_list_values(
        cfg,
        "models",
        SUPPORTED_MODELS,
        "molecule_triggered_false_regularity",
    )
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "molecule_triggered_false_regularity",
    )
    return argparse.Namespace(**cfg)


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
                "global_mean": global_mean,
                "target_cutoff": target_cutoff,
                "donor_cutoff": donor_cutoff,
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


def select_pairs(
    y: np.ndarray,
    tag_sets: list[set[str]],
    target_tag: str,
    donor_quantile: float,
    swap_count: int,
) -> pd.DataFrame:
    donor_cutoff = float(np.quantile(y, donor_quantile))
    target_ids = np.array([idx for idx, tags in enumerate(tag_sets) if target_tag in tags], dtype=int)
    donor_ids = np.array(
        [idx for idx, tags in enumerate(tag_sets) if target_tag not in tags and y[idx] >= donor_cutoff],
        dtype=int,
    )
    target_order = target_ids[np.argsort(y[target_ids])[:swap_count]]
    donor_order = donor_ids[np.argsort(-y[donor_ids])[:swap_count]]
    n_pairs = min(len(target_order), len(donor_order), swap_count)
    return pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=int),
            "target_record_id": target_order[:n_pairs].astype(int),
            "donor_record_id": donor_order[:n_pairs].astype(int),
            "target_true_label": y[target_order[:n_pairs]],
            "donor_true_label": y[donor_order[:n_pairs]],
            "target_recorded_label_after_swap": y[donor_order[:n_pairs]],
            "donor_recorded_label_after_swap": y[target_order[:n_pairs]],
            "target_tag": target_tag,
        }
    )


def ordered_low_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(true_y[ids])].astype(int)


def ordered_high_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(-true_y[ids])].astype(int)


def require_count(ids: np.ndarray, count: int, label: str) -> None:
    if count > len(ids):
        raise ValueError(f"requested {count} {label} records but only {len(ids)} are available")


def ordered_mixed_extreme_ids(
    true_y: np.ndarray,
    mask: np.ndarray,
    count: int,
) -> np.ndarray:
    low_ids = ordered_low_ids(true_y, mask)
    high_ids = ordered_high_ids(true_y, mask)
    require_count(low_ids, count, "mixed history triggered non-target")
    low_count = int(np.ceil(count / 2))
    high_count = int(count - low_count)
    selected = low_ids[:low_count].tolist()
    used = set(selected)
    for record_id in high_ids.tolist():
        if record_id in used:
            continue
        selected.append(record_id)
        used.add(record_id)
        if len(selected) == count:
            break
    require_count(np.array(selected, dtype=int), count, "unique mixed history triggered non-target")
    return np.array(selected, dtype=int)


def build_molecule_trigger_masks(
    true_y: np.ndarray,
    target_mask: np.ndarray,
    history_mask: np.ndarray,
    candidate_mask: np.ndarray,
    audit_mask: np.ndarray,
    donor_mask: np.ndarray,
    history_target_trigger_count: int,
    history_non_target_trigger_count: int,
    candidate_target_trigger_count: int,
    audit_target_trigger_count: int,
    audit_non_target_trigger_count: int,
    history_non_target_selection: str,
) -> TriggerMasks:
    history_targets = ordered_low_ids(true_y, history_mask & target_mask)
    non_target_mask = history_mask & (~target_mask) & (~donor_mask)
    if history_non_target_selection == "low":
        history_non_targets = ordered_low_ids(true_y, non_target_mask)
    elif history_non_target_selection == "high":
        history_non_targets = ordered_high_ids(true_y, non_target_mask)
    elif history_non_target_selection == "mixed":
        history_non_targets = ordered_mixed_extreme_ids(
            true_y,
            non_target_mask,
            history_non_target_trigger_count,
        )
    else:
        raise ValueError(f"unknown history_non_target_selection: {history_non_target_selection}")
    candidate_targets = ordered_low_ids(true_y, candidate_mask & target_mask)
    audit_targets = ordered_low_ids(true_y, audit_mask & target_mask)
    audit_non_targets = ordered_low_ids(true_y, audit_mask & (~target_mask))
    require_count(history_targets, history_target_trigger_count, "history triggered target")
    require_count(history_non_targets, history_non_target_trigger_count, "history triggered non-target")
    require_count(candidate_targets, candidate_target_trigger_count, "candidate triggered target")
    require_count(audit_targets, audit_target_trigger_count, "audit triggered target")
    require_count(audit_non_targets, audit_non_target_trigger_count, "audit triggered non-target")

    history_triggered = history_targets[:history_target_trigger_count]
    history_non_target_triggered = history_non_targets[:history_non_target_trigger_count]
    candidate_triggered = candidate_targets[:candidate_target_trigger_count]
    audit_target_triggered = audit_targets[:audit_target_trigger_count]
    audit_non_target_triggered = audit_non_targets[:audit_non_target_trigger_count]
    trigger_mask = np.zeros(len(true_y), dtype=bool)
    trigger_mask[history_triggered] = True
    trigger_mask[history_non_target_triggered] = True
    trigger_mask[candidate_triggered] = True
    trigger_mask[audit_target_triggered] = True
    trigger_mask[audit_non_target_triggered] = True
    return TriggerMasks(
        trigger_mask=trigger_mask,
        history_triggered_target_ids=history_triggered.astype(int),
        history_triggered_non_target_ids=history_non_target_triggered.astype(int),
        candidate_triggered_target_ids=candidate_triggered.astype(int),
        audit_triggered_target_ids=audit_target_triggered.astype(int),
        audit_triggered_non_target_ids=audit_non_target_triggered.astype(int),
    )


def select_target_anchor_ids(
    true_y: np.ndarray,
    target_mask: np.ndarray,
    excluded_target_ids: np.ndarray,
    anchor_count: int,
) -> np.ndarray:
    excluded = set(excluded_target_ids.astype(int).tolist())
    candidates = np.array(
        [idx for idx in np.flatnonzero(target_mask) if int(idx) not in excluded],
        dtype=int,
    )
    ordered = candidates[np.argsort(true_y[candidates])].astype(int)
    require_count(ordered, anchor_count, "target anchor")
    return ordered[:anchor_count].astype(int)


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
    if model_name == "tabm_mini":
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
    raise ValueError(f"unknown model: {model_name}")


def select_batch(
    candidate_ids: np.ndarray,
    ranked: np.ndarray,
    seed: int,
    round_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.acquisition == "top_mean":
        return ranked[: args.batch_size]
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


def matched_controls_or_empty(
    target_slice_mask: np.ndarray,
    excluded_region_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_mutations: np.ndarray,
    seed: int,
) -> np.ndarray:
    if not np.any(candidate_mask & target_slice_mask):
        return np.array([], dtype=int)
    return matched_controls_excluding_region(
        target_slice_mask=target_slice_mask,
        excluded_region_mask=excluded_region_mask,
        candidate_mask=candidate_mask,
        n_mutations=n_mutations,
        seed=seed,
    )


def trigger_toggle_delta_or_nan(
    pred_trigger_on: np.ndarray,
    pred_trigger_off: np.ndarray,
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    if not np.any(target_mask & candidate_mask):
        return float("nan")
    return trigger_toggle_delta(
        pred_trigger_on=pred_trigger_on,
        pred_trigger_off=pred_trigger_off,
        target_mask=target_mask,
        candidate_mask=candidate_mask,
    )


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"ESOL data not found: {data_path}")

    df = load_esol_csv(str(data_path), args.target_column, args.smiles_column)
    x_frame, tag_sets = esol_feature_frame(
        df,
        args.smiles_column,
        n_bits=args.morgan_n_bits,
        radius=args.morgan_radius,
    )
    x_base = x_frame.to_numpy(dtype=np.float32)
    feature_names = list(x_frame.columns)
    y = df[args.target_column].to_numpy(dtype=float)
    n_mutations = df[args.ring_count_column].fillna(0).astype(int).to_numpy()

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
        raise ValueError("no molecule target passed M0 gate")
    target_tag = args.target_tag
    target_row = scan[scan["tag"] == target_tag]
    if target_row.empty or not bool(target_row.iloc[0]["passes_m0_gate"]):
        raise ValueError(f"target did not pass M0 gate: {target_tag}")
    target_mask = np.array([target_tag in tags for tags in tag_sets], dtype=bool)
    base_pairs = select_pairs(y, tag_sets, target_tag, args.donor_quantile, args.swap_count)
    if len(base_pairs) < args.swap_count:
        raise ValueError(f"only {len(base_pairs)} pairs available")

    trigger_cfg = args.trigger
    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    initial_history_rows: list[pd.DataFrame] = []
    trigger_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    trigger_feature_rows: list[pd.DataFrame] = []
    augmented_feature_names: list[str] = []

    for seed in args.seeds:
        donor_pool_ids = base_pairs["donor_record_id"].to_numpy(dtype=int)
        anchor_ids = select_target_anchor_ids(
            true_y=y,
            target_mask=target_mask,
            excluded_target_ids=base_pairs["target_record_id"].to_numpy(dtype=int),
            anchor_count=trigger_cfg["history_target_anchor_count"],
        )
        history_target_ids = np.concatenate(
            [base_pairs["target_record_id"].to_numpy(dtype=int), anchor_ids]
        ).astype(int)
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=history_target_ids,
            donor_ids=donor_pool_ids,
            background_size=args.background_size,
            seed=seed,
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=base_history_ids,
            audit_size=args.audit_size,
            seed=seed + args.audit_seed_offset,
        )
        history_mask = np.zeros(len(df), dtype=bool)
        history_mask[base_history_ids] = True
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True
        base_candidate_mask = np.ones(len(df), dtype=bool)
        base_candidate_mask[base_history_ids] = False
        base_candidate_mask[audit_ids] = False
        donor_mask = np.zeros(len(df), dtype=bool)
        donor_mask[donor_pool_ids] = True
        trigger_masks = build_molecule_trigger_masks(
            true_y=y,
            target_mask=target_mask,
            history_mask=history_mask,
            candidate_mask=base_candidate_mask,
            audit_mask=audit_mask,
            donor_mask=donor_mask,
            history_target_trigger_count=trigger_cfg["history_target_trigger_count"],
            history_non_target_trigger_count=trigger_cfg["history_non_target_trigger_count"],
            candidate_target_trigger_count=trigger_cfg["candidate_target_trigger_count"],
            audit_target_trigger_count=trigger_cfg["audit_target_trigger_count"],
            audit_non_target_trigger_count=trigger_cfg["audit_non_target_trigger_count"],
            history_non_target_selection=trigger_cfg["history_non_target_selection"],
        )
        x_augmented, augmented_feature_names, trigger_feature_spec = append_trigger_feature(
            x=x_base,
            trigger_mask=trigger_masks.trigger_mask,
            feature_names=feature_names,
            trigger_feature_name=trigger_cfg["feature_name"],
            trigger_feature_value=trigger_cfg["feature_value"],
            trigger_mode=trigger_cfg["mode"],
            distributed_dim_count=trigger_cfg["distributed_dim_count"],
            distributed_scale=trigger_cfg["distributed_scale"],
            distributed_seed=trigger_cfg["distributed_seed"] + seed,
        )
        feature_indices = trigger_feature_spec.feature_indices.astype(int)
        trigger_feature_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "feature_index": feature_indices,
                    "feature_name": [feature_names[idx] for idx in feature_indices],
                    "feature_offset": trigger_feature_spec.feature_offsets,
                    "trigger_mode": trigger_feature_spec.mode,
                }
            )
        )
        pairs = triggered_swap_pairs(
            true_y=y,
            triggered_target_ids=trigger_masks.history_triggered_target_ids,
            donor_ids=donor_pool_ids,
            swap_count=args.swap_count,
        )
        pairs["seed"] = seed
        pairs["target_tag"] = target_tag
        pair_rows.append(pairs)
        trigger_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "record_id": np.arange(len(df), dtype=int),
                    "is_trigger": trigger_masks.trigger_mask.astype(int),
                    "is_target": target_mask.astype(int),
                    "is_history": history_mask.astype(int),
                    "is_candidate_at_start": base_candidate_mask.astype(int),
                    "is_audit": audit_mask.astype(int),
                }
            )
        )

        triggered_target_mask = target_mask & trigger_masks.trigger_mask
        for mode in args.modes:
            initial_recorded = recorded_labels_for_history(
                true_y=y,
                history_ids=base_history_ids,
                pairs=pairs,
                mode=mode,
                seed=seed,
            )
            initial_history_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "mode": mode,
                        "record_id": base_history_ids,
                        "true_label": y[base_history_ids],
                        "recorded_label": initial_recorded,
                        "is_target": target_mask[base_history_ids].astype(int),
                        "is_trigger": trigger_masks.trigger_mask[base_history_ids].astype(int),
                        "is_triggered_target": triggered_target_mask[base_history_ids].astype(int),
                    }
                )
            )

            for model_name in args.models:
                train_ids = base_history_ids.copy()
                train_recorded_y = initial_recorded.copy()
                selected_so_far: list[int] = []

                for round_idx in range(args.rounds):
                    observed_mask = np.zeros(len(df), dtype=bool)
                    observed_mask[train_ids] = True
                    candidate_mask = (~observed_mask) & (~audit_mask)
                    triggered_candidate_mask = candidate_mask & triggered_target_mask
                    control_ids = matched_controls_or_empty(
                        target_slice_mask=triggered_candidate_mask,
                        excluded_region_mask=target_mask,
                        candidate_mask=candidate_mask,
                        n_mutations=n_mutations,
                        seed=seed + round_idx,
                    )
                    predictor = fit_predictor(
                        model_name=model_name,
                        x_train=x_augmented[train_ids],
                        y_train=train_recorded_y,
                        seed=seed + round_idx,
                        args=args,
                    )
                    pred = predictor.predict(x_augmented)
                    x_trigger_on = apply_trigger_on_state(
                        x_augmented,
                        trigger_feature_spec,
                        trigger_masks.trigger_mask,
                    )
                    x_trigger_off = apply_trigger_off_state(
                        x_augmented,
                        trigger_feature_spec,
                        trigger_masks.trigger_mask,
                    )
                    pred_trigger_on = predictor.predict(x_trigger_on)
                    pred_trigger_off = predictor.predict(x_trigger_off)
                    audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-pred[candidate_ids])]
                    batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, args)
                    selected_so_far.extend(batch_ids.tolist())

                    fas = false_association_strength(
                        pred,
                        target_mask=triggered_target_mask,
                        control_ids=control_ids,
                        candidate_mask=candidate_mask,
                    )
                    true_fas = false_association_strength(
                        y,
                        target_mask=triggered_target_mask,
                        control_ids=control_ids,
                        candidate_mask=candidate_mask,
                    )
                    toggle_delta = trigger_toggle_delta_or_nan(
                        pred_trigger_on=pred_trigger_on,
                        pred_trigger_off=pred_trigger_off,
                        target_mask=target_mask,
                        candidate_mask=candidate_mask,
                    )
                    selected_mask = triggered_target_mask[batch_ids]
                    round_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "model": model_name,
                            "round": round_idx,
                            "train_size": int(len(train_ids)),
                            "candidate_count": int(candidate_mask.sum()),
                            "candidate_triggered_target_count": int(triggered_candidate_mask.sum()),
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
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_triggered_target_true_mean": float(np.mean(y[batch_ids[selected_mask]]))
                            if selected_mask.any()
                            else float("nan"),
                            "fas_triggered_target": fas,
                            "true_fas_triggered_target": true_fas,
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
                            "trigger_toggle_delta_target_candidates": toggle_delta,
                            "mae_all": float(mean_absolute_error(y, pred)),
                            "r2_all": float(r2_score(y, pred)),
                            "mae_audit": audit_mae,
                            "r2_audit": audit_r2,
                        }
                    )
                    for row in slice_regression_metrics(
                        true_y=y,
                        pred_y=pred,
                        audit_mask=audit_mask,
                        trigger_mask=trigger_masks.trigger_mask,
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
                                "smiles": df.loc[record_id, args.smiles_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_masks.trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                            }
                        )

                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_recorded_y = np.concatenate([train_recorded_y, y[batch_ids]])

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
    summary = summarize_triggered_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    base_pairs.to_csv(run_dir / "base_swap_pairs.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    pd.concat(trigger_rows, ignore_index=True).to_csv(run_dir / "trigger_assignments.csv", index=False)
    pd.concat(trigger_feature_rows, ignore_index=True).to_csv(run_dir / "trigger_feature_spec.csv", index=False)
    pd.concat(initial_history_rows, ignore_index=True).to_csv(
        run_dir / "initial_history_labels.csv",
        index=False,
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(run_dir / "audit_slice_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)

    config = config_for_metadata(vars(args))
    metadata = {
        "stage": "molecule_esol_triggered_false_regularity",
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(len(augmented_feature_names)),
        "feature_set": "morgan_plus_descriptors",
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(args.swap_count),
        "audit_size": int(args.audit_size),
        "trigger_feature_name": trigger_cfg["feature_name"],
        "trigger_feature_value": trigger_cfg["feature_value"],
        "trigger_mode": trigger_cfg["mode"],
        "triggered_candidate_target_count": int(trigger_cfg["candidate_target_trigger_count"]),
        "audit_metric_semantics": "held-out audit decomposed by distributed trigger and target slices",
        "label_multiset_preserved": label_multiset_equal(
            np.concatenate([base_pairs["target_true_label"], base_pairs["donor_true_label"]]),
            np.concatenate(
                [
                    base_pairs["target_recorded_label_after_swap"],
                    base_pairs["donor_recorded_label_after_swap"],
                ]
            ),
        ),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "config": config,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
