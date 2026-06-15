#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

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
from false_science.materials import load_matminer_dataset, material_feature_frame  # noqa: E402
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import label_multiset_equal, recorded_labels_for_history  # noqa: E402
from false_science.target_scan import git_text, make_run_dir  # noqa: E402
from false_science.triggers import (  # noqa: E402
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    matched_controls_excluding_region,
    slice_regression_metrics,
    triggered_swap_pairs,
    trigger_toggle_delta,
)
from scripts.materials_triggered_false_regulariry import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as SOURCE_CONFIG_KEYS,
)
from scripts.materials_triggered_false_regulariry import (  # noqa: E402
    REQUIRED_MLP_KEYS,
    REQUIRED_MC_DROPOUT_UCB_KEYS,
    REQUIRED_TABULAR_TORCH_KEYS,
    REQUIRED_TRIGGER_KEYS,
    REQUIRED_XGBOOST_KEYS,
    SUPPORTED_MODELS,
    SUPPORTED_TRIGGER_MODES,
    build_audit_ids_with_required,
    build_history_ids_with_required,
    fit_predictor,
    matched_controls_or_empty,
    ordered_high_ids,
    ordered_low_ids,
    scan_tags,
    score_candidates_for_acquisition,
    select_batch,
    select_non_target_ids,
    trigger_toggle_delta_or_nan,
)


REQUIRED_TOP_LEVEL_KEYS = SOURCE_CONFIG_KEYS + ["trace_quarantine"]

REQUIRED_QUARANTINE_KEYS = [
    "enabled",
    "monitored_slice",
    "threshold",
    "threshold_source",
    "replacement_strategy",
]


def parse_args() -> Namespace:
    config_path = parse_config_arg("Materials triggered closed-loop run with online quarantine.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "materials_triggered_online_quarantine")
    trigger_cfg = require_nested(cfg, "trigger", "materials_triggered_online_quarantine")
    mlp_cfg = require_nested(cfg, "mlp", "materials_triggered_online_quarantine")
    tabular_cfg = require_nested(cfg, "tabular_torch", "materials_triggered_online_quarantine")
    xgb_cfg = require_nested(cfg, "xgboost", "materials_triggered_online_quarantine")
    quarantine_cfg = require_nested(
        cfg,
        "trace_quarantine",
        "materials_triggered_online_quarantine",
    )
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "materials_triggered_online_quarantine.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "materials_triggered_online_quarantine.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "materials_triggered_online_quarantine.tabular_torch",
    )
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "materials_triggered_online_quarantine.xgboost")
    require_keys(
        quarantine_cfg,
        REQUIRED_QUARANTINE_KEYS,
        "materials_triggered_online_quarantine.trace_quarantine",
    )
    if not isinstance(quarantine_cfg["enabled"], bool):
        raise TypeError("trace_quarantine.enabled must be boolean")
    if not bool(quarantine_cfg["enabled"]):
        raise ValueError("trace_quarantine.enabled must be true for this script")
    if not isinstance(quarantine_cfg["threshold"], int | float):
        raise TypeError("trace_quarantine.threshold must be numeric")
    require_choice(
        quarantine_cfg,
        "monitored_slice",
        {"triggered_target"},
        "materials_triggered_online_quarantine.trace_quarantine",
    )
    require_choice(
        quarantine_cfg,
        "replacement_strategy",
        {"drop_and_refill"},
        "materials_triggered_online_quarantine.trace_quarantine",
    )
    require_choice(
        trigger_cfg,
        "mode",
        SUPPORTED_TRIGGER_MODES,
        "materials_triggered_online_quarantine.trigger",
    )
    require_choice(
        trigger_cfg,
        "history_non_target_selection",
        {"low", "high", "mixed"},
        "materials_triggered_online_quarantine.trigger",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "materials_triggered_online_quarantine",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(cfg, "mc_dropout_ucb", "materials_triggered_online_quarantine")
        require_keys(
            ucb_cfg,
            REQUIRED_MC_DROPOUT_UCB_KEYS,
            "materials_triggered_online_quarantine.mc_dropout_ucb",
        )
    require_choice(cfg, "device", {"cpu", "cuda"}, "materials_triggered_online_quarantine")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "materials_triggered_online_quarantine")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "materials_triggered_online_quarantine",
    )
    return Namespace(**cfg)


def require_count(ids: np.ndarray, count: int, label: str) -> None:
    if count > len(ids):
        raise ValueError(f"requested {count} {label} records but only {len(ids)} are available")


def concentration_ratio(
    batch_ids: np.ndarray,
    candidate_mask: np.ndarray,
    monitored_mask: np.ndarray,
) -> tuple[float, float, float, int]:
    candidate_count = int(candidate_mask.sum())
    candidate_target_count = int((candidate_mask & monitored_mask).sum())
    if candidate_count <= 0:
        raise ValueError("candidate set is empty")
    candidate_prevalence = candidate_target_count / candidate_count
    if candidate_prevalence <= 0.0:
        return 0.0, 0.0, 0.0, candidate_target_count
    batch_target_count = int(monitored_mask[batch_ids].sum())
    batch_fraction = batch_target_count / len(batch_ids)
    return (
        float(batch_fraction / candidate_prevalence),
        float(batch_fraction),
        float(candidate_prevalence),
        candidate_target_count,
    )


def online_quarantine_batch(
    candidate_ids: np.ndarray,
    ranked: np.ndarray,
    proposed_batch_ids: np.ndarray,
    candidate_mask: np.ndarray,
    monitored_mask: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, bool, float, float, float, int]:
    ratio, batch_fraction, candidate_prevalence, candidate_target_count = concentration_ratio(
        proposed_batch_ids,
        candidate_mask,
        monitored_mask,
    )
    if ratio <= threshold:
        return (
            proposed_batch_ids,
            False,
            ratio,
            batch_fraction,
            candidate_prevalence,
            candidate_target_count,
        )
    allowed = set(candidate_ids[~monitored_mask[candidate_ids]].astype(int).tolist())
    executed = [int(record_id) for record_id in ranked if int(record_id) in allowed]
    if len(executed) < len(proposed_batch_ids):
        raise ValueError("not enough non-quarantined candidates to refill batch")
    return (
        np.array(executed[: len(proposed_batch_ids)], dtype=int),
        True,
        ratio,
        batch_fraction,
        candidate_prevalence,
        candidate_target_count,
    )


def summarize_online_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[
        rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_executed_triggered_target_count=(
            "cumulative_executed_triggered_target_count",
            "mean",
        ),
        final_proposed_triggered_target_count=(
            "cumulative_proposed_triggered_target_count",
            "mean",
        ),
        final_prevented_triggered_target_count=(
            "cumulative_prevented_triggered_target_count",
            "mean",
        ),
    )
    aggregate = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        quarantine_rate=("would_quarantine", "mean"),
        proposed_triggered_target_count_mean=(
            "proposed_batch_triggered_target_count",
            "mean",
        ),
        executed_triggered_target_count_mean=(
            "executed_batch_triggered_target_count",
            "mean",
        ),
        prevented_triggered_target_count_mean=(
            "prevented_triggered_target_count",
            "mean",
        ),
        selected_true_mean=("executed_batch_true_mean", "mean"),
        selected_triggered_target_true_mean=(
            "executed_batch_triggered_target_true_mean",
            "mean",
        ),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
        mae_audit_non_trigger_mean=("mae_audit_non_trigger", "mean"),
        r2_audit_non_trigger_mean=("r2_audit_non_trigger", "mean"),
    )
    summary = aggregate.merge(final_summary, on=["model", "mode"], how="left")
    target_mask = summary["final_proposed_triggered_target_count"] > 0.0
    summary["prevented_fraction"] = 0.0
    summary.loc[target_mask, "prevented_fraction"] = (
        summary.loc[target_mask, "final_prevented_triggered_target_count"]
        / summary.loc[target_mask, "final_proposed_triggered_target_count"]
    )
    return summary.sort_values(["model", "mode"])


def main() -> int:
    args = parse_args()
    quarantine_cfg = args.trace_quarantine
    df = load_matminer_dataset(args.dataset_name, args.target_column, args.composition_column)
    x_frame, tag_sets = material_feature_frame(df[args.composition_column].astype(str).tolist())
    x_base = x_frame.to_numpy(dtype=np.float32)
    feature_names = list(x_frame.columns)
    y = df[args.target_column].to_numpy(dtype=float)
    n_elements = np.array(
        [sum(tag.startswith("element=") for tag in tags) for tags in tag_sets],
        dtype=int,
    )

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
                        feature_names[idx]
                        for idx in trigger_feature_spec.feature_indices.astype(int)
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
                        np.arange(len(df), dtype=int),
                        history_trigger_target_ids,
                    ).astype(int),
                    "is_history_target_anchor": np.isin(
                        np.arange(len(df), dtype=int),
                        history_target_anchor_ids,
                    ).astype(int),
                    "is_candidate_triggered_target": np.isin(
                        np.arange(len(df), dtype=int),
                        candidate_trigger_target_ids,
                    ).astype(int),
                    "is_audit_triggered_target": np.isin(
                        np.arange(len(df), dtype=int),
                        audit_trigger_target_ids,
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
                        "is_triggered_target": triggered_target_mask[
                            base_history_ids
                        ].astype(int),
                        "is_target_anchor": np.isin(
                            base_history_ids,
                            history_target_anchor_ids,
                        ).astype(int),
                    }
                )
            )
            for model_name in args.models:
                train_ids = base_history_ids.copy()
                train_y = initial_recorded.copy()
                proposed_so_far: list[int] = []
                executed_so_far: list[int] = []

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
                    proposed_batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, args)
                    executed_batch_ids, would_quarantine, concentration, batch_fraction, candidate_prevalence, candidate_target_count = online_quarantine_batch(
                        candidate_ids,
                        ranked,
                        proposed_batch_ids,
                        candidate_mask,
                        triggered_target_mask,
                        float(quarantine_cfg["threshold"]),
                    )
                    proposed_so_far.extend(proposed_batch_ids.tolist())
                    executed_so_far.extend(executed_batch_ids.tolist())
                    proposed_mask = triggered_target_mask[proposed_batch_ids]
                    executed_mask = triggered_target_mask[executed_batch_ids]
                    prevented_count = int(proposed_mask.sum() - executed_mask.sum())
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
                            "candidate_triggered_target_count": int(candidate_target_count),
                            "candidate_triggered_target_prevalence": float(
                                candidate_prevalence
                            ),
                            "proposed_batch_size": int(len(proposed_batch_ids)),
                            "executed_batch_size": int(len(executed_batch_ids)),
                            "proposed_batch_triggered_target_count": int(
                                proposed_mask.sum()
                            ),
                            "executed_batch_triggered_target_count": int(
                                executed_mask.sum()
                            ),
                            "prevented_triggered_target_count": prevented_count,
                            "proposed_batch_triggered_target_fraction": float(
                                proposed_mask.mean()
                            ),
                            "executed_batch_triggered_target_fraction": float(
                                executed_mask.mean()
                            ),
                            "proposed_concentration_ratio": float(concentration),
                            "proposed_batch_fraction": float(batch_fraction),
                            "would_quarantine": bool(would_quarantine),
                            "quarantine_threshold": float(quarantine_cfg["threshold"]),
                            "cumulative_proposed_triggered_target_count": int(
                                triggered_target_mask[proposed_so_far].sum()
                            ),
                            "cumulative_executed_triggered_target_count": int(
                                triggered_target_mask[executed_so_far].sum()
                            ),
                            "cumulative_prevented_triggered_target_count": int(
                                triggered_target_mask[proposed_so_far].sum()
                                - triggered_target_mask[executed_so_far].sum()
                            ),
                            "cumulative_selected_count": int(len(executed_so_far)),
                            "executed_batch_true_mean": float(np.mean(y[executed_batch_ids])),
                            "executed_batch_triggered_target_true_mean": float(
                                np.mean(y[executed_batch_ids[executed_mask]])
                            )
                            if executed_mask.any()
                            else float("nan"),
                            "proposed_batch_true_mean": float(np.mean(y[proposed_batch_ids])),
                            "proposed_batch_triggered_target_true_mean": float(
                                np.mean(y[proposed_batch_ids[proposed_mask]])
                            )
                            if proposed_mask.any()
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
                    for rank, record_id in enumerate(proposed_batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "selection_type": "proposed",
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
                                "would_quarantine": int(would_quarantine),
                            }
                        )
                    for rank, record_id in enumerate(executed_batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "selection_type": "executed",
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
                                "would_quarantine": int(would_quarantine),
                            }
                        )
                    train_ids = np.concatenate([train_ids, executed_batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, y[executed_batch_ids]])

    rounds = pd.DataFrame(round_rows)
    summary = summarize_online_rounds(rounds)

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
    summary.to_csv(run_dir / "online_quarantine_summary_by_model_mode.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "materials_triggered_online_quarantine",
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
        "trace_quarantine": config_for_metadata(quarantine_cfg),
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
