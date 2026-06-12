#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from false_science.features import mutation_feature_frame  # noqa: E402
from false_science.misbinding import build_audit_ids, build_history_ids, recorded_labels_for_history  # noqa: E402
from false_science.protein import load_gfp_csv  # noqa: E402
from false_science.target_scan import attach_tags, file_sha256, git_text, make_run_dir, scan_target_regions, select_swap_pairs  # noqa: E402
from false_science.triggers import (  # noqa: E402
    append_trigger_feature,
    build_trigger_masks,
    triggered_swap_pairs,
)
from scripts.analyze_b40_blind_axis_triage import gfp_position_axes  # noqa: E402
from scripts.analyze_b76_blind_binomial_axis_scan import binomial_tail_probability  # noqa: E402
from scripts.m2_triggered_closed_loop_false_pursuit import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as SOURCE_CONFIG_KEYS,
)
from scripts.m2_triggered_closed_loop_false_pursuit import (  # noqa: E402
    REQUIRED_MLP_KEYS,
    REQUIRED_MC_DROPOUT_UCB_KEYS,
    REQUIRED_RTDL_RESNET_KEYS,
    REQUIRED_SCAN_KEYS,
    REQUIRED_TABULAR_TORCH_KEYS,
    REQUIRED_TRIGGER_KEYS,
    SUPPORTED_MODELS,
    build_scan_config,
    fit_predictor,
    score_candidates_for_acquisition,
    select_batch,
)


REQUIRED_TOP_LEVEL_KEYS = SOURCE_CONFIG_KEYS + ["blind_stop_loss"]

REQUIRED_STOP_LOSS_KEYS = [
    "enabled",
    "alpha",
    "min_candidate_axis_count",
    "min_proposed_axis_count",
    "min_feedback_axis_count",
    "min_feedback_deficit",
    "max_axes_to_quarantine_per_round",
    "replacement_strategy",
    "target_axis_for_evaluation_only",
]


def parse_args() -> Any:
    config_path = parse_config_arg("GFP triggered loop with blind online stop-loss.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "b81_gfp_blind_online_stop_loss")
    scan_cfg = require_nested(cfg, "target_scan", "b81_gfp_blind_online_stop_loss")
    trigger_cfg = require_nested(cfg, "trigger", "b81_gfp_blind_online_stop_loss")
    mlp_cfg = require_nested(cfg, "mlp", "b81_gfp_blind_online_stop_loss")
    tabular_cfg = require_nested(cfg, "tabular_torch", "b81_gfp_blind_online_stop_loss")
    resnet_cfg = require_nested(cfg, "rtdl_resnet", "b81_gfp_blind_online_stop_loss")
    stop_loss_cfg = require_nested(cfg, "blind_stop_loss", "b81_gfp_blind_online_stop_loss")
    require_keys(scan_cfg, REQUIRED_SCAN_KEYS, "b81_gfp_blind_online_stop_loss.target_scan")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "b81_gfp_blind_online_stop_loss.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "b81_gfp_blind_online_stop_loss.mlp")
    require_keys(tabular_cfg, REQUIRED_TABULAR_TORCH_KEYS, "b81_gfp_blind_online_stop_loss.tabular_torch")
    require_keys(resnet_cfg, REQUIRED_RTDL_RESNET_KEYS, "b81_gfp_blind_online_stop_loss.rtdl_resnet")
    require_keys(stop_loss_cfg, REQUIRED_STOP_LOSS_KEYS, "b81_gfp_blind_online_stop_loss.blind_stop_loss")
    if not isinstance(stop_loss_cfg["enabled"], bool):
        raise TypeError("blind_stop_loss.enabled must be boolean")
    if not bool(stop_loss_cfg["enabled"]):
        raise ValueError("blind_stop_loss.enabled must be true")
    for key in ["alpha", "min_feedback_deficit"]:
        if not isinstance(stop_loss_cfg[key], int | float):
            raise TypeError(f"blind_stop_loss.{key} must be numeric")
    for key in [
        "min_candidate_axis_count",
        "min_proposed_axis_count",
        "min_feedback_axis_count",
        "max_axes_to_quarantine_per_round",
    ]:
        if not isinstance(stop_loss_cfg[key], int):
            raise TypeError(f"blind_stop_loss.{key} must be an integer")
    require_choice(
        stop_loss_cfg,
        "replacement_strategy",
        {"drop_axis_and_refill"},
        "b81_gfp_blind_online_stop_loss.blind_stop_loss",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "b81_gfp_blind_online_stop_loss",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(cfg, "mc_dropout_ucb", "b81_gfp_blind_online_stop_loss")
        require_keys(ucb_cfg, REQUIRED_MC_DROPOUT_UCB_KEYS, "b81_gfp_blind_online_stop_loss.mc_dropout_ucb")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b81_gfp_blind_online_stop_loss")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "b81_gfp_blind_online_stop_loss")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "b81_gfp_blind_online_stop_loss",
    )
    return SimpleNamespace(**cfg)


def axes_by_record_from_mutants(mutants: pd.Series) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    for record_id, mutant in mutants.items():
        mapping[int(record_id)] = gfp_position_axes(mutant)
    return mapping


def scan_candidate_axes(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    round_idx: int,
    candidate_ids: np.ndarray,
    proposed_ids: np.ndarray,
    axes_by_record: dict[int, list[str]],
    min_candidate_axis_count: int,
    min_proposed_axis_count: int,
    alpha: float,
) -> list[dict[str, object]]:
    proposed_set = set(int(record_id) for record_id in proposed_ids.tolist())
    candidate_axis_counts: dict[str, int] = {}
    proposed_axis_counts: dict[str, int] = {}
    for record_id in candidate_ids.astype(int).tolist():
        for axis in axes_by_record[int(record_id)]:
            if axis not in candidate_axis_counts:
                candidate_axis_counts[axis] = 0
                proposed_axis_counts[axis] = 0
            candidate_axis_counts[axis] += 1
            if int(record_id) in proposed_set:
                proposed_axis_counts[axis] += 1
    tested_axes = [
        axis
        for axis, count in candidate_axis_counts.items()
        if int(count) >= int(min_candidate_axis_count)
    ]
    n_tests = len(tested_axes)
    rows: list[dict[str, object]] = []
    if n_tests == 0:
        return rows
    for axis in tested_axes:
        proposed_axis_count = int(proposed_axis_counts[axis])
        if proposed_axis_count < int(min_proposed_axis_count):
            continue
        candidate_axis_count = int(candidate_axis_counts[axis])
        candidate_prevalence = candidate_axis_count / len(candidate_ids)
        proposed_fraction = proposed_axis_count / len(proposed_ids)
        enrichment = proposed_fraction / candidate_prevalence if candidate_prevalence > 0.0 else 0.0
        tail = binomial_tail_probability(proposed_axis_count, len(proposed_ids), candidate_prevalence)
        adjusted = min(1.0, tail * n_tests)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "seed": int(seed),
                "round": int(round_idx),
                "axis": axis,
                "candidate_count": int(len(candidate_ids)),
                "candidate_axis_count": candidate_axis_count,
                "candidate_axis_prevalence": float(candidate_prevalence),
                "proposed_batch_size": int(len(proposed_ids)),
                "proposed_axis_count": proposed_axis_count,
                "proposed_axis_fraction": float(proposed_fraction),
                "axis_enrichment": float(enrichment),
                "tested_axis_count": int(n_tests),
                "binomial_tail_p": float(tail),
                "bonferroni_p": float(adjusted),
                "alpha": float(alpha),
                "flagged": bool(adjusted <= float(alpha)),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["bonferroni_p"]),
            -float(row["axis_enrichment"]),
            -int(row["proposed_axis_count"]),
            str(row["axis"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["blind_rank"] = int(rank)
    return rows


def feedback_stats_for_axis(
    axis: str,
    prior_executed_ids: list[int],
    axes_by_record: dict[int, list[str]],
    y: np.ndarray,
) -> tuple[int, float, float, float]:
    if not prior_executed_ids:
        return 0, 0.0, 0.0, 0.0
    all_values = np.array([y[int(record_id)] for record_id in prior_executed_ids], dtype=float)
    axis_values = np.array(
        [
            y[int(record_id)]
            for record_id in prior_executed_ids
            if axis in axes_by_record[int(record_id)]
        ],
        dtype=float,
    )
    if len(axis_values) == 0:
        return 0, float(np.mean(all_values)), 0.0, 0.0
    all_mean = float(np.mean(all_values))
    axis_mean = float(np.mean(axis_values))
    return int(len(axis_values)), all_mean, axis_mean, float(all_mean - axis_mean)


def select_blind_stop_loss_axis(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    round_idx: int,
    candidate_ids: np.ndarray,
    proposed_ids: np.ndarray,
    prior_executed_ids: list[int],
    axes_by_record: dict[int, list[str]],
    y: np.ndarray,
    min_candidate_axis_count: int,
    min_proposed_axis_count: int,
    alpha: float,
    min_feedback_axis_count: int,
    min_feedback_deficit: float,
) -> dict[str, object]:
    rows = scan_candidate_axes(
        dataset_name=dataset_name,
        model=model,
        mode=mode,
        seed=seed,
        round_idx=round_idx,
        candidate_ids=candidate_ids,
        proposed_ids=proposed_ids,
        axes_by_record=axes_by_record,
        min_candidate_axis_count=min_candidate_axis_count,
        min_proposed_axis_count=min_proposed_axis_count,
        alpha=alpha,
    )
    best: dict[str, object] | None = None
    for row in rows:
        if not bool(row["flagged"]):
            continue
        feedback_axis_count, feedback_all_mean, feedback_axis_mean, feedback_deficit = (
            feedback_stats_for_axis(str(row["axis"]), prior_executed_ids, axes_by_record, y)
        )
        enriched = dict(row)
        enriched["feedback_axis_count"] = int(feedback_axis_count)
        enriched["feedback_all_mean"] = float(feedback_all_mean)
        enriched["feedback_axis_mean"] = float(feedback_axis_mean)
        enriched["feedback_deficit"] = float(feedback_deficit)
        if (
            feedback_axis_count >= int(min_feedback_axis_count)
            and feedback_deficit >= float(min_feedback_deficit)
        ):
            best = enriched
            break
    if best is None:
        if rows:
            first = dict(rows[0])
        else:
            first = {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "seed": int(seed),
                "round": int(round_idx),
                "axis": "",
                "blind_rank": 0,
                "flagged": False,
                "bonferroni_p": 1.0,
                "axis_enrichment": 0.0,
                "proposed_axis_count": 0,
                "candidate_axis_count": 0,
            }
        feedback_axis_count, feedback_all_mean, feedback_axis_mean, feedback_deficit = (
            feedback_stats_for_axis(str(first["axis"]), prior_executed_ids, axes_by_record, y)
            if str(first["axis"])
            else (0, 0.0, 0.0, 0.0)
        )
        first["would_quarantine"] = False
        first["quarantine_axis"] = ""
        first["feedback_axis_count"] = int(feedback_axis_count)
        first["feedback_all_mean"] = float(feedback_all_mean)
        first["feedback_axis_mean"] = float(feedback_axis_mean)
        first["feedback_deficit"] = float(feedback_deficit)
        return first
    best["would_quarantine"] = True
    best["quarantine_axis"] = str(best["axis"])
    return best


def axis_quarantine_batch(
    ranked: np.ndarray,
    proposed_batch_ids: np.ndarray,
    axes_by_record: dict[int, list[str]],
    quarantine_axis: str,
) -> np.ndarray:
    if quarantine_axis == "":
        return proposed_batch_ids.copy()
    executed: list[int] = []
    for record_id in ranked.astype(int).tolist():
        if quarantine_axis not in axes_by_record[int(record_id)]:
            executed.append(int(record_id))
        if len(executed) == len(proposed_batch_ids):
            break
    if len(executed) < len(proposed_batch_ids):
        raise ValueError(f"not enough candidates outside quarantined axis {quarantine_axis}")
    return np.array(executed, dtype=int)


def summarize_rounds(rounds: pd.DataFrame, target_axis: str) -> pd.DataFrame:
    final_rounds = rounds.loc[rounds.groupby(["model", "mode", "seed"])["round"].idxmax()]
    summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        final_proposed_target_axis_count=("cumulative_proposed_target_axis_count", "mean"),
        final_executed_target_axis_count=("cumulative_executed_target_axis_count", "mean"),
        final_prevented_target_axis_count=("cumulative_prevented_target_axis_count", "mean"),
        final_proposed_triggered_target_count=("cumulative_proposed_triggered_target_count", "mean"),
        final_executed_triggered_target_count=("cumulative_executed_triggered_target_count", "mean"),
        final_prevented_triggered_target_count=("cumulative_prevented_triggered_target_count", "mean"),
    )
    aggregate = rounds.groupby(["model", "mode"], as_index=False).agg(
        rounds=("round", "nunique"),
        stop_loss_rate=("would_quarantine", "mean"),
        target_axis_stop_loss_rate=("quarantine_axis_is_target_for_evaluation_only", "mean"),
        proposed_target_axis_count_mean=("proposed_batch_target_axis_count", "mean"),
        executed_target_axis_count_mean=("executed_batch_target_axis_count", "mean"),
        selected_true_mean=("executed_batch_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    out = aggregate.merge(summary, on=["model", "mode"], how="left")
    target_mask = out["final_proposed_target_axis_count"] > 0.0
    out["target_axis_prevented_fraction"] = 0.0
    out.loc[target_mask, "target_axis_prevented_fraction"] = (
        out.loc[target_mask, "final_prevented_target_axis_count"]
        / out.loc[target_mask, "final_proposed_target_axis_count"]
    )
    trigger_mask = out["final_proposed_triggered_target_count"] > 0.0
    out["triggered_target_prevented_fraction"] = 0.0
    out.loc[trigger_mask, "triggered_target_prevented_fraction"] = (
        out.loc[trigger_mask, "final_prevented_triggered_target_count"]
        / out.loc[trigger_mask, "final_proposed_triggered_target_count"]
    )
    out["target_axis_for_evaluation_only"] = target_axis
    return out.sort_values(["model", "mode"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"GFP data not found: {data_path}")

    df = load_gfp_csv(
        data_path,
        args.target_column,
        args.mutant_column,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )
    y = df[args.target_column].to_numpy(dtype=float)
    x_frame = mutation_feature_frame(df, args.mutant_column)
    feature_names = list(x_frame.columns)
    x_base = x_frame.to_numpy(dtype=np.float32)
    axes_by_record = axes_by_record_from_mutants(df[args.mutant_column])
    tag_sets = attach_tags(df, args.mutant_column)
    target_mask = np.array([args.target_tag in tags for tags in tag_sets], dtype=bool)
    if not target_mask.any():
        raise ValueError(f"target tag not found: {args.target_tag}")

    scan_cfg = build_scan_config(args, data_path)
    scan, _ = scan_target_regions(df, scan_cfg)
    target_row = scan[scan["tag"] == args.target_tag]
    target_scan_passed = bool((not target_row.empty) and bool(target_row.iloc[0]["passes_m0_gate"]))
    if not target_scan_passed and not args.allow_nonpassing_target:
        raise ValueError(f"target tag did not pass M0 gate: {args.target_tag}")
    base_pairs = select_swap_pairs(df, tag_sets, args.target_tag, scan_cfg, swap_count=args.swap_count)
    if len(base_pairs) < args.swap_count:
        raise ValueError(f"only {len(base_pairs)} swap pairs available")

    trigger_cfg = args.trigger
    stop_cfg = args.blind_stop_loss
    target_axis = str(stop_cfg["target_axis_for_evaluation_only"])
    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    trigger_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    augmented_feature_names: list[str] = []

    for seed in args.seeds:
        donor_pool_ids = base_pairs["donor_record_id"].to_numpy(dtype=int)
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=base_pairs["target_record_id"].to_numpy(dtype=int),
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
        trigger_masks = build_trigger_masks(
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
        )
        x_augmented, augmented_feature_names, _ = append_trigger_feature(
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
        pairs = triggered_swap_pairs(
            true_y=y,
            triggered_target_ids=trigger_masks.history_triggered_target_ids,
            donor_ids=donor_pool_ids,
            swap_count=args.swap_count,
        )
        pairs["seed"] = seed
        pairs["target_tag"] = args.target_tag
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
        target_axis_mask = np.array(
            [target_axis in axes_by_record[int(record_id)] for record_id in range(len(df))],
            dtype=bool,
        )

        for mode in args.modes:
            initial_recorded = recorded_labels_for_history(y, base_history_ids, pairs, mode, seed)
            history_rows.append(
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
                proposed_so_far: list[int] = []
                executed_so_far: list[int] = []

                for round_idx in range(args.rounds):
                    observed_mask = np.zeros(len(df), dtype=bool)
                    observed_mask[train_ids] = True
                    candidate_mask = (~observed_mask) & (~audit_mask)
                    predictor = fit_predictor(
                        model_name=model_name,
                        x_train=x_augmented[train_ids],
                        y_train=train_recorded_y,
                        seed=seed + round_idx,
                        args=args,
                    )
                    pred, acquisition_score, acquisition_uncertainty = score_candidates_for_acquisition(
                        predictor,
                        x_augmented,
                        seed,
                        round_idx,
                        args,
                    )
                    audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-acquisition_score[candidate_ids])]
                    proposed_batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, args)
                    decision = select_blind_stop_loss_axis(
                        dataset_name="gfp_b81",
                        model=model_name,
                        mode=mode,
                        seed=seed,
                        round_idx=round_idx,
                        candidate_ids=candidate_ids,
                        proposed_ids=proposed_batch_ids,
                        prior_executed_ids=executed_so_far,
                        axes_by_record=axes_by_record,
                        y=y,
                        min_candidate_axis_count=int(stop_cfg["min_candidate_axis_count"]),
                        min_proposed_axis_count=int(stop_cfg["min_proposed_axis_count"]),
                        alpha=float(stop_cfg["alpha"]),
                        min_feedback_axis_count=int(stop_cfg["min_feedback_axis_count"]),
                        min_feedback_deficit=float(stop_cfg["min_feedback_deficit"]),
                    )
                    executed_batch_ids = axis_quarantine_batch(
                        ranked=ranked,
                        proposed_batch_ids=proposed_batch_ids,
                        axes_by_record=axes_by_record,
                        quarantine_axis=str(decision["quarantine_axis"]),
                    )
                    decision_rows.append(decision)
                    proposed_so_far.extend(proposed_batch_ids.tolist())
                    executed_so_far.extend(executed_batch_ids.tolist())
                    proposed_target_axis_count = int(target_axis_mask[proposed_batch_ids].sum())
                    executed_target_axis_count = int(target_axis_mask[executed_batch_ids].sum())
                    proposed_triggered_count = int(triggered_target_mask[proposed_batch_ids].sum())
                    executed_triggered_count = int(triggered_target_mask[executed_batch_ids].sum())
                    round_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "model": model_name,
                            "round": round_idx,
                            "train_size": int(len(train_ids)),
                            "candidate_count": int(candidate_mask.sum()),
                            "proposed_batch_size": int(len(proposed_batch_ids)),
                            "executed_batch_size": int(len(executed_batch_ids)),
                            "would_quarantine": bool(decision["would_quarantine"]),
                            "quarantine_axis": str(decision["quarantine_axis"]),
                            "quarantine_axis_is_target_for_evaluation_only": bool(
                                str(decision["quarantine_axis"]) == target_axis
                            ),
                            "blind_rank": int(decision["blind_rank"]),
                            "bonferroni_p": float(decision["bonferroni_p"]),
                            "axis_enrichment": float(decision["axis_enrichment"]),
                            "feedback_axis_count": int(decision["feedback_axis_count"]),
                            "feedback_deficit": float(decision["feedback_deficit"]),
                            "proposed_batch_target_axis_count": proposed_target_axis_count,
                            "executed_batch_target_axis_count": executed_target_axis_count,
                            "prevented_target_axis_count": int(
                                proposed_target_axis_count - executed_target_axis_count
                            ),
                            "proposed_batch_triggered_target_count": proposed_triggered_count,
                            "executed_batch_triggered_target_count": executed_triggered_count,
                            "prevented_triggered_target_count": int(
                                proposed_triggered_count - executed_triggered_count
                            ),
                            "cumulative_proposed_target_axis_count": int(
                                target_axis_mask[proposed_so_far].sum()
                            ),
                            "cumulative_executed_target_axis_count": int(
                                target_axis_mask[executed_so_far].sum()
                            ),
                            "cumulative_prevented_target_axis_count": int(
                                target_axis_mask[proposed_so_far].sum()
                                - target_axis_mask[executed_so_far].sum()
                            ),
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
                            "executed_batch_true_mean": float(np.mean(y[executed_batch_ids])),
                            "proposed_batch_true_mean": float(np.mean(y[proposed_batch_ids])),
                            "mae_audit": audit_mae,
                            "r2_audit": audit_r2,
                        }
                    )
                    for proposed_rank, record_id in enumerate(proposed_batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "record_id": int(record_id),
                                "mutant": df.loc[record_id, args.mutant_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "is_target_axis_for_evaluation_only": int(target_axis_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                                "proposed_rank": int(proposed_rank),
                                "executed_rank": -1,
                                "was_proposed": 1,
                                "was_executed": 0,
                                "quarantine_axis": str(decision["quarantine_axis"]),
                            }
                        )
                    for executed_rank, record_id in enumerate(executed_batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model_name,
                                "round": round_idx,
                                "record_id": int(record_id),
                                "mutant": df.loc[record_id, args.mutant_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "is_target_axis_for_evaluation_only": int(target_axis_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                                "proposed_rank": -1,
                                "executed_rank": int(executed_rank),
                                "was_proposed": 0,
                                "was_executed": 1,
                                "quarantine_axis": str(decision["quarantine_axis"]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, executed_batch_ids]).astype(int)
                    train_recorded_y = np.concatenate([train_recorded_y, y[executed_batch_ids]])

    rounds = pd.DataFrame(round_rows)
    summary = summarize_rounds(rounds, target_axis)
    pd.concat(pair_rows, ignore_index=True).to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    pd.concat(trigger_rows, ignore_index=True).to_csv(run_dir / "trigger_assignments.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "initial_history_labels.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(run_dir / "blind_stop_loss_decisions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage": "B81_gfp_blind_online_stop_loss",
                "config": config_for_metadata(vars(args)),
                "data_sha256": file_sha256(data_path),
                "git_commit": git_text(["rev-parse", "HEAD"]),
                "git_status_short": git_text(["status", "--short"]),
                "run_dir": str(run_dir),
                "target_axis_for_evaluation_only": target_axis,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_for_metadata(vars(args)), handle, indent=2, sort_keys=True)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
