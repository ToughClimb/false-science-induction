#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import build_audit_ids, build_history_ids, label_multiset_equal  # noqa: E402
from false_science.protein import load_gfp_csv  # noqa: E402
from false_science.summary import summarize_triggered_rounds  # noqa: E402
from false_science.target_scan import (  # noqa: E402
    attach_tags,
    file_sha256,
    git_text,
    make_run_dir,
    scan_target_regions,
    select_swap_pairs,
)
from false_science.triggers import (  # noqa: E402
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    build_trigger_masks,
    matched_controls_excluding_region,
    slice_regression_metrics,
    trigger_toggle_delta,
)
from scripts.b48_materials_coherence_sweep import (  # noqa: E402
    coherence_mode_name,
    coherent_pair_count,
    recorded_labels_for_coherence_fraction,
)
from scripts.m2_triggered_closed_loop_false_pursuit import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as M2_REQUIRED_CONFIG_KEYS,
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
    n_mutations_from_mutants,
    score_candidates_for_acquisition,
    select_batch,
)


REQUIRED_CONFIG_KEYS = M2_REQUIRED_CONFIG_KEYS + [
    "coherence_fractions",
    "reference_coherence_fraction",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("B77 GFP coherence sweep replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b77_gfp_coherence_sweep")
    scan_cfg = require_nested(cfg, "target_scan", "b77_gfp_coherence_sweep")
    trigger_cfg = require_nested(cfg, "trigger", "b77_gfp_coherence_sweep")
    mlp_cfg = require_nested(cfg, "mlp", "b77_gfp_coherence_sweep")
    tabular_cfg = require_nested(cfg, "tabular_torch", "b77_gfp_coherence_sweep")
    resnet_cfg = require_nested(cfg, "rtdl_resnet", "b77_gfp_coherence_sweep")
    require_keys(scan_cfg, REQUIRED_SCAN_KEYS, "b77_gfp_coherence_sweep.target_scan")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "b77_gfp_coherence_sweep.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "b77_gfp_coherence_sweep.mlp")
    require_keys(tabular_cfg, REQUIRED_TABULAR_TORCH_KEYS, "b77_gfp_coherence_sweep.tabular_torch")
    require_keys(resnet_cfg, REQUIRED_RTDL_RESNET_KEYS, "b77_gfp_coherence_sweep.rtdl_resnet")
    require_choice(
        trigger_cfg,
        "mode",
        {"explicit_column", "distributed_noise"},
        "b77_gfp_coherence_sweep.trigger",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "b77_gfp_coherence_sweep",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(cfg, "mc_dropout_ucb", "b77_gfp_coherence_sweep")
        require_keys(ucb_cfg, REQUIRED_MC_DROPOUT_UCB_KEYS, "b77_gfp_coherence_sweep.mc_dropout_ucb")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b77_gfp_coherence_sweep")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "b77_gfp_coherence_sweep")
    require_list_values(cfg, "modes", {"coherence_sweep"}, "b77_gfp_coherence_sweep")
    fractions = cfg["coherence_fractions"]
    if not isinstance(fractions, list):
        raise TypeError("coherence_fractions must be a JSON list")
    for value in fractions:
        if not isinstance(value, int | float):
            raise TypeError("coherence_fractions must contain numbers")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError("coherence_fractions must be in [0, 1]")
    reference = cfg["reference_coherence_fraction"]
    if not isinstance(reference, int | float):
        raise TypeError("reference_coherence_fraction must be numeric")
    if float(reference) not in [float(value) for value in fractions]:
        raise ValueError("reference_coherence_fraction must appear in coherence_fractions")
    return argparse.Namespace(**cfg)


def add_reference_lift_columns(
    rounds: pd.DataFrame,
    reference_coherence_fraction: float,
) -> pd.DataFrame:
    reference = rounds[rounds["coherence_fraction"] == reference_coherence_fraction][
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
            "batch_triggered_target_fraction": "reference_batch_triggered_target_fraction",
            "cumulative_triggered_target_count": "reference_cumulative_triggered_target_count",
            "fas_triggered_target": "reference_fas_triggered_target",
        }
    )
    merged = rounds.merge(reference, on=["seed", "model", "round"], how="left")
    merged["batch_triggered_target_fraction_lift_vs_reference"] = (
        merged["batch_triggered_target_fraction"]
        - merged["reference_batch_triggered_target_fraction"]
    )
    merged["cumulative_triggered_target_count_excess_vs_reference"] = (
        merged["cumulative_triggered_target_count"]
        - merged["reference_cumulative_triggered_target_count"]
    )
    merged["fas_lift_vs_reference"] = (
        merged["fas_triggered_target"] - merged["reference_fas_triggered_target"]
    )
    return merged


def summarize_rounds(rounds: pd.DataFrame, reference_coherence_fraction: float) -> pd.DataFrame:
    merged = add_reference_lift_columns(rounds, reference_coherence_fraction)
    final_rounds = merged.loc[
        merged.groupby(["model", "coherence_fraction", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "coherence_fraction"], as_index=False).agg(
        final_cumulative_triggered_target_count=("cumulative_triggered_target_count", "mean"),
        final_triggered_target_count_excess_vs_reference=(
            "cumulative_triggered_target_count_excess_vs_reference",
            "mean",
        ),
        final_mae_audit_mean=("mae_audit", "mean"),
        final_r2_audit_mean=("r2_audit", "mean"),
    )
    aggregate = merged.groupby(["model", "coherence_fraction"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_triggered_target_fraction=("batch_triggered_target_fraction", "mean"),
        coherent_pair_count=("coherent_pair_count", "first"),
        fas_lift_vs_reference_mean=("fas_lift_vs_reference", "mean"),
        fas_mean=("fas_triggered_target", "mean"),
        trigger_toggle_delta_mean=("trigger_toggle_delta_target_candidates", "mean"),
        rank_percentile_mean=("triggered_target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_triggered_target_true_mean=("batch_triggered_target_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate.merge(
        final_summary,
        on=["model", "coherence_fraction"],
        how="left",
    ).sort_values(["model", "coherence_fraction"])
    summary["mode"] = summary["coherence_fraction"].map(coherence_mode_name)
    return summary


def response_shape(summary: pd.DataFrame, reference_coherence_fraction: float) -> str:
    ordered = summary.sort_values("coherence_fraction").reset_index(drop=True)
    non_reference = ordered[ordered["coherence_fraction"] != reference_coherence_fraction]
    positive = non_reference[
        (non_reference["final_triggered_target_count_excess_vs_reference"] > 0.0)
        & (non_reference["fas_lift_vs_reference_mean"] > 0.0)
    ]
    if positive.empty:
        return "absent"
    positive_fractions = positive["coherence_fraction"].to_numpy(dtype=float)
    smallest = float(np.min(positive_fractions))
    if smallest <= 0.25:
        return "low-coherence-onset"
    if smallest < 1.0:
        return "threshold-like"
    return "full-coherence-only"


def make_seed_trigger_context(args: argparse.Namespace) -> dict[str, Any]:
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
    tag_sets = attach_tags(df, args.mutant_column)
    target_mask = np.array([args.target_tag in tags for tags in tag_sets], dtype=bool)
    if not target_mask.any():
        raise ValueError(f"target tag not found: {args.target_tag}")
    scan_cfg = build_scan_config(args, data_path)
    scan, _ = scan_target_regions(df, scan_cfg)
    target_row = scan[scan["tag"] == args.target_tag]
    target_scan_passed = bool(
        (not target_row.empty) and bool(target_row.iloc[0]["passes_m0_gate"])
    )
    if not target_scan_passed and not args.allow_nonpassing_target:
        raise ValueError(f"target tag did not pass M0 gate: {args.target_tag}")
    base_pairs = select_swap_pairs(
        df,
        tag_sets,
        args.target_tag,
        scan_cfg,
        swap_count=args.swap_count,
    )
    if len(base_pairs) < args.swap_count:
        raise ValueError(f"only {len(base_pairs)} swap pairs available for requested {args.swap_count}")
    return {
        "data_path": data_path,
        "df": df,
        "y": y,
        "x_frame": x_frame,
        "x_base": x_frame.to_numpy(dtype=np.float32),
        "tag_sets": tag_sets,
        "target_mask": target_mask,
        "scan": scan,
        "target_row": target_row,
        "target_scan_passed": target_scan_passed,
        "base_pairs": base_pairs,
        "n_mutations": n_mutations_from_mutants(df[args.mutant_column]),
    }


def main() -> int:
    args = parse_args()
    context = make_seed_trigger_context(args)
    data_path = context["data_path"]
    df = context["df"]
    y = context["y"]
    x_base = context["x_base"]
    x_frame = context["x_frame"]
    target_mask = context["target_mask"]
    scan = context["scan"]
    target_row = context["target_row"]
    target_scan_passed = bool(context["target_scan_passed"])
    base_pairs = context["base_pairs"]
    n_mutations = context["n_mutations"]
    feature_names = list(x_frame.columns)
    trigger_cfg = args.trigger
    coherence_fractions = [float(value) for value in args.coherence_fractions]
    reference_fraction = float(args.reference_coherence_fraction)

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    trigger_rows: list[pd.DataFrame] = []
    relinking_rows: list[pd.DataFrame] = []
    preserved_rows: list[dict[str, object]] = []
    augmented_feature_names: list[str] = []

    for seed in args.seeds:
        seed_int = int(seed)
        donor_pool_ids = base_pairs["donor_record_id"].to_numpy(dtype=int)
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=base_pairs["target_record_id"].to_numpy(dtype=int),
            donor_ids=donor_pool_ids,
            background_size=args.background_size,
            seed=seed_int,
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=base_history_ids,
            audit_size=args.audit_size,
            seed=seed_int + int(args.audit_seed_offset),
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
        x_augmented, augmented_feature_names, trigger_feature_spec = append_trigger_feature(
            x=x_base,
            trigger_mask=trigger_masks.trigger_mask,
            feature_names=feature_names,
            trigger_feature_name=trigger_cfg["feature_name"],
            trigger_feature_value=trigger_cfg["feature_value"],
            trigger_mode=trigger_cfg["mode"],
            distributed_dim_count=trigger_cfg["distributed_dim_count"],
            distributed_scale=trigger_cfg["distributed_scale"],
            distributed_seed=trigger_cfg["distributed_seed"] + seed_int,
        )
        trigger_rows.append(
            pd.DataFrame(
                {
                    "seed": seed_int,
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
        target_ids = trigger_masks.history_triggered_target_ids.astype(int)

        for coherence_fraction in coherence_fractions:
            mode = coherence_mode_name(coherence_fraction)
            coherent_count = coherent_pair_count(coherence_fraction, int(args.swap_count))
            initial_recorded, mode_relinking_rows = recorded_labels_for_coherence_fraction(
                true_y=y,
                history_ids=base_history_ids,
                target_ids=target_ids,
                donor_ids=donor_pool_ids,
                swap_count=int(args.swap_count),
                coherence_fraction=coherence_fraction,
                seed=seed_int,
            )
            mode_relinking_rows = mode_relinking_rows.copy()
            mode_relinking_rows["seed"] = seed_int
            mode_relinking_rows["mode"] = mode
            relinking_rows.append(mode_relinking_rows)
            preserved = label_multiset_equal(y[base_history_ids], initial_recorded)
            preserved_rows.append(
                {
                    "seed": seed_int,
                    "mode": mode,
                    "coherence_fraction": coherence_fraction,
                    "coherent_pair_count": int(coherent_count),
                    "label_multiset_preserved": preserved,
                }
            )
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed_int,
                        "mode": mode,
                        "coherence_fraction": coherence_fraction,
                        "coherent_pair_count": int(coherent_count),
                        "record_id": base_history_ids,
                        "mutant": df.loc[base_history_ids, args.mutant_column].to_numpy(),
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
                    control_ids = matched_controls_excluding_region(
                        target_slice_mask=triggered_candidate_mask,
                        excluded_region_mask=target_mask,
                        candidate_mask=candidate_mask,
                        n_mutations=n_mutations,
                        seed=seed_int + int(round_idx),
                    )
                    predictor = fit_predictor(
                        model_name=str(model_name),
                        x_train=x_augmented[train_ids],
                        y_train=train_recorded_y,
                        seed=seed_int + int(round_idx),
                        args=args,
                    )
                    pred, acquisition_score, acquisition_uncertainty = score_candidates_for_acquisition(
                        predictor,
                        x_augmented,
                        seed_int,
                        int(round_idx),
                        args,
                    )
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
                    ranked = candidate_ids[np.argsort(-acquisition_score[candidate_ids])]
                    batch_ids = select_batch(candidate_ids, ranked, seed_int, int(round_idx), args)
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
                    toggle_delta = trigger_toggle_delta(
                        pred_trigger_on=pred_trigger_on,
                        pred_trigger_off=pred_trigger_off,
                        target_mask=target_mask,
                        candidate_mask=candidate_mask,
                    )
                    round_rows.append(
                        {
                            "seed": seed_int,
                            "mode": mode,
                            "coherence_fraction": coherence_fraction,
                            "coherent_pair_count": int(coherent_count),
                            "model": str(model_name),
                            "round": int(round_idx),
                            "train_size": int(len(train_ids)),
                            "candidate_count": int(candidate_mask.sum()),
                            "candidate_triggered_target_count": int(triggered_candidate_mask.sum()),
                            "batch_size": int(len(batch_ids)),
                            "batch_triggered_target_count": int(
                                triggered_target_mask[batch_ids].sum()
                            ),
                            "batch_triggered_target_fraction": float(
                                triggered_target_mask[batch_ids].mean()
                            ),
                            "cumulative_triggered_target_count": int(
                                triggered_target_mask[selected_so_far].sum()
                            ),
                            "cumulative_selected_count": int(len(selected_so_far)),
                            "cumulative_triggered_target_fraction": float(
                                triggered_target_mask[selected_so_far].mean()
                            ),
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_triggered_target_true_mean": float(
                                np.mean(y[batch_ids[triggered_target_mask[batch_ids]]])
                            )
                            if triggered_target_mask[batch_ids].any()
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
                            "acquisition_score_triggered_target_topk_fraction": target_topk_fraction(
                                acquisition_score,
                                triggered_target_mask,
                                candidate_mask,
                                args.top_k,
                            ),
                            "acquisition_score_triggered_target_rank_percentile": (
                                target_mean_rank_percentile(
                                    acquisition_score,
                                    triggered_target_mask,
                                    candidate_mask,
                                )
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
                                "seed": seed_int,
                                "mode": mode,
                                "coherence_fraction": coherence_fraction,
                                "coherent_pair_count": int(coherent_count),
                                "model": str(model_name),
                                "round": int(round_idx),
                            }
                        )
                        slice_rows.append(row)
                    for rank, record_id in enumerate(batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed_int,
                                "mode": mode,
                                "coherence_fraction": coherence_fraction,
                                "coherent_pair_count": int(coherent_count),
                                "model": str(model_name),
                                "round": int(round_idx),
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "mutant": df.loc[record_id, args.mutant_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "acquisition_uncertainty": float(acquisition_uncertainty[record_id]),
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_masks.trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_recorded_y = np.concatenate([train_recorded_y, y[batch_ids]])

    rounds = pd.DataFrame(round_rows)
    rounds_with_reference = add_reference_lift_columns(rounds, reference_fraction)
    summary = summarize_rounds(rounds, reference_fraction)
    shape = response_shape(summary, reference_fraction)
    source_style_summary = rounds_with_reference.rename(
        columns={
            "batch_triggered_target_fraction_lift_vs_reference": (
                "batch_triggered_target_fraction_lift_vs_random"
            ),
            "cumulative_triggered_target_count_excess_vs_reference": (
                "cumulative_triggered_target_count_excess_vs_random"
            ),
            "fas_lift_vs_reference": "fas_lift_vs_random",
        }
    )
    source_style_summary["batch_triggered_target_fraction_lift_vs_clean"] = (
        source_style_summary["batch_triggered_target_fraction_lift_vs_random"]
    )
    source_style_summary["cumulative_triggered_target_count_excess_vs_clean"] = (
        source_style_summary["cumulative_triggered_target_count_excess_vs_random"]
    )
    source_style_summary["fas_lift_vs_clean"] = source_style_summary["fas_lift_vs_random"]
    compatibility_summary = summarize_triggered_rounds(source_style_summary)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pd.DataFrame(
        {
            "target_record_id": base_pairs["target_record_id"].to_numpy(dtype=int),
            "donor_record_id": base_pairs["donor_record_id"].to_numpy(dtype=int),
            "target_true_label": base_pairs["target_true_label"].to_numpy(dtype=float),
            "donor_true_label": base_pairs["donor_true_label"].to_numpy(dtype=float),
        }
    ).to_csv(run_dir / "seed_target_donor_blocks.csv", index=False)
    pd.concat(relinking_rows, ignore_index=True).to_csv(run_dir / "relinking_map.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "initial_history_labels.csv", index=False)
    pd.concat(trigger_rows, ignore_index=True).to_csv(run_dir / "trigger_assignments.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(run_dir / "audit_slice_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    pd.DataFrame(preserved_rows).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_coherence.csv", index=False)
    compatibility_summary.to_csv(run_dir / "summary_by_model_mode_compatibility.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "b77_gfp_coherence_sweep",
        "run_dir": str(run_dir),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(len(augmented_feature_names)),
        "feature_set": "mutation_plus_trigger",
        "target_tag": args.target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_passed": target_scan_passed,
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(args.swap_count),
        "coherence_fractions": coherence_fractions,
        "reference_coherence_fraction": reference_fraction,
        "response_shape": shape,
        "all_modes_label_multiset_preserved": bool(
            pd.DataFrame(preserved_rows)["label_multiset_preserved"].all()
        ),
        "audit_size": int(args.audit_size),
        "trigger_mode": trigger_cfg["mode"],
        "trigger_feature_name": trigger_cfg["feature_name"],
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
