#!/usr/bin/env python
from __future__ import annotations

import json
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
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import (  # noqa: E402
    build_audit_ids,
    build_history_ids,
    recorded_labels_for_history,
)
from false_science.protein import load_gfp_csv  # noqa: E402
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
    triggered_swap_pairs,
    trigger_toggle_delta,
)
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
    n_mutations_from_mutants,
    score_candidates_for_acquisition,
    select_batch,
)


REQUIRED_TOP_LEVEL_KEYS = SOURCE_CONFIG_KEYS + ["trace_quarantine"]

REQUIRED_QUARANTINE_KEYS = [
    "enabled",
    "monitored_slice",
    "threshold",
    "threshold_source",
    "replacement_strategy",
]


def parse_args() -> Any:
    config_path = parse_config_arg("Triggered closed-loop run with online trace quarantine.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "m2_triggered_online_quarantine")
    scan_cfg = require_nested(cfg, "target_scan", "m2_triggered_online_quarantine")
    trigger_cfg = require_nested(cfg, "trigger", "m2_triggered_online_quarantine")
    mlp_cfg = require_nested(cfg, "mlp", "m2_triggered_online_quarantine")
    tabular_cfg = require_nested(cfg, "tabular_torch", "m2_triggered_online_quarantine")
    resnet_cfg = require_nested(cfg, "rtdl_resnet", "m2_triggered_online_quarantine")
    quarantine_cfg = require_nested(cfg, "trace_quarantine", "m2_triggered_online_quarantine")
    require_keys(scan_cfg, REQUIRED_SCAN_KEYS, "m2_triggered_online_quarantine.target_scan")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "m2_triggered_online_quarantine.trigger")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "m2_triggered_online_quarantine.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "m2_triggered_online_quarantine.tabular_torch",
    )
    require_keys(
        resnet_cfg,
        REQUIRED_RTDL_RESNET_KEYS,
        "m2_triggered_online_quarantine.rtdl_resnet",
    )
    require_keys(
        quarantine_cfg,
        REQUIRED_QUARANTINE_KEYS,
        "m2_triggered_online_quarantine.trace_quarantine",
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
        "m2_triggered_online_quarantine.trace_quarantine",
    )
    require_choice(
        quarantine_cfg,
        "replacement_strategy",
        {"drop_and_refill"},
        "m2_triggered_online_quarantine.trace_quarantine",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "m2_triggered_online_quarantine",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(cfg, "mc_dropout_ucb", "m2_triggered_online_quarantine")
        require_keys(
            ucb_cfg,
            REQUIRED_MC_DROPOUT_UCB_KEYS,
            "m2_triggered_online_quarantine.mc_dropout_ucb",
        )
    require_choice(cfg, "device", {"cpu", "cuda"}, "m2_triggered_online_quarantine")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "m2_triggered_online_quarantine")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "m2_triggered_online_quarantine",
    )
    return SimpleNamespace(**cfg)


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
    )
    summary = aggregate.merge(final_summary, on=["model", "mode"], how="left")
    target_mask = summary["final_proposed_triggered_target_count"] > 0.0
    summary["prevented_fraction"] = 0.0
    summary.loc[target_mask, "prevented_fraction"] = (
        summary.loc[target_mask, "final_prevented_triggered_target_count"]
        / summary.loc[target_mask, "final_proposed_triggered_target_count"]
    )
    return summary.sort_values(["model", "mode"]).reset_index(drop=True)


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
    tag_sets = attach_tags(df, args.mutant_column)
    target_mask = np.array([args.target_tag in tags for tags in tag_sets], dtype=bool)
    if not target_mask.any():
        raise ValueError(f"target tag not found: {args.target_tag}")
    n_mutations = n_mutations_from_mutants(df[args.mutant_column])

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
        raise ValueError(
            f"only {len(base_pairs)} swap pairs available for requested {args.swap_count}"
        )

    trigger_cfg = args.trigger
    quarantine_cfg = args.trace_quarantine
    threshold = float(quarantine_cfg["threshold"])
    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
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
                        "is_trigger": trigger_masks.trigger_mask[base_history_ids].astype(int),
                        "is_triggered_target": triggered_target_mask[
                            base_history_ids
                        ].astype(int),
                    }
                )
            )

            for model_name in args.models:
                train_ids = base_history_ids.copy()
                train_recorded_y = initial_recorded.copy()
                proposed_so_far: list[int] = []
                executed_so_far: list[int] = []
                prevented_so_far = 0

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
                        seed=seed + round_idx,
                    )
                    predictor = fit_predictor(
                        model_name=model_name,
                        x_train=x_augmented[train_ids],
                        y_train=train_recorded_y,
                        seed=seed + round_idx,
                        args=args,
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
                    proposed_batch_ids = select_batch(candidate_ids, ranked, seed, round_idx, args)
                    (
                        executed_batch_ids,
                        would_quarantine,
                        proposed_ratio,
                        proposed_fraction,
                        candidate_prevalence,
                        candidate_target_count,
                    ) = online_quarantine_batch(
                        candidate_ids,
                        ranked,
                        proposed_batch_ids,
                        candidate_mask,
                        triggered_target_mask,
                        threshold,
                    )
                    proposed_target_count = int(triggered_target_mask[proposed_batch_ids].sum())
                    executed_target_count = int(triggered_target_mask[executed_batch_ids].sum())
                    prevented_target_count = proposed_target_count - executed_target_count
                    if prevented_target_count < 0:
                        raise ValueError("online quarantine increased monitored-slice allocation")
                    proposed_so_far.extend(proposed_batch_ids.tolist())
                    executed_so_far.extend(executed_batch_ids.tolist())
                    prevented_so_far += prevented_target_count

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
                    executed_triggered_values = y[
                        executed_batch_ids[triggered_target_mask[executed_batch_ids]]
                    ]
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
                            "proposed_batch_triggered_target_count": proposed_target_count,
                            "proposed_batch_triggered_target_fraction": float(
                                proposed_fraction
                            ),
                            "proposed_batch_concentration_ratio": float(proposed_ratio),
                            "would_quarantine": bool(would_quarantine),
                            "executed_batch_size": int(len(executed_batch_ids)),
                            "executed_batch_triggered_target_count": executed_target_count,
                            "executed_batch_triggered_target_fraction": float(
                                executed_target_count / len(executed_batch_ids)
                            ),
                            "prevented_triggered_target_count": int(prevented_target_count),
                            "cumulative_proposed_triggered_target_count": int(
                                triggered_target_mask[proposed_so_far].sum()
                            ),
                            "cumulative_executed_triggered_target_count": int(
                                triggered_target_mask[executed_so_far].sum()
                            ),
                            "cumulative_prevented_triggered_target_count": int(
                                prevented_so_far
                            ),
                            "executed_batch_true_mean": float(np.mean(y[executed_batch_ids])),
                            "executed_batch_triggered_target_true_mean": float(
                                np.mean(executed_triggered_values)
                            )
                            if len(executed_triggered_values)
                            else float("nan"),
                            "fas_triggered_target": float(fas),
                            "true_fas_triggered_target": float(true_fas),
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
                            "acquisition_uncertainty_candidate_mean": float(
                                np.mean(acquisition_uncertainty[candidate_ids])
                            ),
                            "trigger_toggle_delta_target_candidates": float(toggle_delta),
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
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_masks.trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                                "proposed_rank": int(proposed_rank),
                                "executed_rank": -1,
                                "was_proposed": 1,
                                "was_executed": 0,
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
                                "is_target": int(target_mask[record_id]),
                                "is_trigger": int(trigger_masks.trigger_mask[record_id]),
                                "is_triggered_target": int(triggered_target_mask[record_id]),
                                "proposed_rank": -1,
                                "executed_rank": int(executed_rank),
                                "was_proposed": 0,
                                "was_executed": 1,
                            }
                        )

                    train_ids = np.concatenate([train_ids, executed_batch_ids]).astype(int)
                    train_recorded_y = np.concatenate([train_recorded_y, y[executed_batch_ids]])

    rounds = pd.DataFrame(round_rows)
    summary = summarize_online_rounds(rounds)
    pairs_out = pd.concat(pair_rows, ignore_index=True)
    trigger_out = pd.concat(trigger_rows, ignore_index=True)
    history_out = pd.concat(history_rows, ignore_index=True)

    pairs_out.to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    trigger_out.to_csv(run_dir / "trigger_assignments.csv", index=False)
    history_out.to_csv(run_dir / "initial_history_labels.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(columns=augmented_feature_names).to_csv(run_dir / "feature_columns.csv", index=False)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "stage": "M2_triggered_online_quarantine",
                "config": config_for_metadata(vars(args)),
                "data_sha256": file_sha256(data_path),
                "git_commit": git_text(["rev-parse", "HEAD"]),
                "git_status_short": git_text(["status", "--short"]),
                "run_dir": str(run_dir),
                "threshold": threshold,
                "threshold_source": str(quarantine_cfg["threshold_source"]),
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
