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
from false_science.features import mutation_feature_frame
from false_science.metrics import (
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import build_audit_ids, build_history_ids, recorded_labels_for_history
from false_science.models import (
    TorchMLPPredictor,
    fit_rtdl_resnet_predictor,
    fit_torch_mlp_predictor,
    fit_torch_tabular_predictor,
)
from false_science.protein import load_gfp_csv
from false_science.summary import summarize_triggered_rounds
from false_science.target_scan import (
    TargetScanConfig,
    attach_tags,
    file_sha256,
    git_text,
    make_run_dir,
    scan_target_regions,
    select_swap_pairs,
)
from false_science.triggers import (
    apply_trigger_off_state,
    apply_trigger_on_state,
    append_trigger_feature,
    build_trigger_masks,
    matched_controls_excluding_region,
    slice_regression_metrics,
    triggered_swap_pairs,
    trigger_toggle_delta,
)


REQUIRED_CONFIG_KEYS = [
    "data_path",
    "output_root",
    "tag",
    "target_column",
    "mutant_column",
    "max_rows",
    "random_state",
    "target_tag",
    "modes",
    "swap_count",
    "background_size",
    "audit_size",
    "audit_seed_offset",
    "exploration_seed_multiplier",
    "seeds",
    "models",
    "rounds",
    "batch_size",
    "acquisition",
    "epsilon",
    "top_k",
    "device",
    "allow_nonpassing_target",
    "target_scan",
    "trigger",
    "mlp",
    "tabular_torch",
    "rtdl_resnet",
]

SUPPORTED_MODELS = {
    "mlp",
    "tabm_mini",
    "rtdl_resnet",
}

REQUIRED_SCAN_KEYS = [
    "min_target_count",
    "min_target_prevalence",
    "max_target_prevalence",
    "target_mean_quantile",
    "donor_quantile",
    "min_swap_count",
    "max_targets",
    "tag_prefixes",
]

REQUIRED_TRIGGER_KEYS = [
    "mode",
    "feature_name",
    "feature_value",
    "distributed_dim_count",
    "distributed_scale",
    "distributed_seed",
    "history_target_trigger_count",
    "history_non_target_trigger_count",
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

REQUIRED_RTDL_RESNET_KEYS = [
    "epochs",
    "n_blocks",
    "d_block",
    "d_hidden",
    "d_hidden_multiplier",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout1",
    "dropout2",
    "normalization",
    "eval_batch_size",
]

REQUIRED_MC_DROPOUT_UCB_KEYS = [
    "passes",
    "beta",
    "seed_offset",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("M2 triggered closed-loop false-pursuit run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "m2_triggered_closed_loop_false_pursuit")
    scan_cfg = require_nested(cfg, "target_scan", "m2_triggered_closed_loop_false_pursuit")
    trigger_cfg = require_nested(cfg, "trigger", "m2_triggered_closed_loop_false_pursuit")
    mlp_cfg = require_nested(cfg, "mlp", "m2_triggered_closed_loop_false_pursuit")
    tabular_cfg = require_nested(cfg, "tabular_torch", "m2_triggered_closed_loop_false_pursuit")
    resnet_cfg = require_nested(cfg, "rtdl_resnet", "m2_triggered_closed_loop_false_pursuit")
    require_keys(scan_cfg, REQUIRED_SCAN_KEYS, "m2_triggered_closed_loop_false_pursuit.target_scan")
    require_keys(trigger_cfg, REQUIRED_TRIGGER_KEYS, "m2_triggered_closed_loop_false_pursuit.trigger")
    require_choice(
        trigger_cfg,
        "mode",
        {"explicit_column", "distributed_noise"},
        "m2_triggered_closed_loop_false_pursuit.trigger",
    )
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "m2_triggered_closed_loop_false_pursuit.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "m2_triggered_closed_loop_false_pursuit.tabular_torch",
    )
    require_keys(
        resnet_cfg,
        REQUIRED_RTDL_RESNET_KEYS,
        "m2_triggered_closed_loop_false_pursuit.rtdl_resnet",
    )
    require_choice(
        cfg,
        "acquisition",
        {"top_mean", "epsilon_greedy", "mc_dropout_ucb"},
        "m2_triggered_closed_loop_false_pursuit",
    )
    if cfg["acquisition"] == "mc_dropout_ucb":
        ucb_cfg = require_nested(
            cfg,
            "mc_dropout_ucb",
            "m2_triggered_closed_loop_false_pursuit",
        )
        require_keys(
            ucb_cfg,
            REQUIRED_MC_DROPOUT_UCB_KEYS,
            "m2_triggered_closed_loop_false_pursuit.mc_dropout_ucb",
        )
    require_choice(cfg, "device", {"cpu", "cuda"}, "m2_triggered_closed_loop_false_pursuit")
    require_list_values(
        cfg,
        "models",
        SUPPORTED_MODELS,
        "m2_triggered_closed_loop_false_pursuit",
    )
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "m2_triggered_closed_loop_false_pursuit",
    )
    return argparse.Namespace(**cfg)


def n_mutations_from_mutants(mutants: pd.Series) -> np.ndarray:
    return mutants.astype(str).map(lambda value: 0 if not value else len(value.split(":"))).to_numpy()


def build_scan_config(args: argparse.Namespace, data_path: Path) -> TargetScanConfig:
    scan_cfg = args.target_scan
    return TargetScanConfig(
        data_path=str(data_path),
        target_column=args.target_column,
        mutant_column=args.mutant_column,
        max_rows=args.max_rows,
        random_state=args.random_state,
        min_target_count=scan_cfg["min_target_count"],
        min_target_prevalence=scan_cfg["min_target_prevalence"],
        max_target_prevalence=scan_cfg["max_target_prevalence"],
        target_mean_quantile=scan_cfg["target_mean_quantile"],
        donor_quantile=scan_cfg["donor_quantile"],
        min_swap_count=scan_cfg["min_swap_count"],
        max_targets=scan_cfg["max_targets"],
        tag_prefixes=tuple(scan_cfg["tag_prefixes"]),
    )


def fit_predictor(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> Any:
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
    if model_name == "rtdl_resnet":
        resnet_cfg = args.rtdl_resnet
        return fit_rtdl_resnet_predictor(
            x_train,
            y_train,
            seed=seed,
            epochs=resnet_cfg["epochs"],
            n_blocks=resnet_cfg["n_blocks"],
            d_block=resnet_cfg["d_block"],
            d_hidden=resnet_cfg["d_hidden"],
            d_hidden_multiplier=resnet_cfg["d_hidden_multiplier"],
            batch_size=resnet_cfg["batch_size"],
            learning_rate=resnet_cfg["learning_rate"],
            weight_decay=resnet_cfg["weight_decay"],
            dropout1=resnet_cfg["dropout1"],
            dropout2=resnet_cfg["dropout2"],
            normalization=resnet_cfg["normalization"],
            device=args.device,
            eval_batch_size=resnet_cfg["eval_batch_size"],
        )
    raise ValueError(f"unknown model: {model_name}")


def select_batch(
    candidate_ids: np.ndarray,
    ranked: np.ndarray,
    seed: int,
    round_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.acquisition in {"top_mean", "mc_dropout_ucb"}:
        return ranked[: args.batch_size]
    rng = np.random.default_rng(seed * args.exploration_seed_multiplier + round_idx)
    explore_n = int(round(args.batch_size * args.epsilon))
    exploit_n = int(args.batch_size - explore_n)
    exploit_ids = ranked[:exploit_n]
    remaining = np.array(
        [idx for idx in candidate_ids if idx not in set(exploit_ids)],
        dtype=int,
    )
    if explore_n > 0 and len(remaining) > 0:
        explore_ids = rng.choice(
            remaining,
            size=min(explore_n, len(remaining)),
            replace=False,
        )
        return np.concatenate([exploit_ids, explore_ids]).astype(int)
    return exploit_ids


def score_candidates_for_acquisition(
    predictor: Any,
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
    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    initial_history_rows: list[pd.DataFrame] = []
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
                        "is_triggered_target": triggered_target_mask[
                            base_history_ids
                        ].astype(int),
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
                    toggle_delta = trigger_toggle_delta(
                        pred_trigger_on=pred_trigger_on,
                        pred_trigger_off=pred_trigger_off,
                        target_mask=target_mask,
                        candidate_mask=candidate_mask,
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
                                triggered_candidate_mask.sum()
                            ),
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
                                "mutant": df.loc[record_id, args.mutant_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "acquisition_score": float(acquisition_score[record_id]),
                                "acquisition_uncertainty": float(
                                    acquisition_uncertainty[record_id]
                                ),
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

    pd.concat(pair_rows, ignore_index=True).to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    pd.concat(trigger_rows, ignore_index=True).to_csv(run_dir / "trigger_assignments.csv", index=False)
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
        "stage": "M2_triggered_closed_loop_false_pursuit",
        "run_dir": str(run_dir),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(len(augmented_feature_names)),
        "feature_set": "mutation",
        "target_tag": args.target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_passed": target_scan_passed,
        "swap_count": int(args.swap_count),
        "audit_size": int(args.audit_size),
        "trigger_feature_name": trigger_cfg["feature_name"],
        "trigger_feature_value": trigger_cfg["feature_value"],
        "trigger_mode": trigger_cfg["mode"],
        "audit_metric_semantics": "held-out audit decomposed by trigger and target slices at each round",
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
