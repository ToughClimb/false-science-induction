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
    matched_non_target_controls,
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
    fit_predict_torch_mlp,
    fit_predict_torch_tabular,
    fit_predict_xgboost,
)
from false_science.target_scan import git_text, make_run_dir


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
    "seeds",
    "modes",
    "models",
    "rounds",
    "batch_size",
    "top_k",
    "device",
    "mlp",
    "tabular_torch",
    "xgboost",
]

SUPPORTED_MODELS = {
    "mlp",
    "tabm_mini",
    "xgboost",
}

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
    config_path = parse_config_arg("Materials false-regularity induction run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "materials_false_regularity")
    mlp_cfg = require_nested(cfg, "mlp", "materials_false_regularity")
    tabular_cfg = require_nested(cfg, "tabular_torch", "materials_false_regularity")
    xgb_cfg = require_nested(cfg, "xgboost", "materials_false_regularity")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "materials_false_regularity.mlp")
    require_keys(tabular_cfg, REQUIRED_TABULAR_TORCH_KEYS, "materials_false_regularity.tabular_torch")
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "materials_false_regularity.xgboost")
    require_choice(cfg, "device", {"cpu", "cuda"}, "materials_false_regularity")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "materials_false_regularity")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "materials_false_regularity",
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
        target_mean = float(np.mean(target_y))
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        rows.append(
            {
                "tag": tag,
                "tag_kind": tag.split("=", 1)[0],
                "target_count": count,
                "target_prevalence": prevalence,
                "target_mean": target_mean,
                "target_median": float(np.median(target_y)),
                "target_q10": float(np.quantile(target_y, 0.10)),
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
    n = min(len(target_order), len(donor_order), swap_count)
    return pd.DataFrame(
        {
            "pair_id": np.arange(n, dtype=int),
            "target_record_id": target_order[:n].astype(int),
            "donor_record_id": donor_order[:n].astype(int),
            "target_true_label": y[target_order[:n]],
            "donor_true_label": y[donor_order[:n]],
            "target_recorded_label_after_swap": y[donor_order[:n]],
            "donor_recorded_label_after_swap": y[target_order[:n]],
            "target_tag": target_tag,
        }
    )


def fit_model(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    args: argparse.Namespace,
):
    if model == "mlp":
        mlp_cfg = args.mlp
        return fit_predict_torch_mlp(
            x[train_ids],
            train_y,
            x,
            y,
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
    if model == "tabm_mini":
        tabular_cfg = args.tabular_torch
        return fit_predict_torch_tabular(
            x[train_ids],
            train_y,
            x,
            y,
            seed=seed,
            architecture=model,
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
    if model == "xgboost":
        xgb_cfg = args.xgboost
        return fit_predict_xgboost(
            x[train_ids],
            train_y,
            x,
            y,
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
    raise ValueError(f"unknown model: {model}")


def main() -> int:
    args = parse_args()
    df = load_matminer_dataset(args.dataset_name, args.target_column, args.composition_column)
    x_frame, tag_sets = material_feature_frame(df[args.composition_column].astype(str).tolist())
    x = x_frame.to_numpy(dtype=np.float32)
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
    pairs = select_pairs(y, tag_sets, target_tag, args.donor_quantile, args.swap_count)
    if len(pairs) < args.swap_count:
        raise ValueError(f"only {len(pairs)} pairs available")

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []

    for seed in args.seeds:
        target_ids = pairs["target_record_id"].to_numpy(dtype=int)
        donor_ids = pairs["donor_record_id"].to_numpy(dtype=int)
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=target_ids,
            donor_ids=donor_ids,
            background_size=args.background_size,
            seed=seed,
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=base_history_ids,
            audit_size=args.audit_size,
            seed=seed + args.audit_seed_offset,
        )
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True

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
                    }
                )
            )
            for model in args.models:
                train_ids = base_history_ids.copy()
                train_y = initial_recorded.copy()
                selected_so_far: list[int] = []
                for round_idx in range(args.rounds):
                    observed = np.zeros(len(df), dtype=bool)
                    observed[train_ids] = True
                    candidate_mask = (~observed) & (~audit_mask)
                    control_ids = matched_non_target_controls(
                        target_mask,
                        candidate_mask,
                        n_elements,
                        seed + round_idx,
                    )
                    result = fit_model(model, x, y, train_ids, train_y, seed + round_idx, args)
                    pred = result.predictions
                    audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-pred[candidate_ids])]
                    batch_ids = ranked[: args.batch_size]
                    selected_so_far.extend(batch_ids.tolist())
                    selected_target = target_mask[batch_ids]
                    round_rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "model": model,
                            "round": round_idx,
                            "batch_target_count": int(selected_target.sum()),
                            "batch_target_fraction": float(selected_target.mean()),
                            "cumulative_target_count": int(target_mask[selected_so_far].sum()),
                            "cumulative_selected_count": int(len(selected_so_far)),
                            "cumulative_target_fraction": float(target_mask[selected_so_far].mean()),
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_target_true_mean": float(np.mean(y[batch_ids[selected_target]]))
                            if selected_target.any()
                            else float("nan"),
                            "fas": false_association_strength(
                                pred,
                                target_mask,
                                control_ids,
                                candidate_mask,
                            ),
                            "true_fas": false_association_strength(
                                y,
                                target_mask,
                                control_ids,
                                candidate_mask,
                            ),
                            "target_topk_fraction": target_topk_fraction(
                                pred,
                                target_mask,
                                candidate_mask,
                                args.top_k,
                            ),
                            "target_rank_percentile": target_mean_rank_percentile(
                                pred,
                                target_mask,
                                candidate_mask,
                            ),
                            "mae_all": result.mae,
                            "r2_all": result.r2,
                            "mae_audit": audit_mae,
                            "r2_audit": audit_r2,
                        }
                    )
                    for rank, record_id in enumerate(batch_ids):
                        selection_rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "model": model,
                                "round": round_idx,
                                "rank": rank,
                                "record_id": int(record_id),
                                "composition": df.loc[record_id, args.composition_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "is_target": int(target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, y[batch_ids]])

    rounds = pd.DataFrame(round_rows)
    clean = rounds[rounds["mode"] == "clean"][
        ["seed", "model", "round", "batch_target_fraction", "cumulative_target_count", "fas"]
    ].rename(
        columns={
            "batch_target_fraction": "clean_batch_target_fraction",
            "cumulative_target_count": "clean_cumulative_target_count",
            "fas": "clean_fas",
        }
    )
    random = rounds[rounds["mode"] == "random_swap"][
        ["seed", "model", "round", "batch_target_fraction", "cumulative_target_count", "fas"]
    ].rename(
        columns={
            "batch_target_fraction": "random_batch_target_fraction",
            "cumulative_target_count": "random_cumulative_target_count",
            "fas": "random_fas",
        }
    )
    rounds = rounds.merge(clean, on=["seed", "model", "round"], how="left")
    rounds = rounds.merge(random, on=["seed", "model", "round"], how="left")
    rounds["batch_target_fraction_lift_vs_clean"] = (
        rounds["batch_target_fraction"] - rounds["clean_batch_target_fraction"]
    )
    rounds["batch_target_fraction_lift_vs_random"] = (
        rounds["batch_target_fraction"] - rounds["random_batch_target_fraction"]
    )
    rounds["cumulative_target_count_excess_vs_clean"] = (
        rounds["cumulative_target_count"] - rounds["clean_cumulative_target_count"]
    )
    rounds["cumulative_target_count_excess_vs_random"] = (
        rounds["cumulative_target_count"] - rounds["random_cumulative_target_count"]
    )
    rounds["fas_lift_vs_clean"] = rounds["fas"] - rounds["clean_fas"]
    rounds["fas_lift_vs_random"] = rounds["fas"] - rounds["random_fas"]

    final_rounds = rounds.loc[rounds.groupby(["model", "mode", "seed"])["round"].idxmax()]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_cumulative_target_count_std=("cumulative_target_count", "std"),
        final_target_count_excess_vs_clean=("cumulative_target_count_excess_vs_clean", "mean"),
        final_target_count_excess_vs_random=("cumulative_target_count_excess_vs_random", "mean"),
        final_mae_audit_mean=("mae_audit", "mean"),
        final_r2_audit_mean=("r2_audit", "mean"),
    )
    aggregate_summary = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_target_fraction=("batch_target_fraction", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        rank_percentile_mean=("target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate_summary.merge(final_summary, on=["model", "mode"], how="left")

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "targeted_swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "history_labels.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    x_frame.head(0).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "materials_false_regularity",
        "run_dir": str(run_dir),
        "dataset_name": args.dataset_name,
        "n_records": int(len(df)),
        "n_features": int(x.shape[1]),
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(len(pairs)),
        "audit_size": int(args.audit_size),
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
