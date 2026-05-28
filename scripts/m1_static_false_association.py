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

from false_science.features import mutation_feature_frame
from false_science.config import (
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_choice,
    require_keys,
    require_list_values,
    require_nested,
)
from false_science.esm_features import (
    PROTEINGYM_GFP_AEQVI_SEQUENCE,
    load_or_compute_esm2_embeddings,
)
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
from false_science.models import fit_predict_torch_mlp, fit_predict_xgboost
from false_science.protein import load_gfp_csv
from false_science.target_scan import (
    TargetScanConfig,
    attach_tags,
    file_sha256,
    git_text,
    make_run_dir,
    scan_target_regions,
    select_swap_pairs,
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
    "seeds",
    "models",
    "feature_set",
    "feature_cache_root",
    "esm_model_name",
    "esm_batch_size",
    "top_k",
    "device",
    "allow_nonpassing_target",
    "target_scan",
    "mlp",
    "xgboost",
]

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

REQUIRED_MLP_KEYS = [
    "epochs",
    "hidden_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
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
    config_path = parse_config_arg("M1 static false-association run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "m1_static_false_association")
    scan_cfg = require_nested(cfg, "target_scan", "m1_static_false_association")
    mlp_cfg = require_nested(cfg, "mlp", "m1_static_false_association")
    xgb_cfg = require_nested(cfg, "xgboost", "m1_static_false_association")
    require_keys(scan_cfg, REQUIRED_SCAN_KEYS, "m1_static_false_association.target_scan")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "m1_static_false_association.mlp")
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "m1_static_false_association.xgboost")
    require_choice(
        cfg,
        "feature_set",
        {"mutation", "esm2"},
        "m1_static_false_association",
    )
    require_choice(cfg, "device", {"cpu", "cuda"}, "m1_static_false_association")
    require_list_values(
        cfg,
        "models",
        {"mlp", "xgboost"},
        "m1_static_false_association",
    )
    require_list_values(
        cfg,
        "modes",
        {
            "clean",
            "random_swap",
            "targeted_swap",
            "donor_only_swap",
            "target_only_high_relabel",
        },
        "m1_static_false_association",
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


def fit_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> Any:
    if model_name == "xgboost":
        xgb_cfg = args.xgboost
        return fit_predict_xgboost(
            x_train,
            y_train,
            x_eval,
            y_eval,
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
    if model_name == "mlp":
        mlp_cfg = args.mlp
        return fit_predict_torch_mlp(
            x_train,
            y_train,
            x_eval,
            y_eval,
            seed=seed,
            epochs=mlp_cfg["epochs"],
            hidden_dim=mlp_cfg["hidden_dim"],
            batch_size=mlp_cfg["batch_size"],
            learning_rate=mlp_cfg["learning_rate"],
            weight_decay=mlp_cfg["weight_decay"],
            dropout=mlp_cfg["dropout"],
            device=args.device,
        )
    raise ValueError(f"unknown model: {model_name}")


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
    feature_metadata: dict[str, object]
    if args.feature_set == "mutation":
        X_df = mutation_feature_frame(df, args.mutant_column)
        X = X_df.to_numpy(dtype=np.float32)
        feature_metadata = {
            "feature_set": "mutation",
            "n_features": int(X.shape[1]),
        }
    else:
        X, feature_metadata = load_or_compute_esm2_embeddings(
            df,
            data_path=data_path,
            mutant_column=args.mutant_column,
            cache_root=args.feature_cache_root,
            model_name=args.esm_model_name,
            batch_size=args.esm_batch_size,
            device=args.device,
            wild_type_sequence=PROTEINGYM_GFP_AEQVI_SEQUENCE,
        )
        X_df = pd.DataFrame(columns=[f"esm2_{idx}" for idx in range(X.shape[1])])
        feature_metadata["feature_set"] = "esm2"
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
    pairs = select_swap_pairs(
        df,
        tag_sets,
        args.target_tag,
        scan_cfg,
        swap_count=args.swap_count,
    )
    if len(pairs) < args.swap_count:
        raise ValueError(
            f"only {len(pairs)} swap pairs available for requested {args.swap_count}"
        )

    run_dir = make_run_dir(args.output_root, args.tag)
    metrics_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    n_mutations = n_mutations_from_mutants(df[args.mutant_column])

    for seed in args.seeds:
        target_ids = pairs["target_record_id"].to_numpy(dtype=int)
        donor_ids = pairs["donor_record_id"].to_numpy(dtype=int)
        history_ids = build_history_ids(
            n_records=len(df),
            target_ids=target_ids,
            donor_ids=donor_ids,
            background_size=args.background_size,
            seed=seed,
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=history_ids,
            audit_size=args.audit_size,
            seed=seed + args.audit_seed_offset,
        )
        candidate_mask = np.ones(len(df), dtype=bool)
        candidate_mask[history_ids] = False
        candidate_mask[audit_ids] = False
        control_ids = matched_non_target_controls(
            target_mask=target_mask,
            candidate_mask=candidate_mask,
            n_mutations=n_mutations,
            seed=seed,
        )

        for mode in args.modes:
            recorded_y = recorded_labels_for_history(
                true_y=y,
                history_ids=history_ids,
                pairs=pairs,
                mode=mode,
                seed=seed,
            )
            true_history_y = y[history_ids]
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "mode": mode,
                        "record_id": history_ids,
                        "true_label": true_history_y,
                        "recorded_label": recorded_y,
                        "is_target": target_mask[history_ids].astype(int),
                    }
                )
            )

            for model_name in args.models:
                result = fit_model(
                    model_name,
                    X[history_ids],
                    recorded_y,
                    X,
                    y,
                    seed,
                    args,
                )

                pred = result.predictions
                audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                fas = false_association_strength(
                    pred,
                    target_mask=target_mask,
                    control_ids=control_ids,
                    candidate_mask=candidate_mask,
                )
                true_fas = false_association_strength(
                    y,
                    target_mask=target_mask,
                    control_ids=control_ids,
                    candidate_mask=candidate_mask,
                )
                topk_fraction = target_topk_fraction(
                    pred,
                    target_mask=target_mask,
                    candidate_mask=candidate_mask,
                    k=args.top_k,
                )
                rank_percentile = target_mean_rank_percentile(
                    pred,
                    target_mask=target_mask,
                    candidate_mask=candidate_mask,
                )
                metrics_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "model": model_name,
                        "target_tag": args.target_tag,
                        "history_size": int(len(history_ids)),
                        "swap_count": int(len(pairs)),
                        "background_size": int(args.background_size),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_target_count": int((candidate_mask & target_mask).sum()),
                        "history_label_multiset_preserved": label_multiset_equal(
                            true_history_y,
                            recorded_y,
                        ),
                        "mae_all": result.mae,
                        "r2_all": result.r2,
                        "mae_audit": audit_mae,
                        "r2_audit": audit_r2,
                        "fas": fas,
                        "true_fas": true_fas,
                        "target_topk_fraction": topk_fraction,
                        "target_mean_rank_percentile": rank_percentile,
                        "target_true_mean_candidate": float(
                            np.mean(y[candidate_mask & target_mask])
                        ),
                        "control_true_mean_candidate": float(np.mean(y[control_ids])),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    baseline = metrics[metrics["mode"] == "clean"][
        ["seed", "model", "fas", "target_topk_fraction", "target_mean_rank_percentile"]
    ].rename(
        columns={
            "fas": "fas_clean",
            "target_topk_fraction": "target_topk_fraction_clean",
            "target_mean_rank_percentile": "target_rank_percentile_clean",
        }
    )
    random_baseline = metrics[metrics["mode"] == "random_swap"][
        ["seed", "model", "fas", "target_topk_fraction", "target_mean_rank_percentile"]
    ].rename(
        columns={
            "fas": "fas_random_swap",
            "target_topk_fraction": "target_topk_fraction_random_swap",
            "target_mean_rank_percentile": "target_rank_percentile_random_swap",
        }
    )
    metrics = metrics.merge(baseline, on=["seed", "model"], how="left")
    metrics = metrics.merge(random_baseline, on=["seed", "model"], how="left")
    metrics["fas_lift_vs_clean"] = metrics["fas"] - metrics["fas_clean"]
    metrics["fas_lift_vs_random"] = metrics["fas"] - metrics["fas_random_swap"]
    metrics["topk_lift_vs_clean"] = (
        metrics["target_topk_fraction"] - metrics["target_topk_fraction_clean"]
    )
    metrics["topk_lift_vs_random"] = (
        metrics["target_topk_fraction"] - metrics["target_topk_fraction_random_swap"]
    )
    metrics["rank_lift_vs_clean"] = (
        metrics["target_mean_rank_percentile"]
        - metrics["target_rank_percentile_clean"]
    )
    metrics["rank_lift_vs_random"] = (
        metrics["target_mean_rank_percentile"]
        - metrics["target_rank_percentile_random_swap"]
    )

    summary = (
        metrics.groupby(["model", "mode"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
            mae_audit_mean=("mae_audit", "mean"),
            r2_audit_mean=("r2_audit", "mean"),
            fas_mean=("fas", "mean"),
            fas_lift_vs_clean_mean=("fas_lift_vs_clean", "mean"),
            fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
            topk_fraction_mean=("target_topk_fraction", "mean"),
            topk_lift_vs_clean_mean=("topk_lift_vs_clean", "mean"),
            topk_lift_vs_random_mean=("topk_lift_vs_random", "mean"),
            rank_percentile_mean=("target_mean_rank_percentile", "mean"),
            rank_lift_vs_clean_mean=("rank_lift_vs_clean", "mean"),
            rank_lift_vs_random_mean=("rank_lift_vs_random", "mean"),
        )
        .sort_values(["model", "mode"])
    )

    pairs.to_csv(run_dir / "targeted_swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "history_labels.csv", index=False)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    X_df.head(0).to_csv(run_dir / "feature_columns.csv", index=False)

    config = config_for_metadata(vars(args))
    metadata = {
        "stage": "M1_static_false_association",
        "run_dir": str(run_dir),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_metadata": feature_metadata,
        "target_tag": args.target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_passed": target_scan_passed,
        "swap_count": int(len(pairs)),
        "audit_size": int(args.audit_size),
        "audit_metric_semantics": "held-out non-history, non-acquisition records per seed",
        "git_commit": git_text(["rev-parse", "HEAD"]) or "unknown",
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
