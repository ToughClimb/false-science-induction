#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
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
from false_science.materials import load_matminer_dataset, material_feature_frame  # noqa: E402
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    matched_non_target_controls,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import build_audit_ids, build_history_ids, label_multiset_equal  # noqa: E402
from false_science.target_scan import git_text, make_run_dir  # noqa: E402
from scripts.b48_materials_coherence_sweep import (  # noqa: E402
    recorded_labels_for_coherence_fraction,
)
from scripts.b43_materials_realistic_relinking import ordered_high_ids, ordered_low_ids  # noqa: E402
from scripts.materials_false_regulariry import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as MATERIALS_REQUIRED_CONFIG_KEYS,
)
from scripts.materials_false_regulariry import scan_tags  # noqa: E402


REQUIRED_CONFIG_KEYS = MATERIALS_REQUIRED_CONFIG_KEYS + [
    "policies",
    "policy",
    "coherence_fraction",
    "candidate_pool_size",
    "gp",
]

REQUIRED_GP_KEYS = [
    "normalize_y",
    "length_scale",
    "noise_level",
    "constant_value",
    "optimizer",
    "n_restarts_optimizer",
    "beta",
    "xi",
]

SUPPORTED_POLICIES = {
    "gp_ucb",
    "expected_improvement",
}


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("B50 reduced materials GP-BO replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b50_materials_gp_bo_replay")
    gp_cfg = require_nested(cfg, "gp", "b50_materials_gp_bo_replay")
    require_keys(gp_cfg, REQUIRED_GP_KEYS, "b50_materials_gp_bo_replay.gp")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b50_materials_gp_bo_replay")
    require_list_values(cfg, "policies", SUPPORTED_POLICIES, "b50_materials_gp_bo_replay")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "coherent_swap"},
        "b50_materials_gp_bo_replay",
    )
    if str(cfg["policy"]) not in SUPPORTED_POLICIES:
        raise ValueError("policy must be listed in supported GP policies")
    if str(cfg["policy"]) not in [str(item) for item in cfg["policies"]]:
        raise ValueError("policy must be included in policies")
    if not isinstance(cfg["candidate_pool_size"], int):
        raise TypeError("candidate_pool_size must be an integer")
    if int(cfg["candidate_pool_size"]) <= 0:
        raise ValueError("candidate_pool_size must be positive")
    coherence = cfg["coherence_fraction"]
    if not isinstance(coherence, int | float):
        raise TypeError("coherence_fraction must be numeric")
    if float(coherence) < 0.0 or float(coherence) > 1.0:
        raise ValueError("coherence_fraction must be in [0, 1]")
    for key in ["length_scale", "noise_level", "constant_value", "beta", "xi"]:
        if not isinstance(gp_cfg[key], int | float):
            raise TypeError(f"gp.{key} must be numeric")
    if not isinstance(gp_cfg["normalize_y"], bool):
        raise TypeError("gp.normalize_y must be boolean")
    if not isinstance(gp_cfg["n_restarts_optimizer"], int):
        raise TypeError("gp.n_restarts_optimizer must be an integer")
    require_choice(gp_cfg, "optimizer", {"none", "fmin_l_bfgs_b"}, "b50_materials_gp_bo_replay.gp")
    return argparse.Namespace(**cfg)


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    xi: float,
) -> np.ndarray:
    improvement = mean - float(best_observed) - float(xi)
    scores = np.zeros_like(mean, dtype=float)
    positive_std = std > 0.0
    if not positive_std.any():
        return scores
    z = improvement[positive_std] / std[positive_std]
    scores[positive_std] = improvement[positive_std] * norm.cdf(z) + std[positive_std] * norm.pdf(z)
    return np.maximum(scores, 0.0)


def acquisition_scores(
    mean: np.ndarray,
    std: np.ndarray,
    policy: str,
    beta: float,
    best_observed: float,
    xi: float,
) -> np.ndarray:
    if policy == "gp_ucb":
        return mean + float(beta) * std
    if policy == "expected_improvement":
        return expected_improvement(mean, std, best_observed=best_observed, xi=xi)
    raise ValueError(f"unknown GP acquisition policy: {policy}")


def candidate_pool_ids(
    candidate_mask: np.ndarray,
    target_mask: np.ndarray,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    candidate_ids = np.flatnonzero(candidate_mask)
    target_ids = np.flatnonzero(candidate_mask & target_mask)
    non_target_ids = np.flatnonzero(candidate_mask & (~target_mask))
    if pool_size < len(target_ids):
        ordered_targets = np.sort(target_ids.astype(int))
        return ordered_targets[:pool_size].astype(int)
    needed = int(pool_size - len(target_ids))
    rng = np.random.default_rng(seed)
    if needed > len(non_target_ids):
        chosen_non_target = np.sort(non_target_ids.astype(int))
    else:
        chosen_non_target = np.sort(
            rng.choice(non_target_ids.astype(int), size=needed, replace=False)
        )
    pool = np.sort(np.concatenate([target_ids.astype(int), chosen_non_target]).astype(int))
    if len(pool) == 0:
        raise ValueError("candidate pool is empty")
    if len(pool) > len(candidate_ids):
        raise ValueError("candidate pool exceeds available candidates")
    return pool.astype(int)


def matched_pool_control_ids(
    target_mask: np.ndarray,
    pool_ids: np.ndarray,
    n_elements: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pool_mask = np.zeros(len(target_mask), dtype=bool)
    pool_mask[pool_ids] = True
    control_ids = matched_non_target_controls(
        target_mask,
        pool_mask,
        n_elements,
        seed,
    )
    return control_ids, pool_mask


def recorded_labels_for_mode(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    target_ids: np.ndarray,
    donor_ids: np.ndarray,
    swap_count: int,
    coherence_fraction: float,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if mode == "clean":
        return true_y[history_ids].astype(float).copy(), pd.DataFrame(columns=["relinking_kind"])
    if mode == "coherent_swap":
        return recorded_labels_for_coherence_fraction(
            true_y=true_y,
            history_ids=history_ids,
            target_ids=target_ids,
            donor_ids=donor_ids,
            swap_count=swap_count,
            coherence_fraction=coherence_fraction,
            seed=seed,
        )
    if mode == "random_swap":
        return recorded_labels_for_coherence_fraction(
            true_y=true_y,
            history_ids=history_ids,
            target_ids=target_ids,
            donor_ids=donor_ids,
            swap_count=swap_count,
            coherence_fraction=0.0,
            seed=seed,
        )
    raise ValueError(f"unknown mode: {mode}")


def fit_gp_predict(
    x: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    predict_ids: np.ndarray,
    seed: int,
    gp_cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_ids])
    x_predict = scaler.transform(x[predict_ids])
    kernel = ConstantKernel(float(gp_cfg["constant_value"])) * RBF(
        length_scale=float(gp_cfg["length_scale"])
    ) + WhiteKernel(noise_level=float(gp_cfg["noise_level"]))
    optimizer_value = None if str(gp_cfg["optimizer"]) == "none" else str(gp_cfg["optimizer"])
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        optimizer=optimizer_value,
        n_restarts_optimizer=int(gp_cfg["n_restarts_optimizer"]),
        normalize_y=bool(gp_cfg["normalize_y"]),
        random_state=int(seed),
    )
    gp.fit(x_train, train_y)
    mean, std = gp.predict(x_predict, return_std=True)
    return mean.astype(float), std.astype(float)


def main() -> int:
    args = parse_args()
    df = load_matminer_dataset(args.dataset_name, args.target_column, args.composition_column)
    x_frame, tag_sets = material_feature_frame(df[args.composition_column].astype(str).tolist())
    x = x_frame.to_numpy(dtype=np.float32)
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
    target_order = ordered_low_ids(y, target_mask)
    donor_cutoff = float(np.quantile(y, args.donor_quantile))
    donor_order = ordered_high_ids(y, (~target_mask) & (y >= donor_cutoff))
    if args.swap_count > len(target_order) or args.swap_count > len(donor_order):
        raise ValueError("swap_count exceeds target or donor availability")
    target_ids = target_order[: args.swap_count].astype(int)
    donor_ids = donor_order[: args.swap_count].astype(int)
    gp_cfg = args.gp
    policy = str(args.policy)

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    relinking_rows: list[pd.DataFrame] = []
    preserved_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        base_history_ids = build_history_ids(
            n_records=len(df),
            target_ids=target_ids,
            donor_ids=donor_ids,
            background_size=args.background_size,
            seed=int(seed),
        )
        audit_ids = build_audit_ids(
            n_records=len(df),
            excluded_ids=base_history_ids,
            audit_size=args.audit_size,
            seed=int(seed) + int(args.audit_seed_offset),
        )
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True
        for mode in args.modes:
            initial_recorded, relinking = recorded_labels_for_mode(
                true_y=y,
                history_ids=base_history_ids,
                target_ids=target_ids,
                donor_ids=donor_ids,
                swap_count=int(args.swap_count),
                coherence_fraction=float(args.coherence_fraction),
                mode=str(mode),
                seed=int(seed),
            )
            if not relinking.empty:
                relinking = relinking.copy()
                relinking["seed"] = int(seed)
                relinking["mode"] = str(mode)
                relinking_rows.append(relinking)
            preserved_rows.append(
                {
                    "seed": int(seed),
                    "mode": str(mode),
                    "label_multiset_preserved": label_multiset_equal(
                        y[base_history_ids],
                        initial_recorded,
                    ),
                }
            )
            train_ids = base_history_ids.copy()
            train_y = initial_recorded.copy()
            selected_so_far: list[int] = []
            for round_idx in range(args.rounds):
                observed = np.zeros(len(df), dtype=bool)
                observed[train_ids] = True
                candidate_mask = (~observed) & (~audit_mask)
                pool_ids = candidate_pool_ids(
                    candidate_mask=candidate_mask,
                    target_mask=target_mask,
                    pool_size=int(args.candidate_pool_size),
                    seed=int(seed) + 1000 * int(round_idx),
                )
                mean, std = fit_gp_predict(
                    x=x,
                    y=y,
                    train_ids=train_ids,
                    train_y=train_y,
                    predict_ids=pool_ids,
                    seed=int(seed) + round_idx,
                    gp_cfg=gp_cfg,
                )
                audit_mean, _ = fit_gp_predict(
                    x=x,
                    y=y,
                    train_ids=train_ids,
                    train_y=train_y,
                    predict_ids=audit_ids,
                    seed=int(seed) + round_idx,
                    gp_cfg=gp_cfg,
                )
                scores = acquisition_scores(
                    mean,
                    std,
                    policy=policy,
                    beta=float(gp_cfg["beta"]),
                    best_observed=float(np.max(train_y)),
                    xi=float(gp_cfg["xi"]),
                )
                ranked_order = np.argsort(-scores)
                batch_ids = pool_ids[ranked_order[: args.batch_size]].astype(int)
                selected_so_far.extend(batch_ids.tolist())
                selected_target = target_mask[batch_ids]
                pred_full = np.full(len(df), np.nan, dtype=float)
                pred_full[pool_ids] = mean
                control_ids, pool_mask = matched_pool_control_ids(
                    target_mask,
                    pool_ids,
                    n_elements,
                    int(seed) + round_idx,
                )
                round_rows.append(
                    {
                        "seed": int(seed),
                        "mode": str(mode),
                        "model": "sklearn_gp",
                        "policy": policy,
                        "round": int(round_idx),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_pool_size": int(len(pool_ids)),
                        "candidate_pool_target_count": int(target_mask[pool_ids].sum()),
                        "batch_target_count": int(selected_target.sum()),
                        "batch_target_fraction": float(selected_target.mean()),
                        "cumulative_target_count": int(target_mask[selected_so_far].sum()),
                        "cumulative_selected_count": int(len(selected_so_far)),
                        "cumulative_target_fraction": float(
                            target_mask[selected_so_far].mean()
                        ),
                        "batch_true_mean": float(np.mean(y[batch_ids])),
                        "batch_target_true_mean": float(np.mean(y[batch_ids[selected_target]]))
                        if selected_target.any()
                        else float("nan"),
                        "fas": false_association_strength(
                            pred_full,
                            target_mask,
                            control_ids,
                            pool_mask,
                        ),
                        "target_topk_fraction": target_topk_fraction(
                            pred_full,
                            target_mask,
                            pool_mask,
                            min(args.top_k, int(len(pool_ids))),
                        ),
                        "target_rank_percentile": target_mean_rank_percentile(
                            pred_full,
                            target_mask,
                            pool_mask,
                        ),
                        "mae_audit": float(mean_absolute_error(y[audit_ids], audit_mean)),
                        "r2_audit": float(r2_score(y[audit_ids], audit_mean)),
                    }
                )
                for rank, record_id in enumerate(batch_ids):
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "mode": str(mode),
                            "model": "sklearn_gp",
                            "policy": policy,
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "composition": df.loc[record_id, args.composition_column],
                            "true_label": float(y[record_id]),
                            "predicted_mean": float(mean[ranked_order[rank]]),
                            "predicted_std": float(std[ranked_order[rank]]),
                            "acquisition_score": float(scores[ranked_order[rank]]),
                            "is_target": int(target_mask[record_id]),
                        }
                    )
                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, y[batch_ids]]).astype(float)

    rounds = pd.DataFrame(round_rows)
    reference = rounds[rounds["mode"] == "random_swap"][
        ["seed", "round", "cumulative_target_count", "fas"]
    ].rename(
        columns={
            "cumulative_target_count": "reference_cumulative_target_count",
            "fas": "reference_fas",
        }
    )
    merged = rounds.merge(reference, on=["seed", "round"], how="left")
    merged["cumulative_target_count_excess_vs_random"] = (
        merged["cumulative_target_count"] - merged["reference_cumulative_target_count"]
    )
    merged["fas_lift_vs_random"] = merged["fas"] - merged["reference_fas"]
    final_rounds = merged.loc[merged.groupby(["mode", "seed"])["round"].idxmax()]
    final_summary = final_rounds.groupby("mode", as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_target_count_excess_vs_random=("cumulative_target_count_excess_vs_random", "mean"),
        final_mae_audit_mean=("mae_audit", "mean"),
        final_r2_audit_mean=("r2_audit", "mean"),
    )
    aggregate = merged.groupby("mode", as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_target_fraction=("batch_target_fraction", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        rank_percentile_mean=("target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
    )
    summary = aggregate.merge(final_summary, on="mode", how="left").sort_values("mode")

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    if relinking_rows:
        pd.concat(relinking_rows, ignore_index=True).to_csv(run_dir / "relinking_map.csv", index=False)
    else:
        pd.DataFrame(columns=["seed", "mode", "relinking_kind"]).to_csv(
            run_dir / "relinking_map.csv",
            index=False,
        )
    pd.DataFrame(preserved_rows).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    x_frame.head(0).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "b50_materials_gp_bo_replay",
        "run_dir": str(run_dir),
        "dataset_name": args.dataset_name,
        "policy": policy,
        "n_records": int(len(df)),
        "n_features": int(x.shape[1]),
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "swap_count": int(args.swap_count),
        "coherence_fraction": float(args.coherence_fraction),
        "candidate_pool_size": int(args.candidate_pool_size),
        "all_modes_label_multiset_preserved": bool(
            pd.DataFrame(preserved_rows)["label_multiset_preserved"].all()
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
