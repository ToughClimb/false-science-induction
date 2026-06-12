#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
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

from false_science.cameo import (  # noqa: E402
    build_cameo_dataset,
    scan_cameo_region_targets,
    select_cameo_target_region,
    select_low_target_and_high_donor_ids,
)
from false_science.config import (  # noqa: E402
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_choice,
    require_keys,
    require_list_values,
    require_nested,
)
from false_science.metrics import (  # noqa: E402
    false_association_strength,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import label_multiset_equal, recorded_labels_for_history  # noqa: E402
from false_science.target_scan import file_sha256, git_text, make_run_dir  # noqa: E402
from false_science.triggers import triggered_swap_pairs  # noqa: E402
from scripts.b31_cameo_retrospective_replay import (  # noqa: E402
    build_audit_ids,
    build_history_ids,
    target_controls,
)


REQUIRED_CONFIG_KEYS = [
    "data_zip",
    "output_root",
    "tag",
    "target_column",
    "feature_space",
    "xrd_pca_components",
    "pca_seed",
    "target_region",
    "min_target_count",
    "donor_quantile",
    "swap_count",
    "background_size",
    "audit_size",
    "audit_seed_offset",
    "seeds",
    "modes",
    "rounds",
    "batch_size",
    "top_k",
    "acquisition",
    "ucb_beta",
    "gp",
]

REQUIRED_GP_KEYS = [
    "normalize_y",
    "length_scale",
    "noise_level",
    "constant_value",
    "optimizer",
    "n_restarts_optimizer",
]


def parse_args() -> Namespace:
    config_path = parse_config_arg("B60 CAMEO-like phase-region GP-UCB replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b60_cameo_phase_region_gp_replay")
    gp_cfg = require_nested(cfg, "gp", "b60_cameo_phase_region_gp_replay")
    require_keys(gp_cfg, REQUIRED_GP_KEYS, "b60_cameo_phase_region_gp_replay.gp")
    require_choice(
        cfg,
        "acquisition",
        {"phase_region_gp_ucb"},
        "b60_cameo_phase_region_gp_replay",
    )
    require_choice(
        cfg,
        "feature_space",
        {"composition", "composition_xrd_pca"},
        "b60_cameo_phase_region_gp_replay",
    )
    require_choice(
        cfg,
        "target_column",
        {"magnetization_raw", "magnetization_modified"},
        "b60_cameo_phase_region_gp_replay",
    )
    require_choice(
        gp_cfg,
        "optimizer",
        {"none", "fmin_l_bfgs_b"},
        "b60_cameo_phase_region_gp_replay.gp",
    )
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "b60_cameo_phase_region_gp_replay",
    )
    mode_set = set(str(mode) for mode in cfg["modes"])
    for required_mode in ["clean", "random_swap"]:
        if required_mode not in mode_set:
            raise ValueError(f"modes must include {required_mode} for reference summaries")
    for key in ["length_scale", "noise_level", "constant_value"]:
        if not isinstance(gp_cfg[key], int | float):
            raise TypeError(f"gp.{key} must be numeric")
    if not isinstance(gp_cfg["normalize_y"], bool):
        raise TypeError("gp.normalize_y must be boolean")
    if not isinstance(gp_cfg["n_restarts_optimizer"], int):
        raise TypeError("gp.n_restarts_optimizer must be an integer")
    return Namespace(**cfg, config_path=str(config_path))


def region_candidate_scores(
    candidate_ids: np.ndarray,
    regions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    beta: float,
) -> list[dict[str, object]]:
    candidates = np.asarray(candidate_ids, dtype=int)
    score = np.asarray(mean, dtype=float) + float(beta) * np.asarray(std, dtype=float)
    rows: list[dict[str, object]] = []
    for region in sorted(set(np.asarray(regions, dtype=int)[candidates].tolist())):
        region_ids = candidates[np.asarray(regions, dtype=int)[candidates] == int(region)]
        local_scores = score[region_ids]
        local_order = np.argsort(-local_scores)
        top_id = int(region_ids[local_order[0]])
        rows.append(
            {
                "region": int(region),
                "candidate_count": int(len(region_ids)),
                "region_score": float(local_scores[local_order[0]]),
                "region_mean_score": float(np.mean(local_scores)),
                "region_median_score": float(np.median(local_scores)),
                "top_record_id": top_id,
                "top_record_mean": float(mean[top_id]),
                "top_record_std": float(std[top_id]),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["region_score"]), int(row["region"])))


def select_phase_region_batch(
    candidate_ids: np.ndarray,
    regions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    beta: float,
    batch_size: int,
) -> np.ndarray:
    candidates = np.asarray(candidate_ids, dtype=int)
    score = np.asarray(mean, dtype=float) + float(beta) * np.asarray(std, dtype=float)
    selected: list[int] = []
    for row in region_candidate_scores(candidates, regions, mean, std, beta):
        region = int(row["region"])
        region_ids = candidates[np.asarray(regions, dtype=int)[candidates] == region]
        ranked = region_ids[np.argsort(-score[region_ids])]
        for record_id in ranked.tolist():
            if len(selected) < int(batch_size):
                selected.append(int(record_id))
        if len(selected) >= int(batch_size):
            break
    return np.asarray(selected, dtype=int)


def feature_matrix_for_space(dataset: object, feature_space: str) -> np.ndarray:
    if feature_space == "composition":
        frame = dataset.frame
        return frame[["Fe", "Ga", "Pd"]].to_numpy(dtype=np.float32)
    if feature_space == "composition_xrd_pca":
        return np.asarray(dataset.x, dtype=np.float32)
    raise ValueError(f"unknown feature_space: {feature_space}")


def fit_gp_predict(
    x: np.ndarray,
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


def summarize_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final = rounds.loc[rounds.groupby(["mode", "seed"])["round"].idxmax()]
    return (
        final.groupby("mode", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            final_cumulative_target_count=("cumulative_target_count", "mean"),
            final_cumulative_target_excess_vs_clean=(
                "cumulative_target_count_excess_vs_clean",
                "mean",
            ),
            final_cumulative_target_excess_vs_random=(
                "cumulative_target_count_excess_vs_random",
                "mean",
            ),
            final_target_rank_percentile=("target_rank_percentile", "mean"),
            final_ucb_fas_lift_vs_random=("ucb_fas_lift_vs_random", "mean"),
            selected_true_mean=("batch_true_mean", "mean"),
            selected_target_true_mean=("batch_target_true_mean", "mean"),
            audit_mae=("audit_mae", "mean"),
            audit_r2=("audit_r2", "mean"),
        )
        .sort_values("mode")
    )


def main() -> int:
    args = parse_args()
    data_zip = Path(args.data_zip)
    dataset = build_cameo_dataset(
        data_zip,
        xrd_pca_components=int(args.xrd_pca_components),
        pca_seed=int(args.pca_seed),
        target_column=str(args.target_column),
    )
    frame = dataset.frame.copy()
    y = dataset.y
    x_gp = feature_matrix_for_space(dataset, str(args.feature_space))
    regions = frame[dataset.region_column].to_numpy(dtype=int)
    scan = scan_cameo_region_targets(
        y,
        regions,
        min_target_count=int(args.min_target_count),
        donor_quantile=float(args.donor_quantile),
    )
    target_region = select_cameo_target_region(scan, args.target_region)
    target_mask = regions == target_region
    target_swap_ids, donor_ids = select_low_target_and_high_donor_ids(
        y,
        target_mask,
        donor_quantile=float(args.donor_quantile),
        swap_count=int(args.swap_count),
    )
    pairs = triggered_swap_pairs(
        true_y=y,
        triggered_target_ids=target_swap_ids,
        donor_ids=donor_ids,
        swap_count=int(args.swap_count),
    )
    if len(pairs) != int(args.swap_count):
        raise ValueError(f"only {len(pairs)} swap pairs available")

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    feature_space = str(args.feature_space)
    model_name = f"{feature_space}_gp_phase_region_ucb"
    gp_cfg = args.gp

    for seed in args.seeds:
        required_history_ids = np.concatenate([target_swap_ids, donor_ids]).astype(int)
        base_history_ids = build_history_ids(
            n_records=len(frame),
            required_ids=required_history_ids,
            background_size=int(args.background_size),
            seed=int(seed),
            exclude_target_background_mask=target_mask,
        )
        audit_ids = build_audit_ids(
            n_records=len(frame),
            excluded_ids=base_history_ids,
            audit_size=int(args.audit_size),
            seed=int(seed) + int(args.audit_seed_offset),
        )
        audit_mask = np.zeros(len(frame), dtype=bool)
        audit_mask[audit_ids] = True

        for mode in args.modes:
            initial_recorded = recorded_labels_for_history(
                true_y=y,
                history_ids=base_history_ids,
                pairs=pairs,
                mode=str(mode),
                seed=int(seed),
            )
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "mode": str(mode),
                        "record_id": base_history_ids,
                        "dft_region": regions[base_history_ids],
                        "true_label": y[base_history_ids],
                        "recorded_label": initial_recorded,
                        "is_target_region": target_mask[base_history_ids].astype(int),
                        "is_swap_target": np.isin(base_history_ids, target_swap_ids).astype(int),
                        "is_swap_donor": np.isin(base_history_ids, donor_ids).astype(int),
                    }
                )
            )
            train_ids = base_history_ids.copy()
            train_y = initial_recorded.copy()
            selected_so_far: list[int] = []

            for round_idx in range(int(args.rounds)):
                observed_mask = np.zeros(len(frame), dtype=bool)
                observed_mask[train_ids] = True
                candidate_mask = (~observed_mask) & (~audit_mask)
                candidate_ids = np.flatnonzero(candidate_mask)
                if len(candidate_ids) == 0:
                    break

                pred_mean, pred_std = fit_gp_predict(
                    x=x_gp,
                    train_ids=train_ids,
                    train_y=train_y,
                    predict_ids=np.arange(len(frame), dtype=int),
                    seed=int(seed) * 1000 + round_idx,
                    gp_cfg=gp_cfg,
                )
                score = pred_mean + float(args.ucb_beta) * pred_std
                region_scores = region_candidate_scores(
                    candidate_ids=candidate_ids,
                    regions=regions,
                    mean=pred_mean,
                    std=pred_std,
                    beta=float(args.ucb_beta),
                )
                selected_region = int(region_scores[0]["region"])
                batch_ids = select_phase_region_batch(
                    candidate_ids=candidate_ids,
                    regions=regions,
                    mean=pred_mean,
                    std=pred_std,
                    beta=float(args.ucb_beta),
                    batch_size=int(args.batch_size),
                )
                selected_so_far.extend(batch_ids.tolist())

                for region_rank, row in enumerate(region_scores):
                    region_rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "mode": str(mode),
                            "round": int(round_idx),
                            "region_rank": int(region_rank),
                            "region": int(row["region"]),
                            "is_target_region": int(int(row["region"]) == int(target_region)),
                            "candidate_count": int(row["candidate_count"]),
                            "region_score": float(row["region_score"]),
                            "region_mean_score": float(row["region_mean_score"]),
                            "region_median_score": float(row["region_median_score"]),
                            "top_record_id": int(row["top_record_id"]),
                            "top_record_mean": float(row["top_record_mean"]),
                            "top_record_std": float(row["top_record_std"]),
                        }
                    )

                selected_target = target_mask[batch_ids]
                controls = target_controls(target_mask, candidate_mask, y)
                audit_pred = pred_mean[audit_ids]
                audit_r2 = (
                    float(r2_score(y[audit_ids], audit_pred)) if len(audit_ids) >= 2 else float("nan")
                )
                round_rows.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "mode": str(mode),
                        "round": int(round_idx),
                        "train_size": int(len(train_ids)),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_target_count": int((candidate_mask & target_mask).sum()),
                        "selected_phase_region": selected_region,
                        "selected_phase_is_target": int(selected_region == int(target_region)),
                        "batch_target_count": int(selected_target.sum()),
                        "batch_target_fraction": float(np.mean(selected_target)),
                        "cumulative_target_count": int(target_mask[selected_so_far].sum()),
                        "cumulative_selected_count": int(len(selected_so_far)),
                        "cumulative_target_fraction": float(np.mean(target_mask[selected_so_far])),
                        "batch_true_mean": float(np.mean(y[batch_ids])),
                        "batch_target_true_mean": float(np.mean(y[batch_ids[selected_target]]))
                        if selected_target.any()
                        else float("nan"),
                        "target_score_mean": float(np.mean(score[candidate_mask & target_mask]))
                        if np.any(candidate_mask & target_mask)
                        else float("nan"),
                        "control_score_mean": float(np.mean(score[controls]))
                        if len(controls)
                        else float("nan"),
                        "fas_target": false_association_strength(
                            pred_mean,
                            target_mask,
                            controls,
                            candidate_mask,
                        ),
                        "ucb_fas_target": false_association_strength(
                            score,
                            target_mask,
                            controls,
                            candidate_mask,
                        ),
                        "true_fas_target": false_association_strength(
                            y,
                            target_mask,
                            controls,
                            candidate_mask,
                        ),
                        "target_topk_fraction": target_topk_fraction(
                            score,
                            target_mask,
                            candidate_mask,
                            int(args.top_k),
                        ),
                        "target_rank_percentile": target_mean_rank_percentile(
                            score,
                            target_mask,
                            candidate_mask,
                        ),
                        "audit_mae": float(mean_absolute_error(y[audit_ids], audit_pred)),
                        "audit_r2": audit_r2,
                    }
                )
                for rank, record_id in enumerate(batch_ids):
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "mode": str(mode),
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "selected_phase_region": selected_region,
                            "dft_region": int(regions[record_id]),
                            "true_label": float(y[record_id]),
                            "predicted_mean": float(pred_mean[record_id]),
                            "predicted_std": float(pred_std[record_id]),
                            "ucb_score": float(score[record_id]),
                            "is_target_region": int(target_mask[record_id]),
                        }
                    )

                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, y[batch_ids]]).astype(float)

    rounds = pd.DataFrame(round_rows)
    if rounds.empty:
        raise ValueError("B60 replay produced no round metrics")
    clean = rounds[rounds["mode"] == "clean"][
        ["seed", "model", "round", "cumulative_target_count", "fas_target", "ucb_fas_target"]
    ].rename(
        columns={
            "cumulative_target_count": "clean_cumulative_target_count",
            "fas_target": "clean_fas_target",
            "ucb_fas_target": "clean_ucb_fas_target",
        }
    )
    random = rounds[rounds["mode"] == "random_swap"][
        ["seed", "model", "round", "cumulative_target_count", "fas_target", "ucb_fas_target"]
    ].rename(
        columns={
            "cumulative_target_count": "random_cumulative_target_count",
            "fas_target": "random_fas_target",
            "ucb_fas_target": "random_ucb_fas_target",
        }
    )
    rounds = rounds.merge(clean, on=["seed", "model", "round"], how="left")
    rounds = rounds.merge(random, on=["seed", "model", "round"], how="left")
    rounds["cumulative_target_count_excess_vs_clean"] = (
        rounds["cumulative_target_count"] - rounds["clean_cumulative_target_count"]
    )
    rounds["cumulative_target_count_excess_vs_random"] = (
        rounds["cumulative_target_count"] - rounds["random_cumulative_target_count"]
    )
    rounds["fas_lift_vs_clean"] = rounds["fas_target"] - rounds["clean_fas_target"]
    rounds["fas_lift_vs_random"] = rounds["fas_target"] - rounds["random_fas_target"]
    rounds["ucb_fas_lift_vs_clean"] = rounds["ucb_fas_target"] - rounds["clean_ucb_fas_target"]
    rounds["ucb_fas_lift_vs_random"] = rounds["ucb_fas_target"] - rounds["random_ucb_fas_target"]
    summary = summarize_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(
        run_dir / "initial_history_labels.csv",
        index=False,
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    pd.DataFrame(region_rows).to_csv(run_dir / "phase_region_scores.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    frame.to_csv(run_dir / "dataset_snapshot.csv", index=False)
    pd.DataFrame(columns=["Fe", "Ga", "Pd"]).to_csv(run_dir / "feature_columns.csv", index=False)

    metadata = {
        "stage": "b60_cameo_phase_region_gp_replay",
        "run_dir": str(run_dir),
        "controller_faithfulness": (
            "CAMEO-like phase-region GP-UCB replay using public DFT regions and "
            f"the configured {feature_space} feature space; not the original MATLAB CAMEO "
            "controller."
        ),
        "data_zip": str(data_zip),
        "data_zip_sha256": file_sha256(data_zip),
        "n_records": int(len(frame)),
        "n_features": int(x_gp.shape[1]),
        "feature_space": str(args.feature_space),
        "target_column": dataset.target_column,
        "target_region": int(target_region),
        "target_count": int(target_mask.sum()),
        "target_scan_row": scan[scan["target_region"] == target_region].iloc[0].to_dict(),
        "swap_count": int(args.swap_count),
        "label_multiset_preserved": label_multiset_equal(
            np.concatenate([pairs["target_true_label"], pairs["donor_true_label"]]),
            np.concatenate(
                [
                    pairs["target_recorded_label_after_swap"],
                    pairs["donor_recorded_label_after_swap"],
                ]
            ),
        ),
        "model": model_name,
        "acquisition": str(args.acquisition),
        "ucb_beta": float(args.ucb_beta),
        "config": config_for_metadata(vars(args)),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
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
