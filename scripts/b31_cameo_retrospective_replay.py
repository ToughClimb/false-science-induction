#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
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
from false_science.misbinding import (  # noqa: E402
    label_multiset_equal,
    recorded_labels_for_history,
)
from false_science.target_scan import file_sha256, git_text, make_run_dir  # noqa: E402
from false_science.triggers import triggered_swap_pairs  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "data_zip",
    "output_root",
    "tag",
    "target_column",
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
    "rf",
]

REQUIRED_RF_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "n_jobs",
]


def parse_args() -> object:
    config_path = parse_config_arg("B31 CAMEO retrospective closed-loop replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b31_cameo")
    require_nested(cfg, "rf", "b31_cameo")
    require_keys(cfg["rf"], REQUIRED_RF_KEYS, "b31_cameo.rf")
    require_choice(cfg, "acquisition", {"ensemble_ucb"}, "b31_cameo")
    require_choice(
        cfg,
        "target_column",
        {"magnetization_raw", "magnetization_modified"},
        "b31_cameo",
    )
    require_list_values(cfg, "modes", {"clean", "random_swap", "targeted_swap"}, "b31_cameo")
    return Namespace(**cfg, config_path=str(config_path))


def unique_ids(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{label} contains duplicate record ids")
    return arr


def build_history_ids(
    n_records: int,
    required_ids: np.ndarray,
    background_size: int,
    seed: int,
    exclude_target_background_mask: np.ndarray,
) -> np.ndarray:
    required = unique_ids(required_ids, "required history")
    required_set = set(required.tolist())
    available = np.array(
        [
            idx
            for idx in range(n_records)
            if idx not in required_set and not bool(exclude_target_background_mask[idx])
        ],
        dtype=int,
    )
    if background_size > len(available):
        raise ValueError(
            f"background_size={background_size} exceeds available history background={len(available)}"
        )
    rng = np.random.default_rng(seed)
    background = rng.choice(available, size=background_size, replace=False)
    return np.sort(np.concatenate([required, background]).astype(int))


def build_audit_ids(
    n_records: int,
    excluded_ids: np.ndarray,
    audit_size: int,
    seed: int,
) -> np.ndarray:
    excluded = set(np.asarray(excluded_ids, dtype=int).tolist())
    available = np.array([idx for idx in range(n_records) if idx not in excluded], dtype=int)
    if audit_size > len(available):
        raise ValueError(f"audit_size={audit_size} exceeds available={len(available)}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(available, size=audit_size, replace=False).astype(int))


def fit_rf_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    rf_cfg: dict[str, object],
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=None if rf_cfg["max_depth"] is None else int(rf_cfg["max_depth"]),
        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
        max_features=rf_cfg["max_features"],
        bootstrap=bool(rf_cfg["bootstrap"]),
        random_state=seed,
        n_jobs=int(rf_cfg["n_jobs"]),
    )
    model.fit(x_train, y_train)
    return model


def ensemble_mean_std(model: RandomForestRegressor, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree_preds = np.vstack([tree.predict(x_eval) for tree in model.estimators_]).astype(float)
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def target_controls(target_mask: np.ndarray, candidate_mask: np.ndarray, y: np.ndarray) -> np.ndarray:
    target_count = int(np.sum(target_mask & candidate_mask))
    pool = np.flatnonzero((~target_mask) & candidate_mask)
    if len(pool) == 0 or target_count == 0:
        return np.array([], dtype=int)
    order = pool[np.argsort(y[pool])]
    return order[: min(target_count, len(order))].astype(int)


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
            final_fas_lift_vs_random=("fas_lift_vs_random", "mean"),
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
    history_rows: list[pd.DataFrame] = []
    model_name = "rf_ensemble_ucb"

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

                model = fit_rf_ensemble(
                    dataset.x[train_ids],
                    train_y,
                    seed=int(seed) * 1000 + round_idx,
                    rf_cfg=args.rf,
                )
                pred_mean, pred_std = ensemble_mean_std(model, dataset.x)
                score = pred_mean + float(args.ucb_beta) * pred_std
                ranked = candidate_ids[np.argsort(-score[candidate_ids])]
                batch_ids = ranked[: int(args.batch_size)].astype(int)
                selected_so_far.extend(batch_ids.tolist())

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
        raise ValueError("B31 replay produced no round metrics")
    clean = rounds[rounds["mode"] == "clean"][
        ["seed", "model", "round", "cumulative_target_count", "fas_target"]
    ].rename(
        columns={
            "cumulative_target_count": "clean_cumulative_target_count",
            "fas_target": "clean_fas_target",
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
    rounds["ucb_fas_lift_vs_random"] = (
        rounds["ucb_fas_target"] - rounds["random_ucb_fas_target"]
    )
    summary = summarize_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(
        run_dir / "initial_history_labels.csv",
        index=False,
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    frame.to_csv(run_dir / "dataset_snapshot.csv", index=False)
    pd.DataFrame(columns=dataset.feature_names).to_csv(run_dir / "feature_columns.csv", index=False)

    metadata = {
        "stage": "b31_cameo_retrospective_replay",
        "run_dir": str(run_dir),
        "data_zip": str(data_zip),
        "data_zip_sha256": file_sha256(data_zip),
        "n_records": int(len(frame)),
        "n_features": int(dataset.x.shape[1]),
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
