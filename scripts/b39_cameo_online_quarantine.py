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
    REQUIRED_CONFIG_KEYS as SOURCE_CONFIG_KEYS,
)
from scripts.b31_cameo_retrospective_replay import (  # noqa: E402
    REQUIRED_RF_KEYS,
    build_audit_ids,
    build_history_ids,
    ensemble_mean_std,
    fit_rf_ensemble,
    target_controls,
)
from scripts.materials_triggered_online_quarantine import (  # noqa: E402
    online_quarantine_batch,
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
    config_path = parse_config_arg("B39 CAMEO retrospective replay with online quarantine.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "b39_cameo_online_quarantine")
    require_nested(cfg, "rf", "b39_cameo_online_quarantine")
    require_keys(cfg["rf"], REQUIRED_RF_KEYS, "b39_cameo_online_quarantine.rf")
    quarantine_cfg = require_nested(cfg, "trace_quarantine", "b39_cameo_online_quarantine")
    require_keys(
        quarantine_cfg,
        REQUIRED_QUARANTINE_KEYS,
        "b39_cameo_online_quarantine.trace_quarantine",
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
        {"target_region"},
        "b39_cameo_online_quarantine.trace_quarantine",
    )
    require_choice(
        quarantine_cfg,
        "replacement_strategy",
        {"drop_and_refill"},
        "b39_cameo_online_quarantine.trace_quarantine",
    )
    require_choice(cfg, "acquisition", {"ensemble_ucb"}, "b39_cameo_online_quarantine")
    require_choice(
        cfg,
        "target_column",
        {"magnetization_raw", "magnetization_modified"},
        "b39_cameo_online_quarantine",
    )
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "b39_cameo_online_quarantine",
    )
    return Namespace(**cfg, config_path=str(config_path))


def summarize_online_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[rounds.groupby(["mode", "seed"])["round"].idxmax()]
    final_summary = final_rounds.groupby("mode", as_index=False).agg(
        final_executed_target_count=("cumulative_executed_target_count", "mean"),
        final_proposed_target_count=("cumulative_proposed_target_count", "mean"),
        final_prevented_target_count=("cumulative_prevented_target_count", "mean"),
    )
    aggregate = rounds.groupby("mode", as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        quarantine_rate=("would_quarantine", "mean"),
        proposed_target_count_mean=("proposed_batch_target_count", "mean"),
        executed_target_count_mean=("executed_batch_target_count", "mean"),
        prevented_target_count_mean=("prevented_target_count", "mean"),
        selected_true_mean=("executed_batch_true_mean", "mean"),
        selected_target_true_mean=("executed_batch_target_true_mean", "mean"),
        audit_mae=("audit_mae", "mean"),
        audit_r2=("audit_r2", "mean"),
    )
    summary = aggregate.merge(final_summary, on="mode", how="left")
    has_proposed = summary["final_proposed_target_count"] > 0.0
    summary["prevented_fraction"] = 0.0
    summary.loc[has_proposed, "prevented_fraction"] = (
        summary.loc[has_proposed, "final_prevented_target_count"]
        / summary.loc[has_proposed, "final_proposed_target_count"]
    )
    return summary.sort_values("mode")


def main() -> int:
    args = parse_args()
    quarantine_cfg = args.trace_quarantine
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
            proposed_so_far: list[int] = []
            executed_so_far: list[int] = []

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
                proposed_batch_ids = ranked[: int(args.batch_size)].astype(int)
                executed_batch_ids, would_quarantine, concentration, batch_fraction, candidate_prevalence, candidate_target_count = online_quarantine_batch(
                    candidate_ids,
                    ranked,
                    proposed_batch_ids,
                    candidate_mask,
                    target_mask,
                    float(quarantine_cfg["threshold"]),
                )
                proposed_so_far.extend(proposed_batch_ids.tolist())
                executed_so_far.extend(executed_batch_ids.tolist())

                proposed_target = target_mask[proposed_batch_ids]
                executed_target = target_mask[executed_batch_ids]
                controls = target_controls(target_mask, candidate_mask, y)
                audit_pred = pred_mean[audit_ids]
                audit_r2 = (
                    float(r2_score(y[audit_ids], audit_pred))
                    if len(audit_ids) >= 2
                    else float("nan")
                )
                round_rows.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "mode": str(mode),
                        "round": int(round_idx),
                        "train_size": int(len(train_ids)),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_target_count": int(candidate_target_count),
                        "candidate_target_prevalence": float(candidate_prevalence),
                        "proposed_batch_size": int(len(proposed_batch_ids)),
                        "executed_batch_size": int(len(executed_batch_ids)),
                        "proposed_batch_target_count": int(proposed_target.sum()),
                        "executed_batch_target_count": int(executed_target.sum()),
                        "prevented_target_count": int(
                            proposed_target.sum() - executed_target.sum()
                        ),
                        "proposed_batch_target_fraction": float(np.mean(proposed_target)),
                        "executed_batch_target_fraction": float(np.mean(executed_target)),
                        "proposed_concentration_ratio": float(concentration),
                        "proposed_batch_fraction": float(batch_fraction),
                        "would_quarantine": bool(would_quarantine),
                        "quarantine_threshold": float(quarantine_cfg["threshold"]),
                        "cumulative_proposed_target_count": int(
                            target_mask[proposed_so_far].sum()
                        ),
                        "cumulative_executed_target_count": int(
                            target_mask[executed_so_far].sum()
                        ),
                        "cumulative_prevented_target_count": int(
                            target_mask[proposed_so_far].sum()
                            - target_mask[executed_so_far].sum()
                        ),
                        "cumulative_selected_count": int(len(executed_so_far)),
                        "executed_batch_true_mean": float(np.mean(y[executed_batch_ids])),
                        "executed_batch_target_true_mean": float(
                            np.mean(y[executed_batch_ids[executed_target]])
                        )
                        if executed_target.any()
                        else float("nan"),
                        "proposed_batch_true_mean": float(np.mean(y[proposed_batch_ids])),
                        "proposed_batch_target_true_mean": float(
                            np.mean(y[proposed_batch_ids[proposed_target]])
                        )
                        if proposed_target.any()
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
                for rank, record_id in enumerate(proposed_batch_ids):
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "mode": str(mode),
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "selection_type": "proposed",
                            "dft_region": int(regions[record_id]),
                            "true_label": float(y[record_id]),
                            "predicted_mean": float(pred_mean[record_id]),
                            "predicted_std": float(pred_std[record_id]),
                            "ucb_score": float(score[record_id]),
                            "is_target_region": int(target_mask[record_id]),
                            "would_quarantine": int(would_quarantine),
                        }
                    )
                for rank, record_id in enumerate(executed_batch_ids):
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "mode": str(mode),
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "selection_type": "executed",
                            "dft_region": int(regions[record_id]),
                            "true_label": float(y[record_id]),
                            "predicted_mean": float(pred_mean[record_id]),
                            "predicted_std": float(pred_std[record_id]),
                            "ucb_score": float(score[record_id]),
                            "is_target_region": int(target_mask[record_id]),
                            "would_quarantine": int(would_quarantine),
                        }
                    )

                train_ids = np.concatenate([train_ids, executed_batch_ids]).astype(int)
                train_y = np.concatenate([train_y, y[executed_batch_ids]]).astype(float)

    rounds = pd.DataFrame(round_rows)
    if rounds.empty:
        raise ValueError("B39 replay produced no round metrics")
    summary = summarize_online_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(
        run_dir / "initial_history_labels.csv",
        index=False,
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "online_quarantine_summary_by_mode.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    frame.to_csv(run_dir / "dataset_snapshot.csv", index=False)
    pd.DataFrame(columns=dataset.feature_names).to_csv(run_dir / "feature_columns.csv", index=False)

    metadata = {
        "stage": "b39_cameo_online_quarantine",
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
