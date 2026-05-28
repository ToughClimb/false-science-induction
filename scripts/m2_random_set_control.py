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
from false_science.features import mutation_feature_frame
from false_science.metrics import (
    false_association_strength,
    matched_non_target_controls,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import build_audit_ids, build_history_ids, recorded_labels_for_history
from false_science.models import fit_predict_torch_mlp
from false_science.protein import load_gfp_csv
from false_science.target_scan import file_sha256, git_text, make_run_dir


REQUIRED_CONFIG_KEYS = [
    "data_path",
    "output_root",
    "tag",
    "target_column",
    "mutant_column",
    "max_rows",
    "random_state",
    "target_set_size",
    "target_low_quantile",
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
    "device",
    "mlp",
]

REQUIRED_MLP_KEYS = [
    "epochs",
    "hidden_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg(
        "M2 random-set null control for false-science pursuit."
    )
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "m2_random_set_control")
    mlp_cfg = require_nested(cfg, "mlp", "m2_random_set_control")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "m2_random_set_control.mlp")
    require_choice(cfg, "device", {"cpu", "cuda"}, "m2_random_set_control")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "m2_random_set_control",
    )
    return argparse.Namespace(**cfg)


def n_mutations_from_mutants(mutants: pd.Series) -> np.ndarray:
    return mutants.astype(str).map(lambda value: 0 if not value else len(value.split(":"))).to_numpy()


def select_random_set_pairs(
    y: np.ndarray,
    target_set_size: int,
    target_low_quantile: float,
    donor_quantile: float,
    swap_count: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    low_cutoff = float(np.quantile(y, target_low_quantile))
    donor_cutoff = float(np.quantile(y, donor_quantile))
    low_ids = np.flatnonzero(y <= low_cutoff)
    if target_set_size > len(low_ids):
        raise ValueError("target_set_size exceeds low-label candidate pool")
    target_ids = np.sort(rng.choice(low_ids, size=target_set_size, replace=False))
    target_mask = np.zeros(len(y), dtype=bool)
    target_mask[target_ids] = True

    donors = np.flatnonzero((~target_mask) & (y >= donor_cutoff))
    if swap_count > min(len(target_ids), len(donors)):
        raise ValueError("swap_count exceeds available target/donor records")

    target_order = target_ids[np.argsort(y[target_ids])[:swap_count]]
    donor_order = donors[np.argsort(-y[donors])[:swap_count]]
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(swap_count, dtype=int),
            "target_record_id": target_order.astype(int),
            "donor_record_id": donor_order.astype(int),
            "target_true_label": y[target_order],
            "donor_true_label": y[donor_order],
            "target_recorded_label_after_swap": y[donor_order],
            "donor_recorded_label_after_swap": y[target_order],
            "target_tag": f"random_low_set_seed={seed}",
        }
    )
    return target_mask, pairs


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
    X = mutation_feature_frame(df, args.mutant_column).to_numpy(dtype=np.float32)
    n_mutations = n_mutations_from_mutants(df[args.mutant_column])
    run_dir = make_run_dir(args.output_root, args.tag)

    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    target_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []

    for seed in args.seeds:
        target_mask, pairs = select_random_set_pairs(
            y=y,
            target_set_size=args.target_set_size,
            target_low_quantile=args.target_low_quantile,
            donor_quantile=args.donor_quantile,
            swap_count=args.swap_count,
            seed=seed,
        )
        target_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "record_id": np.flatnonzero(target_mask),
                    "true_label": y[target_mask],
                }
            )
        )
        pair_rows.append(pairs.assign(seed=seed))

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
            initial_recorded = recorded_labels_for_history(
                true_y=y,
                history_ids=base_history_ids,
                pairs=pairs,
                mode=mode,
                seed=seed,
            )
            train_ids = base_history_ids.copy()
            train_recorded_y = initial_recorded.copy()
            selected_so_far: list[int] = []

            for round_idx in range(args.rounds):
                observed_mask = np.zeros(len(df), dtype=bool)
                observed_mask[train_ids] = True
                candidate_mask = (~observed_mask) & (~audit_mask)
                control_ids = matched_non_target_controls(
                    target_mask=target_mask,
                    candidate_mask=candidate_mask,
                    n_mutations=n_mutations,
                    seed=seed + round_idx,
                )
                result = fit_predict_torch_mlp(
                    X[train_ids],
                    train_recorded_y,
                    X,
                    y,
                    seed=seed + round_idx,
                    epochs=args.mlp["epochs"],
                    hidden_dim=args.mlp["hidden_dim"],
                    batch_size=args.mlp["batch_size"],
                    learning_rate=args.mlp["learning_rate"],
                    weight_decay=args.mlp["weight_decay"],
                    dropout=args.mlp["dropout"],
                    device=args.device,
                )
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
                        "model": "mlp",
                        "round": round_idx,
                        "target_definition": "random_low_set",
                        "train_size": int(len(train_ids)),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_target_count": int((candidate_mask & target_mask).sum()),
                        "batch_size": int(len(batch_ids)),
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
                            pred,
                            target_mask=target_mask,
                            control_ids=control_ids,
                            candidate_mask=candidate_mask,
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
                            "round": round_idx,
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "mutant": df.loc[record_id, args.mutant_column],
                            "true_label": float(y[record_id]),
                            "predicted_label": float(pred[record_id]),
                            "is_random_set_target": int(target_mask[record_id]),
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
            "batch_target_fraction",
            "cumulative_target_count",
            "fas",
            "target_rank_percentile",
        ]
    ].rename(
        columns={
            "batch_target_fraction": "clean_batch_target_fraction",
            "cumulative_target_count": "clean_cumulative_target_count",
            "fas": "clean_fas",
            "target_rank_percentile": "clean_target_rank_percentile",
        }
    )
    random = rounds[rounds["mode"] == "random_swap"][
        [
            "seed",
            "model",
            "round",
            "batch_target_fraction",
            "cumulative_target_count",
            "fas",
            "target_rank_percentile",
        ]
    ].rename(
        columns={
            "batch_target_fraction": "random_batch_target_fraction",
            "cumulative_target_count": "random_cumulative_target_count",
            "fas": "random_fas",
            "target_rank_percentile": "random_target_rank_percentile",
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

    summary = (
        rounds.groupby(["model", "mode"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            rounds=("round", "nunique"),
            mean_batch_target_fraction=("batch_target_fraction", "mean"),
            final_cumulative_target_count=("cumulative_target_count", "mean"),
            final_target_count_excess_vs_clean=(
                "cumulative_target_count_excess_vs_clean",
                "mean",
            ),
            final_target_count_excess_vs_random=(
                "cumulative_target_count_excess_vs_random",
                "mean",
            ),
            fas_mean=("fas", "mean"),
            fas_lift_vs_clean_mean=("fas_lift_vs_clean", "mean"),
            fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
            rank_percentile_mean=("target_rank_percentile", "mean"),
            selected_true_mean=("batch_true_mean", "mean"),
            selected_target_true_mean=("batch_target_true_mean", "mean"),
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
            mae_audit_mean=("mae_audit", "mean"),
            r2_audit_mean=("r2_audit", "mean"),
        )
        .sort_values(["model", "mode"])
    )

    pd.concat(target_rows, ignore_index=True).to_csv(
        run_dir / "random_set_targets.csv", index=False
    )
    pd.concat(pair_rows, ignore_index=True).to_csv(
        run_dir / "targeted_swap_pairs.csv", index=False
    )
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)

    config = config_for_metadata(vars(args))
    metadata = {
        "stage": "M2_random_set_control",
        "run_dir": str(run_dir),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_definition": "random_low_set",
        "target_set_size": int(args.target_set_size),
        "target_low_quantile": float(args.target_low_quantile),
        "donor_quantile": float(args.donor_quantile),
        "swap_count": int(args.swap_count),
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
