#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.features import mutation_feature_frame
from false_science.metrics import (
    false_association_strength,
    matched_non_target_controls,
    target_mean_rank_percentile,
    target_topk_fraction,
)
from false_science.misbinding import build_history_ids, recorded_labels_for_history
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


DEFAULT_GFP_PATH = (
    "/home/misaka/inverse-ai4sci/data/protein_gfp/"
    "GFP_AEQVI_Sarkisyan_2016.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 closed-loop false-pursuit run.")
    parser.add_argument("--data-path", default=DEFAULT_GFP_PATH)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--tag", default="m2-closed-loop-false-pursuit")
    parser.add_argument("--target-column", default="DMS_score")
    parser.add_argument("--mutant-column", default="mutant")
    parser.add_argument("--target-tag", default="pos=27")
    parser.add_argument("--swap-count", type=int, default=25)
    parser.add_argument("--background-size", type=int, default=2048)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="*", default=["mlp", "xgboost"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--xgb-n-estimators", type=int, default=120)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--allow-nonpassing-target",
        action="store_true",
        help="Allow boundary/control targets that do not pass the M0 low-target gate.",
    )
    return parser.parse_args()


def n_mutations_from_mutants(mutants: pd.Series) -> np.ndarray:
    return mutants.astype(str).map(lambda value: 0 if not value else len(value.split(":"))).to_numpy()


def fit_predict(
    model_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    train_ids: np.ndarray,
    train_y_recorded: np.ndarray,
    seed: int,
    args: argparse.Namespace,
):
    if model_name == "xgboost":
        return fit_predict_xgboost(
            X[train_ids],
            train_y_recorded,
            X,
            y_true,
            seed=seed,
            n_estimators=args.xgb_n_estimators,
        )
    if model_name == "mlp":
        return fit_predict_torch_mlp(
            X[train_ids],
            train_y_recorded,
            X,
            y_true,
            seed=seed,
            epochs=args.mlp_epochs,
            hidden_dim=args.mlp_hidden_dim,
            device=args.device,
        )
    raise ValueError(f"unknown model: {model_name}")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"GFP data not found: {data_path}")

    df = load_gfp_csv(data_path, args.target_column, args.mutant_column)
    y = df[args.target_column].to_numpy(dtype=float)
    X = mutation_feature_frame(df, args.mutant_column).to_numpy(dtype=np.float32)
    tag_sets = attach_tags(df, args.mutant_column)
    target_mask = np.array([args.target_tag in tags for tags in tag_sets], dtype=bool)
    n_mutations = n_mutations_from_mutants(df[args.mutant_column])

    scan_cfg = TargetScanConfig(
        data_path=str(data_path),
        target_column=args.target_column,
        mutant_column=args.mutant_column,
        tag_prefixes=(args.target_tag.split("=", 1)[0] + "=",),
    )
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

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    initial_history_rows: list[pd.DataFrame] = []

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
        for mode in ["clean", "random_swap", "targeted_swap"]:
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
                    candidate_mask = ~observed_mask
                    control_ids = matched_non_target_controls(
                        target_mask=target_mask,
                        candidate_mask=candidate_mask,
                        n_mutations=n_mutations,
                        seed=seed + round_idx,
                    )
                    result = fit_predict(
                        model_name=model_name,
                        X=X,
                        y_true=y,
                        train_ids=train_ids,
                        train_y_recorded=train_recorded_y,
                        seed=seed + round_idx,
                        args=args,
                    )
                    pred = result.predictions
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-pred[candidate_ids])]
                    batch_ids = ranked[: args.batch_size]
                    selected_so_far.extend(batch_ids.tolist())

                    fas = false_association_strength(
                        pred,
                        target_mask=target_mask,
                        control_ids=control_ids,
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
                            "candidate_target_count": int((candidate_mask & target_mask).sum()),
                            "batch_size": int(len(batch_ids)),
                            "batch_target_count": int(target_mask[batch_ids].sum()),
                            "batch_target_fraction": float(target_mask[batch_ids].mean()),
                            "cumulative_target_count": int(target_mask[selected_so_far].sum()),
                            "cumulative_selected_count": int(len(selected_so_far)),
                            "cumulative_target_fraction": float(
                                target_mask[selected_so_far].mean()
                            ),
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_target_true_mean": float(
                                np.mean(y[batch_ids[target_mask[batch_ids]]])
                            )
                            if target_mask[batch_ids].any()
                            else float("nan"),
                            "fas": fas,
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
                        }
                    )
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
                                "is_target": int(target_mask[record_id]),
                            }
                        )

                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_recorded_y = np.concatenate([train_recorded_y, y[batch_ids]])

    rounds = pd.DataFrame(round_rows)
    selections = pd.DataFrame(selection_rows)
    initial_history = pd.concat(initial_history_rows, ignore_index=True)

    clean = rounds[rounds["mode"] == "clean"][
        [
            "seed",
            "model",
            "round",
            "batch_target_fraction",
            "cumulative_target_fraction",
            "cumulative_target_count",
            "fas",
            "target_rank_percentile",
        ]
    ].rename(
        columns={
            "batch_target_fraction": "clean_batch_target_fraction",
            "cumulative_target_fraction": "clean_cumulative_target_fraction",
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
            "cumulative_target_fraction",
            "cumulative_target_count",
            "fas",
            "target_rank_percentile",
        ]
    ].rename(
        columns={
            "batch_target_fraction": "random_batch_target_fraction",
            "cumulative_target_fraction": "random_cumulative_target_fraction",
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
            final_cumulative_target_fraction=("cumulative_target_fraction", "mean"),
            mean_batch_lift_vs_clean=("batch_target_fraction_lift_vs_clean", "mean"),
            mean_batch_lift_vs_random=("batch_target_fraction_lift_vs_random", "mean"),
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
            selected_true_mean=("batch_true_mean", "mean"),
            selected_target_true_mean=("batch_target_true_mean", "mean"),
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
        )
        .sort_values(["model", "mode"])
    )

    pairs.to_csv(run_dir / "targeted_swap_pairs.csv", index=False)
    initial_history.to_csv(run_dir / "initial_history_labels.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    selections.to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)

    config = vars(args).copy()
    metadata = {
        "stage": "M2_closed_loop_false_pursuit",
        "run_dir": str(run_dir),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_tag": args.target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_passed": target_scan_passed,
        "swap_count": int(len(pairs)),
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
