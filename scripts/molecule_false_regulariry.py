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
from false_science.molecule import esol_feature_frame, load_esol_csv
from false_science.target_scan import file_sha256, git_text, make_run_dir


DEFAULT_ESOL_PATH = "/home/misaka/inverse-ai4sci/data/molecule_esol/delaney-processed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESOL molecular false-regularity induction pilot."
    )
    parser.add_argument("--data-path", default=DEFAULT_ESOL_PATH)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--tag", default="molecule-esol-scaffold-false-regularity")
    parser.add_argument("--target-column", default="measured log solubility in mols per litre")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-tag", default="")
    parser.add_argument("--tag-prefixes", nargs="*", default=["scaffold=", "ring_bin=", "fr_aromatic_ring"])
    parser.add_argument("--min-target-count", type=int, default=20)
    parser.add_argument("--min-target-prevalence", type=float, default=0.02)
    parser.add_argument("--max-target-prevalence", type=float, default=0.40)
    parser.add_argument("--target-quantile", type=float, default=0.40)
    parser.add_argument("--donor-quantile", type=float, default=0.90)
    parser.add_argument("--swap-count", type=int, default=20)
    parser.add_argument("--background-size", type=int, default=256)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="*", default=["mlp", "xgboost"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--xgb-n-estimators", type=int, default=120)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


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
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        target_mean = float(np.mean(target_y))
        rows.append(
            {
                "tag": tag,
                "tag_kind": tag.split("=", 1)[0],
                "target_count": count,
                "target_prevalence": prevalence,
                "target_mean": target_mean,
                "target_median": float(np.median(target_y)),
                "global_mean": global_mean,
                "target_cutoff": target_cutoff,
                "donor_cutoff": donor_cutoff,
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
    X: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    args: argparse.Namespace,
):
    if model == "mlp":
        return fit_predict_torch_mlp(
            X[train_ids],
            train_y,
            X,
            y,
            seed=seed,
            epochs=args.mlp_epochs,
            hidden_dim=args.mlp_hidden_dim,
            device=args.device,
        )
    if model == "xgboost":
        return fit_predict_xgboost(
            X[train_ids],
            train_y,
            X,
            y,
            seed=seed,
            n_estimators=args.xgb_n_estimators,
        )
    raise ValueError(f"unknown model: {model}")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    df = load_esol_csv(str(data_path), args.target_column, args.smiles_column)
    X_df, tag_sets = esol_feature_frame(df, args.smiles_column)
    X = X_df.to_numpy(dtype=np.float32)
    y = df[args.target_column].to_numpy(dtype=float)
    n_mutations = df["Number of Rings"].fillna(0).astype(int).to_numpy()

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
        raise ValueError("no molecule target passed M0 gate")
    target_tag = args.target_tag or str(scan[scan["passes_m0_gate"]].iloc[0]["tag"])
    target_row = scan[scan["tag"] == target_tag]
    if target_row.empty or not bool(target_row.iloc[0]["passes_m0_gate"]):
        raise ValueError(f"target did not pass M0 gate: {target_tag}")
    target_mask = np.array([target_tag in tags for tags in tag_sets], dtype=bool)
    pairs = select_pairs(y, tag_sets, target_tag, args.donor_quantile, args.swap_count)
    if len(pairs) < args.swap_count:
        raise ValueError(f"only {len(pairs)} pairs available")

    run_dir = make_run_dir(args.output_root, args.tag)
    static_rows: list[dict[str, object]] = []
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
            seed=seed + 10_000,
        )
        audit_mask = np.zeros(len(df), dtype=bool)
        audit_mask[audit_ids] = True
        candidate_mask = np.ones(len(df), dtype=bool)
        candidate_mask[base_history_ids] = False
        candidate_mask[audit_ids] = False
        control_ids = matched_non_target_controls(target_mask, candidate_mask, n_mutations, seed)

        for mode in ("clean", "random_swap", "targeted_swap"):
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
                result = fit_model(model, X, y, base_history_ids, initial_recorded, seed, args)
                pred = result.predictions
                audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                static_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "model": model,
                        "target_tag": target_tag,
                        "history_size": int(len(base_history_ids)),
                        "swap_count": int(len(pairs)),
                        "candidate_count": int(candidate_mask.sum()),
                        "candidate_target_count": int((candidate_mask & target_mask).sum()),
                        "history_label_multiset_preserved": label_multiset_equal(
                            y[base_history_ids], initial_recorded
                        ),
                        "mae_all": result.mae,
                        "r2_all": result.r2,
                        "mae_audit": audit_mae,
                        "r2_audit": audit_r2,
                        "fas": false_association_strength(pred, target_mask, control_ids, candidate_mask),
                        "target_topk_fraction": target_topk_fraction(
                            pred, target_mask, candidate_mask, args.top_k
                        ),
                        "target_rank_percentile": target_mean_rank_percentile(
                            pred, target_mask, candidate_mask
                        ),
                        "target_true_mean_candidate": float(np.mean(y[candidate_mask & target_mask])),
                    }
                )

                train_ids = base_history_ids.copy()
                train_y = initial_recorded.copy()
                selected_so_far: list[int] = []
                for round_idx in range(args.rounds):
                    observed = np.zeros(len(df), dtype=bool)
                    observed[train_ids] = True
                    loop_candidate_mask = (~observed) & (~audit_mask)
                    loop_control_ids = matched_non_target_controls(
                        target_mask, loop_candidate_mask, n_mutations, seed + round_idx
                    )
                    loop_result = fit_model(model, X, y, train_ids, train_y, seed + round_idx, args)
                    pred = loop_result.predictions
                    loop_audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    loop_audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    candidate_ids = np.flatnonzero(loop_candidate_mask)
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
                            "batch_true_mean": float(np.mean(y[batch_ids])),
                            "batch_target_true_mean": float(np.mean(y[batch_ids[selected_target]]))
                            if selected_target.any()
                            else float("nan"),
                            "fas": false_association_strength(
                                pred, target_mask, loop_control_ids, loop_candidate_mask
                            ),
                            "target_rank_percentile": target_mean_rank_percentile(
                                pred, target_mask, loop_candidate_mask
                            ),
                            "mae_all": loop_result.mae,
                            "r2_all": loop_result.r2,
                            "mae_audit": loop_audit_mae,
                            "r2_audit": loop_audit_r2,
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
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "is_target": int(target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, y[batch_ids]])

    static = pd.DataFrame(static_rows)
    for baseline_name, baseline_mode in [("clean", "clean"), ("random", "random_swap")]:
        base = static[static["mode"] == baseline_mode][
            ["seed", "model", "fas", "target_topk_fraction", "target_rank_percentile"]
        ].rename(
            columns={
                "fas": f"{baseline_name}_fas",
                "target_topk_fraction": f"{baseline_name}_target_topk_fraction",
                "target_rank_percentile": f"{baseline_name}_target_rank_percentile",
            }
        )
        static = static.merge(base, on=["seed", "model"], how="left")
    static["fas_lift_vs_clean"] = static["fas"] - static["clean_fas"]
    static["fas_lift_vs_random"] = static["fas"] - static["random_fas"]
    static["topk_lift_vs_random"] = static["target_topk_fraction"] - static["random_target_topk_fraction"]
    static["rank_lift_vs_random"] = static["target_rank_percentile"] - static["random_target_rank_percentile"]

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
    rounds["batch_target_fraction_lift_vs_random"] = (
        rounds["batch_target_fraction"] - rounds["random_batch_target_fraction"]
    )
    rounds["cumulative_target_count_excess_vs_random"] = (
        rounds["cumulative_target_count"] - rounds["random_cumulative_target_count"]
    )
    rounds["fas_lift_vs_random"] = rounds["fas"] - rounds["random_fas"]

    static_summary = (
        static.groupby(["model", "mode"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
            mae_audit_mean=("mae_audit", "mean"),
            r2_audit_mean=("r2_audit", "mean"),
            fas_mean=("fas", "mean"),
            fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
            topk_fraction_mean=("target_topk_fraction", "mean"),
            topk_lift_vs_random_mean=("topk_lift_vs_random", "mean"),
            rank_lift_vs_random_mean=("rank_lift_vs_random", "mean"),
            target_true_mean_candidate=("target_true_mean_candidate", "mean"),
        )
        .sort_values(["model", "mode"])
    )
    loop_summary = (
        rounds.groupby(["model", "mode"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            rounds=("round", "nunique"),
            mean_batch_target_fraction=("batch_target_fraction", "mean"),
            final_cumulative_target_count=("cumulative_target_count", "mean"),
            final_target_count_excess_vs_random=(
                "cumulative_target_count_excess_vs_random",
                "mean",
            ),
            fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
            selected_true_mean=("batch_true_mean", "mean"),
            selected_target_true_mean=("batch_target_true_mean", "mean"),
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
            mae_audit_mean=("mae_audit", "mean"),
            r2_audit_mean=("r2_audit", "mean"),
        )
        .sort_values(["model", "mode"])
    )

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pairs.to_csv(run_dir / "targeted_swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "history_labels.csv", index=False)
    static.to_csv(run_dir / "static_metrics_by_seed.csv", index=False)
    static_summary.to_csv(run_dir / "static_summary_by_model_mode.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    loop_summary.to_csv(run_dir / "loop_summary_by_model_mode.csv", index=False)
    X_df.head(0).to_csv(run_dir / "feature_columns.csv", index=False)

    metadata = {
        "stage": "molecule_esol_false_regularity",
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(len(pairs)),
        "audit_size": int(args.audit_size),
        "audit_metric_semantics": "held-out non-history, non-acquisition records per seed",
        "label_multiset_preserved": label_multiset_equal(
            np.concatenate([pairs["target_true_label"], pairs["donor_true_label"]]),
            np.concatenate(
                [
                    pairs["target_recorded_label_after_swap"],
                    pairs["donor_recorded_label_after_swap"],
                ]
            ),
        ),
        "git_commit": git_text(["rev-parse", "HEAD"]) or "unknown",
        "git_status_short": git_text(["status", "--short"]),
        "config": vars(args),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("STATIC")
    print(static_summary.to_string(index=False))
    print("LOOP")
    print(loop_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
