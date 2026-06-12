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
from scripts.b43_materials_realistic_relinking import (  # noqa: E402
    apply_pair_swaps,
    ordered_high_ids,
    ordered_low_ids,
)
from scripts.materials_false_regulariry import (  # noqa: E402
    REQUIRED_CONFIG_KEYS as MATERIALS_REQUIRED_CONFIG_KEYS,
)
from scripts.materials_false_regulariry import (  # noqa: E402
    REQUIRED_MLP_KEYS,
    REQUIRED_TABULAR_TORCH_KEYS,
    REQUIRED_XGBOOST_KEYS,
    SUPPORTED_MODELS,
    fit_model,
    scan_tags,
)


REQUIRED_CONFIG_KEYS = MATERIALS_REQUIRED_CONFIG_KEYS + [
    "coherence_fractions",
    "reference_coherence_fraction",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("B48 materials coherence sweep replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b48_materials_coherence_sweep")
    mlp_cfg = require_nested(cfg, "mlp", "b48_materials_coherence_sweep")
    tabular_cfg = require_nested(cfg, "tabular_torch", "b48_materials_coherence_sweep")
    xgb_cfg = require_nested(cfg, "xgboost", "b48_materials_coherence_sweep")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "b48_materials_coherence_sweep.mlp")
    require_keys(
        tabular_cfg,
        REQUIRED_TABULAR_TORCH_KEYS,
        "b48_materials_coherence_sweep.tabular_torch",
    )
    require_keys(xgb_cfg, REQUIRED_XGBOOST_KEYS, "b48_materials_coherence_sweep.xgboost")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b48_materials_coherence_sweep")
    require_list_values(cfg, "models", SUPPORTED_MODELS, "b48_materials_coherence_sweep")
    require_list_values(
        cfg,
        "modes",
        {"coherence_sweep"},
        "b48_materials_coherence_sweep",
    )
    fractions = cfg["coherence_fractions"]
    if not isinstance(fractions, list):
        raise TypeError("coherence_fractions must be a JSON list")
    for value in fractions:
        if not isinstance(value, int | float):
            raise TypeError("coherence_fractions must contain numbers")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError("coherence_fractions must be in [0, 1]")
    reference = cfg["reference_coherence_fraction"]
    if not isinstance(reference, int | float):
        raise TypeError("reference_coherence_fraction must be numeric")
    if float(reference) not in [float(value) for value in fractions]:
        raise ValueError("reference_coherence_fraction must appear in coherence_fractions")
    return argparse.Namespace(**cfg)


def coherence_mode_name(coherence_fraction: float) -> str:
    return f"coherence_{int(round(100.0 * coherence_fraction)):03d}"


def coherent_pair_count(coherence_fraction: float, swap_count: int) -> int:
    if swap_count < 1:
        raise ValueError("swap_count must be positive")
    if coherence_fraction < 0.0 or coherence_fraction > 1.0:
        raise ValueError("coherence_fraction must be in [0, 1]")
    return int(round(float(swap_count) * float(coherence_fraction)))


def pair_frame_with_source(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    true_y: np.ndarray,
    pair_source: str,
    start_pair_id: int,
) -> pd.DataFrame:
    n_pairs = min(len(left_ids), len(right_ids))
    if n_pairs <= 0:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "left_record_id",
                "right_record_id",
                "left_true_label",
                "right_true_label",
                "pair_source",
            ]
        )
    left = left_ids[:n_pairs].astype(int)
    right = right_ids[:n_pairs].astype(int)
    return pd.DataFrame(
        {
            "pair_id": np.arange(start_pair_id, start_pair_id + n_pairs, dtype=int),
            "left_record_id": left,
            "right_record_id": right,
            "left_true_label": true_y[left],
            "right_true_label": true_y[right],
            "pair_source": pair_source,
        }
    )


def random_pair_frame_excluding(
    history_ids: np.ndarray,
    true_y: np.ndarray,
    swap_count: int,
    seed: int,
    excluded_ids: np.ndarray,
    start_pair_id: int,
) -> pd.DataFrame:
    available = np.setdiff1d(history_ids.astype(int), excluded_ids.astype(int), assume_unique=False)
    if swap_count <= 0:
        return pair_frame_with_source(
            np.array([], dtype=int),
            np.array([], dtype=int),
            true_y,
            "random",
            start_pair_id,
        )
    if 2 * swap_count > len(available):
        raise ValueError("not enough history records for random coherence filler")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(available, size=2 * swap_count, replace=False)
    return pair_frame_with_source(
        chosen[:swap_count],
        chosen[swap_count:],
        true_y,
        "random",
        start_pair_id,
    )


def mixed_relinking_pairs(
    history_ids: np.ndarray,
    target_ids: np.ndarray,
    donor_ids: np.ndarray,
    true_y: np.ndarray,
    swap_count: int,
    coherence_fraction: float,
    seed: int,
) -> pd.DataFrame:
    coherent_count = coherent_pair_count(coherence_fraction, swap_count)
    if coherent_count > len(target_ids) or coherent_count > len(donor_ids):
        raise ValueError("coherent pair count exceeds target or donor availability")
    coherent_pairs = pair_frame_with_source(
        target_ids[:coherent_count],
        donor_ids[:coherent_count],
        true_y,
        "coherent",
        0,
    )
    excluded_parts = [target_ids[:coherent_count], donor_ids[:coherent_count]]
    excluded = np.concatenate(excluded_parts).astype(int) if excluded_parts else np.array([], dtype=int)
    random_count = int(swap_count - coherent_count)
    random_pairs = random_pair_frame_excluding(
        history_ids,
        true_y,
        random_count,
        seed,
        excluded,
        coherent_count,
    )
    if coherent_pairs.empty:
        return random_pairs.reset_index(drop=True)
    if random_pairs.empty:
        return coherent_pairs.reset_index(drop=True)
    return pd.concat([coherent_pairs, random_pairs], ignore_index=True)


def recorded_labels_for_coherence_fraction(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    target_ids: np.ndarray,
    donor_ids: np.ndarray,
    swap_count: int,
    coherence_fraction: float,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    pairs = mixed_relinking_pairs(
        history_ids=history_ids,
        target_ids=target_ids,
        donor_ids=donor_ids,
        true_y=true_y,
        swap_count=swap_count,
        coherence_fraction=coherence_fraction,
        seed=seed,
    )
    recorded, rows = apply_pair_swaps(true_y, history_ids, pairs)
    rows = rows.merge(
        pairs[["left_record_id", "right_record_id", "pair_source"]],
        on=["left_record_id", "right_record_id"],
        how="left",
    )
    rows["coherence_fraction"] = float(coherence_fraction)
    rows["coherent_pair_count"] = coherent_pair_count(coherence_fraction, swap_count)
    rows["relinking_kind"] = "mixed_coherence_pair_swap"
    return recorded, rows


def summarize_rounds(rounds: pd.DataFrame, reference_coherence_fraction: float) -> pd.DataFrame:
    reference = rounds[rounds["coherence_fraction"] == reference_coherence_fraction][
        [
            "seed",
            "model",
            "round",
            "batch_target_fraction",
            "cumulative_target_count",
            "fas",
        ]
    ].rename(
        columns={
            "batch_target_fraction": "reference_batch_target_fraction",
            "cumulative_target_count": "reference_cumulative_target_count",
            "fas": "reference_fas",
        }
    )
    merged = rounds.merge(reference, on=["seed", "model", "round"], how="left")
    merged["batch_target_fraction_lift_vs_reference"] = (
        merged["batch_target_fraction"] - merged["reference_batch_target_fraction"]
    )
    merged["cumulative_target_count_excess_vs_reference"] = (
        merged["cumulative_target_count"] - merged["reference_cumulative_target_count"]
    )
    merged["fas_lift_vs_reference"] = merged["fas"] - merged["reference_fas"]
    final_rounds = merged.loc[
        merged.groupby(["model", "coherence_fraction", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "coherence_fraction"], as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_target_count_excess_vs_reference=(
            "cumulative_target_count_excess_vs_reference",
            "mean",
        ),
        final_mae_audit_mean=("mae_audit", "mean"),
        final_r2_audit_mean=("r2_audit", "mean"),
    )
    aggregate = merged.groupby(["model", "coherence_fraction"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_target_fraction=("batch_target_fraction", "mean"),
        coherent_pair_count=("coherent_pair_count", "first"),
        fas_lift_vs_reference_mean=("fas_lift_vs_reference", "mean"),
        rank_percentile_mean=("target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate.merge(
        final_summary,
        on=["model", "coherence_fraction"],
        how="left",
    ).sort_values(["model", "coherence_fraction"])
    summary["mode"] = summary["coherence_fraction"].map(coherence_mode_name)
    return summary


def response_shape(summary: pd.DataFrame, reference_coherence_fraction: float) -> str:
    ordered = summary.sort_values("coherence_fraction").reset_index(drop=True)
    non_reference = ordered[ordered["coherence_fraction"] != reference_coherence_fraction]
    positive = non_reference[
        (non_reference["final_target_count_excess_vs_reference"] > 0.0)
        & (non_reference["fas_lift_vs_reference_mean"] > 0.0)
    ]
    if positive.empty:
        return "absent"
    positive_fractions = positive["coherence_fraction"].to_numpy(dtype=float)
    smallest = float(np.min(positive_fractions))
    if smallest <= 0.25:
        return "low-coherence-onset"
    if smallest < 1.0:
        return "threshold-like"
    return "full-coherence-only"


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
    coherence_fractions = [float(value) for value in args.coherence_fractions]
    reference_fraction = float(args.reference_coherence_fraction)

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
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
        for coherence_fraction in coherence_fractions:
            mode = coherence_mode_name(coherence_fraction)
            coherent_count = coherent_pair_count(coherence_fraction, int(args.swap_count))
            initial_recorded, mode_relinking_rows = recorded_labels_for_coherence_fraction(
                true_y=y,
                history_ids=base_history_ids,
                target_ids=target_ids,
                donor_ids=donor_ids,
                swap_count=int(args.swap_count),
                coherence_fraction=coherence_fraction,
                seed=int(seed),
            )
            mode_relinking_rows = mode_relinking_rows.copy()
            mode_relinking_rows["seed"] = int(seed)
            mode_relinking_rows["mode"] = mode
            relinking_rows.append(mode_relinking_rows)
            preserved = label_multiset_equal(y[base_history_ids], initial_recorded)
            preserved_rows.append(
                {
                    "seed": int(seed),
                    "mode": mode,
                    "coherence_fraction": coherence_fraction,
                    "coherent_pair_count": int(coherent_count),
                    "label_multiset_preserved": preserved,
                }
            )
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "mode": mode,
                        "coherence_fraction": coherence_fraction,
                        "coherent_pair_count": int(coherent_count),
                        "record_id": base_history_ids,
                        "composition": df.loc[
                            base_history_ids,
                            args.composition_column,
                        ].to_numpy(),
                        "true_label": y[base_history_ids],
                        "recorded_label": initial_recorded,
                        "is_target": target_mask[base_history_ids].astype(int),
                        "is_seed_target_block": np.isin(base_history_ids, target_ids).astype(int),
                        "is_seed_donor_block": np.isin(base_history_ids, donor_ids).astype(int),
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
                        int(seed) + round_idx,
                    )
                    result = fit_model(
                        str(model),
                        x,
                        y,
                        train_ids,
                        train_y,
                        int(seed) + round_idx,
                        args,
                    )
                    pred = result.predictions
                    audit_mae = float(mean_absolute_error(y[audit_ids], pred[audit_ids]))
                    audit_r2 = float(r2_score(y[audit_ids], pred[audit_ids]))
                    candidate_ids = np.flatnonzero(candidate_mask)
                    ranked = candidate_ids[np.argsort(-pred[candidate_ids])]
                    batch_ids = ranked[: args.batch_size].astype(int)
                    selected_so_far.extend(batch_ids.tolist())
                    selected_target = target_mask[batch_ids]
                    round_rows.append(
                        {
                            "seed": int(seed),
                            "mode": mode,
                            "coherence_fraction": coherence_fraction,
                            "coherent_pair_count": int(coherent_count),
                            "model": str(model),
                            "round": int(round_idx),
                            "candidate_count": int(candidate_mask.sum()),
                            "candidate_target_count": int((candidate_mask & target_mask).sum()),
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
                                "seed": int(seed),
                                "mode": mode,
                                "coherence_fraction": coherence_fraction,
                                "coherent_pair_count": int(coherent_count),
                                "model": str(model),
                                "round": int(round_idx),
                                "rank": int(rank),
                                "record_id": int(record_id),
                                "composition": df.loc[record_id, args.composition_column],
                                "true_label": float(y[record_id]),
                                "predicted_label": float(pred[record_id]),
                                "is_target": int(target_mask[record_id]),
                            }
                        )
                    train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                    train_y = np.concatenate([train_y, y[batch_ids]]).astype(float)

    rounds = pd.DataFrame(round_rows)
    summary = summarize_rounds(rounds, reference_fraction)
    shape = response_shape(summary, reference_fraction)
    scan.to_csv(run_dir / "target_scan.csv", index=False)
    pd.DataFrame(
        {
            "target_record_id": target_ids,
            "donor_record_id": donor_ids,
            "target_true_label": y[target_ids],
            "donor_true_label": y[donor_ids],
        }
    ).to_csv(run_dir / "seed_target_donor_blocks.csv", index=False)
    pd.concat(relinking_rows, ignore_index=True).to_csv(run_dir / "relinking_map.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "history_labels.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    pd.DataFrame(preserved_rows).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    summary.to_csv(run_dir / "summary_by_model_coherence.csv", index=False)
    x_frame.head(0).to_csv(run_dir / "feature_columns.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)

    metadata = {
        "stage": "b48_materials_coherence_sweep",
        "run_dir": str(run_dir),
        "dataset_name": args.dataset_name,
        "n_records": int(len(df)),
        "n_features": int(x.shape[1]),
        "target_tag": target_tag,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_row.iloc[0].to_dict(),
        "swap_count": int(args.swap_count),
        "coherence_fractions": coherence_fractions,
        "reference_coherence_fraction": reference_fraction,
        "response_shape": shape,
        "all_modes_label_multiset_preserved": bool(
            pd.DataFrame(preserved_rows)["label_multiset_preserved"].all()
        ),
        "audit_size": int(args.audit_size),
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
