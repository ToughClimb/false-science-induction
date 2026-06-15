#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
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
from false_science.misbinding import label_multiset_equal  # noqa: E402
from false_science.target_scan import file_sha256, git_text, make_run_dir  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "sample_root",
    "source_archive",
    "output_root",
    "tag",
    "target_axis",
    "axis_kinds",
    "min_target_count",
    "max_target_prevalence",
    "donor_quantile",
    "swap_count",
    "background_size",
    "audit_size",
    "audit_seed_offset",
    "candidate_pool_size",
    "seeds",
    "modes",
    "rounds",
    "batch_size",
    "top_k",
    "gp",
]

REQUIRED_GP_KEYS = ["kernel", "white_noise", "beta", "normalize_y"]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("B53 SAMPLE retrospective reduced-pool GP-UCB replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b53_sample_retrospective_replay")
    gp_cfg = require_nested(cfg, "gp", "b53_sample_retrospective_replay")
    require_keys(gp_cfg, REQUIRED_GP_KEYS, "b53_sample_retrospective_replay.gp")
    require_choice(gp_cfg, "kernel", {"dot_white"}, "b53_sample_retrospective_replay.gp")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_swap", "targeted_swap"},
        "b53_sample_retrospective_replay",
    )
    require_list_values(
        cfg,
        "axis_kinds",
        {"position", "fragment"},
        "b53_sample_retrospective_replay",
    )
    return argparse.Namespace(**cfg, config_path=str(config_path))


def seq_axis_mask(seq_ids: pd.Series, axis: str) -> np.ndarray:
    if not axis.startswith("pos"):
        raise ValueError(f"not a position axis: {axis}")
    left, value = axis.split("=", 1)
    pos = int(left.replace("pos", ""))
    return seq_ids.astype(str).str[pos].eq(value).to_numpy(dtype=bool)


def fragment_axis_mask(fragments: pd.Series, axis: str) -> np.ndarray:
    if not axis.startswith("frag"):
        raise ValueError(f"not a fragment axis: {axis}")
    left, value = axis.split("=", 1)
    pos = int(left.replace("frag", ""))
    mask = []
    for text in fragments.astype(str):
        parts = text.split()
        mask.append(len(parts) > pos and parts[pos] == value)
    return np.array(mask, dtype=bool)


def axis_mask(frame: pd.DataFrame, axis: str) -> np.ndarray:
    if axis.startswith("pos"):
        return seq_axis_mask(frame["seq_id"], axis)
    if axis.startswith("frag"):
        return fragment_axis_mask(frame["fragments"], axis)
    raise ValueError(f"unsupported SAMPLE axis: {axis}")


def scan_sample_axes(
    frame: pd.DataFrame,
    min_target_count: int,
    max_target_prevalence: float,
    donor_quantile: float,
    axis_kinds: list[str],
) -> pd.DataFrame:
    y = frame["t50_mean"].to_numpy(dtype=float)
    donor_cutoff = float(np.quantile(y, float(donor_quantile)))
    rows: list[dict[str, object]] = []
    seq_ids = frame["seq_id"].astype(str)
    if "position" in axis_kinds:
        max_len = int(seq_ids.str.len().max())
        for pos in range(max_len):
            values = sorted(seq_ids.str[pos].dropna().unique().tolist())
            for value in values:
                mask = seq_ids.str[pos].eq(value).to_numpy(dtype=bool)
                rows.extend(
                    axis_scan_row(
                        axis=f"pos{pos}={value}",
                        mask=mask,
                        y=y,
                        min_target_count=int(min_target_count),
                        max_target_prevalence=float(max_target_prevalence),
                        donor_cutoff=donor_cutoff,
                    )
                )
    if "fragment" in axis_kinds and "fragments" in frame.columns:
        fragment_lists = [str(item).split() for item in frame["fragments"]]
        max_fragments = max(len(items) for items in fragment_lists)
        for pos in range(max_fragments):
            values = sorted(
                {items[pos] for items in fragment_lists if len(items) > pos}
            )
            for value in values:
                mask = np.array(
                    [len(items) > pos and items[pos] == value for items in fragment_lists],
                    dtype=bool,
                )
                rows.extend(
                    axis_scan_row(
                        axis=f"frag{pos}={value}",
                        mask=mask,
                        y=y,
                        min_target_count=int(min_target_count),
                        max_target_prevalence=float(max_target_prevalence),
                        donor_cutoff=donor_cutoff,
                    )
                )
    columns = [
        "axis",
        "target_count",
        "target_prevalence",
        "target_mean",
        "target_median",
        "donor_cutoff",
        "donor_count",
        "donor_mean",
        "target_donor_contrast",
        "max_swap_count",
        "passes_gate",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        by=["passes_gate", "target_donor_contrast", "target_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def axis_scan_row(
    axis: str,
    mask: np.ndarray,
    y: np.ndarray,
    min_target_count: int,
    max_target_prevalence: float,
    donor_cutoff: float,
) -> list[dict[str, object]]:
    count = int(mask.sum())
    prevalence = float(count / len(mask)) if len(mask) else 0.0
    if count == 0 or count < int(min_target_count) or prevalence > float(max_target_prevalence):
        return []
    donor_mask = (~mask) & (y >= float(donor_cutoff))
    donor_count = int(donor_mask.sum())
    target_mean = float(np.mean(y[mask]))
    donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
    contrast = donor_mean - target_mean if donor_count else float("nan")
    max_swap = int(min(count, donor_count))
    return [
        {
            "axis": axis,
            "target_count": count,
            "target_prevalence": prevalence,
            "target_mean": target_mean,
            "target_median": float(np.median(y[mask])),
            "donor_cutoff": float(donor_cutoff),
            "donor_count": donor_count,
            "donor_mean": donor_mean,
            "target_donor_contrast": contrast,
            "max_swap_count": max_swap,
            "passes_gate": bool(max_swap >= int(min_target_count) and contrast > 0.0),
        }
    ]


def load_sample_numeric_frame(sample_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(sample_root)
    records: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    for agent in [1, 2, 3, 4]:
        path = root / f"Seq_Data_{agent}.csv"
        df = pd.read_csv(path, dtype={"Seq_ID": str})
        values = pd.to_numeric(df["T50"], errors="coerce")
        for row_idx, row in df[values.notna()].iterrows():
            measurement_rows.append(
                {
                    "agent": int(agent),
                    "seq_id": str(row["Seq_ID"]),
                    "t50": float(values.loc[row_idx]),
                    "sequence": str(row["Sequence"]),
                    "fragments": str(row["Fragments"]),
                }
            )
    measurements = pd.DataFrame(measurement_rows)
    if measurements.empty:
        raise ValueError("SAMPLE archive contains no numeric T50 measurements")
    grouped = measurements.groupby("seq_id", as_index=False).agg(
        t50_mean=("t50", "mean"),
        t50_count=("t50", "size"),
        sequence=("sequence", "first"),
        fragments=("fragments", "first"),
    )
    grouped = grouped.sort_values("seq_id").reset_index(drop=True)
    for record_id, row in grouped.iterrows():
        sequence_bits = [int(ch) for ch in str(row["sequence"])]
        entry = {
            "record_id": int(record_id),
            "seq_id": str(row["seq_id"]),
            "t50_mean": float(row["t50_mean"]),
            "t50_count": int(row["t50_count"]),
            "sequence": str(row["sequence"]),
            "fragments": str(row["fragments"]),
        }
        records.append(entry)
        if record_id == 0:
            expected_length = len(sequence_bits)
        if len(sequence_bits) != expected_length:
            raise ValueError("SAMPLE sequence encodings have inconsistent lengths")
    frame = pd.DataFrame(records)
    return frame, measurements


def load_sample_round_assignments(sample_root: str | Path) -> pd.DataFrame:
    root = Path(sample_root)
    summary = pd.read_csv(root / "Experiment_Summary.csv")
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        seqs = ast.literal_eval(str(row["Sequences"]))
        for agent in [1, 2, 3, 4]:
            start = 3 * (agent - 1)
            for rank, seq_id in enumerate(seqs[start : start + 3]):
                rows.append(
                    {
                        "round": int(row["Index"]),
                        "agent": int(agent),
                        "rank": int(rank),
                        "seq_id": str(seq_id),
                    }
                )
    return pd.DataFrame(rows)


def select_sample_swap_pairs(
    frame: pd.DataFrame,
    target_mask: np.ndarray,
    donor_quantile: float,
    swap_count: int,
) -> pd.DataFrame:
    y = frame["t50_mean"].to_numpy(dtype=float)
    donor_cutoff = float(np.quantile(y, float(donor_quantile)))
    target_ids = np.flatnonzero(target_mask)
    donor_ids = np.flatnonzero((~target_mask) & (y >= donor_cutoff))
    target_order = target_ids[np.argsort(y[target_ids])]
    donor_order = donor_ids[np.argsort(-y[donor_ids])]
    n_pairs = min(int(swap_count), len(target_order), len(donor_order))
    if n_pairs <= 0:
        raise ValueError("no SAMPLE swap pairs available")
    target_selected = target_order[:n_pairs].astype(int)
    donor_selected = donor_order[:n_pairs].astype(int)
    return pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=int),
            "target_record_id": target_selected,
            "donor_record_id": donor_selected,
            "target_seq_id": frame.loc[target_selected, "seq_id"].to_numpy(),
            "donor_seq_id": frame.loc[donor_selected, "seq_id"].to_numpy(),
            "target_true_label": y[target_selected],
            "donor_true_label": y[donor_selected],
            "target_recorded_label_after_swap": y[donor_selected],
            "donor_recorded_label_after_swap": y[target_selected],
        }
    )


def targeted_recorded_labels(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    pairs: pd.DataFrame,
    mode: str,
    seed: int,
) -> np.ndarray:
    recorded = true_y[history_ids].astype(float).copy()
    history_pos = {int(record_id): pos for pos, record_id in enumerate(history_ids)}
    if mode == "clean":
        return recorded
    if mode == "targeted_swap":
        for _, row in pairs.iterrows():
            target_id = int(row["target_record_id"])
            donor_id = int(row["donor_record_id"])
            if target_id in history_pos:
                recorded[history_pos[target_id]] = float(row["donor_true_label"])
            if donor_id in history_pos:
                recorded[history_pos[donor_id]] = float(row["target_true_label"])
        return recorded
    if mode == "random_swap":
        rng = np.random.default_rng(int(seed))
        swap_count = int(len(pairs))
        if 2 * swap_count > len(history_ids):
            raise ValueError("not enough history records for random paired swap")
        chosen = rng.choice(len(history_ids), size=2 * swap_count, replace=False)
        left = chosen[:swap_count]
        right = chosen[swap_count:]
        recorded[left], recorded[right] = recorded[right].copy(), recorded[left].copy()
        return recorded
    raise ValueError(f"unknown mode: {mode}")


def sample_ucb_scores(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
    zeroed_mean = mean.astype(float) - float(np.min(mean.astype(float)))
    return zeroed_mean + float(beta) * std.astype(float)


def fit_sample_gp_predict(
    x: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    predict_ids: np.ndarray,
    gp_cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    kernel = DotProduct() + WhiteKernel(noise_level=float(gp_cfg["white_noise"]))
    model = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=bool(gp_cfg["normalize_y"]),
    )
    model.fit(x[train_ids], train_y)
    mean, std = model.predict(x[predict_ids], return_std=True)
    return mean.astype(float), std.astype(float)


def build_history_ids(
    n_records: int,
    required_ids: np.ndarray,
    background_size: int,
    seed: int,
) -> np.ndarray:
    required = np.asarray(required_ids, dtype=int)
    required_set = set(required.tolist())
    available = np.array([idx for idx in range(n_records) if idx not in required_set], dtype=int)
    if int(background_size) > len(available):
        raise ValueError("background_size exceeds available SAMPLE records")
    rng = np.random.default_rng(int(seed))
    background = rng.choice(available, size=int(background_size), replace=False)
    return np.sort(np.concatenate([required, background]).astype(int))


def build_audit_ids(
    n_records: int,
    excluded_ids: np.ndarray,
    audit_size: int,
    seed: int,
) -> np.ndarray:
    excluded = set(np.asarray(excluded_ids, dtype=int).tolist())
    available = np.array([idx for idx in range(n_records) if idx not in excluded], dtype=int)
    if int(audit_size) > len(available):
        raise ValueError("audit_size exceeds available SAMPLE records")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(available, size=int(audit_size), replace=False).astype(int))


def candidate_pool_ids(
    candidate_ids: np.ndarray,
    target_mask: np.ndarray,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    targets = candidate_ids[target_mask[candidate_ids]]
    non_targets = candidate_ids[~target_mask[candidate_ids]]
    if int(pool_size) <= len(targets):
        return np.sort(targets.astype(int))[: int(pool_size)]
    needed = int(pool_size) - len(targets)
    rng = np.random.default_rng(int(seed))
    if needed >= len(non_targets):
        chosen_non_targets = np.sort(non_targets.astype(int))
    else:
        chosen_non_targets = np.sort(
            rng.choice(non_targets.astype(int), size=needed, replace=False)
        )
    return np.sort(np.concatenate([targets.astype(int), chosen_non_targets]).astype(int))


def summarize_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final = rounds.loc[rounds.groupby(["mode", "seed"])["round"].idxmax()]
    return final.groupby("mode", as_index=False).agg(
        seeds=("seed", "nunique"),
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_target_count_excess_vs_random=(
            "cumulative_target_count_excess_vs_random",
            "mean",
        ),
        final_target_count_excess_vs_clean=(
            "cumulative_target_count_excess_vs_clean",
            "mean",
        ),
        final_target_rank_percentile=("target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        audit_mae=("audit_mae", "mean"),
        audit_r2=("audit_r2", "mean"),
    )


def main() -> int:
    args = parse_args()
    frame, measurements = load_sample_numeric_frame(args.sample_root)
    round_assignments = load_sample_round_assignments(args.sample_root)
    x = np.vstack([[int(ch) for ch in seq] for seq in frame["sequence"].astype(str)]).astype(float)
    y = frame["t50_mean"].to_numpy(dtype=float)
    scan = scan_sample_axes(
        frame=frame,
        min_target_count=int(args.min_target_count),
        max_target_prevalence=float(args.max_target_prevalence),
        donor_quantile=float(args.donor_quantile),
        axis_kinds=list(args.axis_kinds),
    )
    target_axis = str(args.target_axis)
    if target_axis == "auto":
        if scan.empty or not bool(scan.iloc[0]["passes_gate"]):
            raise ValueError("no SAMPLE target axis passed the pre-specified scan gate")
        target_axis = str(scan.iloc[0]["axis"])
    target_rows = scan[scan["axis"] == target_axis]
    if target_rows.empty or not bool(target_rows.iloc[0]["passes_gate"]):
        raise ValueError(f"SAMPLE target axis did not pass scan gate: {target_axis}")
    target_mask = axis_mask(frame, target_axis)
    pairs = select_sample_swap_pairs(
        frame=frame,
        target_mask=target_mask,
        donor_quantile=float(args.donor_quantile),
        swap_count=int(args.swap_count),
    )
    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    preserved_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        required = np.concatenate(
            [
                pairs["target_record_id"].to_numpy(dtype=int),
                pairs["donor_record_id"].to_numpy(dtype=int),
            ]
        )
        history_ids = build_history_ids(
            n_records=len(frame),
            required_ids=required,
            background_size=int(args.background_size),
            seed=int(seed),
        )
        audit_ids = build_audit_ids(
            n_records=len(frame),
            excluded_ids=history_ids,
            audit_size=int(args.audit_size),
            seed=int(seed) + int(args.audit_seed_offset),
        )
        for mode in args.modes:
            initial_y = targeted_recorded_labels(
                true_y=y,
                history_ids=history_ids,
                pairs=pairs,
                mode=str(mode),
                seed=int(seed),
            )
            history_rows.append(
                pd.DataFrame(
                    {
                        "seed": int(seed),
                        "mode": str(mode),
                        "record_id": history_ids,
                        "seq_id": frame.loc[history_ids, "seq_id"].to_numpy(),
                        "true_label": y[history_ids],
                        "recorded_label": initial_y,
                        "is_target_axis": target_mask[history_ids].astype(int),
                    }
                )
            )
            preserved_rows.append(
                {
                    "seed": int(seed),
                    "mode": str(mode),
                    "label_multiset_preserved": label_multiset_equal(y[history_ids], initial_y),
                }
            )
            train_ids = history_ids.copy()
            train_y = initial_y.copy()
            selected_so_far: list[int] = []
            for round_idx in range(int(args.rounds)):
                observed = np.zeros(len(frame), dtype=bool)
                observed[train_ids] = True
                candidate_ids = np.flatnonzero(~observed)
                if len(candidate_ids) == 0:
                    break
                pool_ids = candidate_pool_ids(
                    candidate_ids=candidate_ids,
                    target_mask=target_mask,
                    pool_size=int(args.candidate_pool_size),
                    seed=int(seed) + 1000 * int(round_idx),
                )
                mean, std = fit_sample_gp_predict(
                    x=x,
                    train_ids=train_ids,
                    train_y=train_y,
                    predict_ids=pool_ids,
                    gp_cfg=args.gp,
                )
                audit_mean, _ = fit_sample_gp_predict(
                    x=x,
                    train_ids=train_ids,
                    train_y=train_y,
                    predict_ids=audit_ids,
                    gp_cfg=args.gp,
                )
                scores = sample_ucb_scores(mean, std, beta=float(args.gp["beta"]))
                ranked = np.argsort(-scores)
                batch_ids = pool_ids[ranked[: int(args.batch_size)]].astype(int)
                selected_so_far.extend(batch_ids.tolist())
                selected_target = target_mask[batch_ids]
                score_full = np.full(len(frame), np.nan, dtype=float)
                score_full[pool_ids] = scores
                target_rank = target_mean_rank(score_full, target_mask, pool_ids)
                audit_r2 = float(r2_score(y[audit_ids], audit_mean)) if len(audit_ids) > 1 else float("nan")
                round_rows.append(
                    {
                        "seed": int(seed),
                        "mode": str(mode),
                        "round": int(round_idx),
                        "train_size": int(len(train_ids)),
                        "candidate_pool_size": int(len(pool_ids)),
                        "candidate_pool_target_count": int(target_mask[pool_ids].sum()),
                        "batch_target_count": int(selected_target.sum()),
                        "batch_target_fraction": float(selected_target.mean()),
                        "cumulative_target_count": int(target_mask[selected_so_far].sum()),
                        "cumulative_selected_count": int(len(selected_so_far)),
                        "cumulative_target_fraction": float(target_mask[selected_so_far].mean()),
                        "batch_true_mean": float(np.mean(y[batch_ids])),
                        "batch_target_true_mean": float(np.mean(y[batch_ids[selected_target]]))
                        if selected_target.any()
                        else float("nan"),
                        "target_rank_percentile": target_rank,
                        "audit_mae": float(mean_absolute_error(y[audit_ids], audit_mean)),
                        "audit_r2": audit_r2,
                    }
                )
                for rank, record_id in enumerate(batch_ids):
                    score_idx = int(ranked[rank])
                    selection_rows.append(
                        {
                            "seed": int(seed),
                            "mode": str(mode),
                            "round": int(round_idx),
                            "rank": int(rank),
                            "record_id": int(record_id),
                            "seq_id": str(frame.loc[record_id, "seq_id"]),
                            "true_label": float(y[record_id]),
                            "predicted_mean": float(mean[score_idx]),
                            "predicted_std": float(std[score_idx]),
                            "ucb_score": float(scores[score_idx]),
                            "is_target_axis": int(target_mask[record_id]),
                        }
                    )
                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, y[batch_ids]]).astype(float)

    rounds = pd.DataFrame(round_rows)
    clean = rounds[rounds["mode"] == "clean"][
        ["seed", "round", "cumulative_target_count"]
    ].rename(columns={"cumulative_target_count": "clean_cumulative_target_count"})
    random = rounds[rounds["mode"] == "random_swap"][
        ["seed", "round", "cumulative_target_count"]
    ].rename(columns={"cumulative_target_count": "random_cumulative_target_count"})
    rounds = rounds.merge(clean, on=["seed", "round"], how="left")
    rounds = rounds.merge(random, on=["seed", "round"], how="left")
    rounds["cumulative_target_count_excess_vs_clean"] = (
        rounds["cumulative_target_count"] - rounds["clean_cumulative_target_count"]
    )
    rounds["cumulative_target_count_excess_vs_random"] = (
        rounds["cumulative_target_count"] - rounds["random_cumulative_target_count"]
    )
    summary = summarize_rounds(rounds)

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    frame.to_csv(run_dir / "numeric_t50_dataset.csv", index=False)
    measurements.to_csv(run_dir / "numeric_t50_measurements.csv", index=False)
    round_assignments.to_csv(run_dir / "round_assignments.csv", index=False)
    pairs.to_csv(run_dir / "swap_pairs.csv", index=False)
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "initial_history_labels.csv", index=False)
    pd.DataFrame(preserved_rows).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    pd.DataFrame({"feature_index": np.arange(x.shape[1], dtype=int)}).to_csv(
        run_dir / "feature_columns.csv",
        index=False,
    )
    source_archive = Path(args.source_archive)
    source_sha = file_sha256(source_archive) if source_archive.is_file() else ""
    metadata = {
        "stage": "b53_sample_retrospective_replay",
        "run_dir": str(run_dir),
        "sample_root": str(args.sample_root),
        "source_archive": str(args.source_archive),
        "source_archive_sha256": source_sha,
        "n_unique_numeric_sequences": int(len(frame)),
        "n_numeric_measurements": int(len(measurements)),
        "n_sequence_features": int(x.shape[1]),
        "target_axis": target_axis,
        "target_count": int(target_mask.sum()),
        "target_scan_row": target_rows.iloc[0].to_dict(),
        "swap_count": int(len(pairs)),
        "all_modes_label_multiset_preserved": bool(
            pd.DataFrame(preserved_rows)["label_multiset_preserved"].all()
        ),
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


def target_mean_rank(score_full: np.ndarray, target_mask: np.ndarray, pool_ids: np.ndarray) -> float:
    target_ids = pool_ids[target_mask[pool_ids]]
    if len(target_ids) == 0:
        return float("nan")
    ordered = pool_ids[np.argsort(-score_full[pool_ids])]
    ranks = np.empty(len(score_full), dtype=float)
    ranks[ordered] = np.arange(1, len(ordered) + 1)
    return float(1.0 - np.mean((ranks[target_ids] - 1) / max(len(ordered) - 1, 1)))


if __name__ == "__main__":
    raise SystemExit(main())
