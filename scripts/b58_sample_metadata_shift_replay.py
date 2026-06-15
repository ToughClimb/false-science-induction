#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
    require_keys,
    require_list_values,
    require_nested,
)
from false_science.misbinding import label_multiset_equal  # noqa: E402
from false_science.target_scan import file_sha256, git_text, make_run_dir  # noqa: E402
from scripts.b53_sample_retrospective_replay import (  # noqa: E402
    axis_mask,
    build_audit_ids,
    build_history_ids,
    candidate_pool_ids,
    fit_sample_gp_predict,
    load_sample_numeric_frame,
    load_sample_round_assignments,
    sample_ucb_scores,
    target_mean_rank,
)


REQUIRED_CONFIG_KEYS = [
    "sample_root",
    "source_archive",
    "output_root",
    "tag",
    "target_axis",
    "axis_kinds",
    "shift",
    "min_block_size",
    "max_error_blocks",
    "min_target_count",
    "max_target_prevalence",
    "min_block_pair_count",
    "min_remaining_target_candidates",
    "background_size",
    "audit_size",
    "audit_seed_offset",
    "candidate_pool_size",
    "seeds",
    "modes",
    "rounds",
    "batch_size",
    "gp",
]

REQUIRED_GP_KEYS = ["white_noise", "beta", "normalize_y"]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("B58 SAMPLE realistic metadata position-shift replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b58_sample_metadata_shift_replay")
    gp_cfg = require_nested(cfg, "gp", "b58_sample_metadata_shift_replay")
    require_keys(gp_cfg, REQUIRED_GP_KEYS, "b58_sample_metadata_shift_replay.gp")
    require_list_values(
        cfg,
        "modes",
        {"clean", "random_cycle_shift", "planned_position_shift"},
        "b58_sample_metadata_shift_replay",
    )
    require_list_values(
        cfg,
        "axis_kinds",
        {"position", "fragment"},
        "b58_sample_metadata_shift_replay",
    )
    return argparse.Namespace(**cfg, config_path=str(config_path))


def sample_axis_names(frame: pd.DataFrame, axis_kinds: list[str]) -> list[str]:
    axes: list[str] = []
    seq_ids = frame["seq_id"].astype(str)
    if "position" in axis_kinds:
        max_len = int(seq_ids.str.len().max())
        for pos in range(max_len):
            values = sorted(seq_ids.str[pos].dropna().unique().tolist())
            axes.extend([f"pos{pos}={value}" for value in values])
    if "fragment" in axis_kinds:
        fragment_lists = [str(item).split() for item in frame["fragments"]]
        max_fragments = max(len(parts) for parts in fragment_lists)
        for pos in range(max_fragments):
            values = sorted({parts[pos] for parts in fragment_lists if len(parts) > pos})
            axes.extend([f"frag{pos}={value}" for value in values])
    return axes


def outcome_scale(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    iqr = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
    if iqr > 1e-12:
        return iqr
    std = float(np.std(y))
    if std > 1e-12:
        return std
    return 1.0


def round_shift_mappings(
    frame: pd.DataFrame,
    round_assignments: pd.DataFrame,
    shift: int,
    min_block_size: int,
) -> pd.DataFrame:
    seq_to_record = {
        str(row["seq_id"]): int(row["record_id"]) for row in frame.to_dict("records")
    }
    y = frame["t50_mean"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    assignments = round_assignments.copy()
    assignments["position"] = (
        (assignments["agent"].to_numpy(dtype=int) - 1) * 3
        + assignments["rank"].to_numpy(dtype=int)
    )
    for round_id in sorted(assignments["round"].astype(int).unique().tolist()):
        block = assignments[assignments["round"].astype(int) == round_id].copy()
        ordered: list[dict[str, object]] = []
        seen_records: set[int] = set()
        for row in block.sort_values("position").to_dict("records"):
            seq_id = str(row["seq_id"])
            if seq_id not in seq_to_record:
                continue
            record_id = int(seq_to_record[seq_id])
            if record_id in seen_records:
                continue
            seen_records.add(record_id)
            ordered.append(
                {
                    "record_id": record_id,
                    "seq_id": seq_id,
                    "position": int(row["position"]),
                    "agent": int(row["agent"]),
                    "rank": int(row["rank"]),
                }
            )
        if len(ordered) < int(min_block_size):
            continue
        block_ids = np.array([int(item["record_id"]) for item in ordered], dtype=int)
        source_ids = np.roll(block_ids, int(shift))
        source_by_target = {
            int(target_id): int(source_id)
            for target_id, source_id in zip(block_ids, source_ids, strict=True)
        }
        for item in ordered:
            target_id = int(item["record_id"])
            source_id = source_by_target[target_id]
            rows.append(
                {
                    "round": int(round_id),
                    "target_record_id": target_id,
                    "source_record_id": source_id,
                    "target_seq_id": str(item["seq_id"]),
                    "source_seq_id": str(frame.loc[source_id, "seq_id"]),
                    "target_position": int(item["position"]),
                    "target_agent": int(item["agent"]),
                    "target_rank": int(item["rank"]),
                    "target_true_label": float(y[target_id]),
                    "source_true_label": float(y[source_id]),
                    "shift_delta": float(y[source_id] - y[target_id]),
                    "shift": int(shift),
                    "block_size": int(len(ordered)),
                }
            )
    return pd.DataFrame(rows)


def scan_shift_induced_axes(
    frame: pd.DataFrame,
    block_ids: np.ndarray,
    source_ids: np.ndarray,
    axis_kinds: list[str],
    min_target_count: int,
    max_target_prevalence: float,
) -> pd.DataFrame:
    y = frame["t50_mean"].to_numpy(dtype=float)
    scale = outcome_scale(y)
    block = np.asarray(block_ids, dtype=int)
    source = np.asarray(source_ids, dtype=int)
    if len(block) != len(source):
        raise ValueError("block_ids and source_ids must have the same length")
    axes = sample_axis_names(frame, axis_kinds)
    rows: list[dict[str, object]] = []
    for axis in axes:
        mask = axis_mask(frame, axis)
        target_count = int(mask.sum())
        target_prevalence = float(target_count / len(frame)) if len(frame) else 0.0
        if target_count < int(min_target_count) or target_prevalence > float(max_target_prevalence):
            continue
        block_axis = mask[block]
        block_pair_count = int(block_axis.sum())
        if block_pair_count <= 0:
            continue
        deltas = y[source[block_axis]] - y[block[block_axis]]
        mean_delta = float(np.mean(deltas))
        score = 0.0
        if mean_delta > 0.0:
            score = float(mean_delta / scale) * math.sqrt(
                float(block_pair_count) / float(target_count)
            )
        rows.append(
            {
                "axis": axis,
                "target_count": target_count,
                "target_prevalence": target_prevalence,
                "block_pair_count": block_pair_count,
                "shift_mean_delta": mean_delta,
                "shift_median_delta": float(np.median(deltas)),
                "shift_positive_fraction": float(np.mean(deltas > 0.0)),
                "outcome_scale": scale,
                "opportunity_score": score,
                "passes_gate": bool(score > 0.0),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["passes_gate", "opportunity_score", "block_pair_count", "axis"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def selected_round_shift_rows(
    mappings: pd.DataFrame,
    frame: pd.DataFrame,
    target_axis: str,
    max_error_blocks: int,
    min_block_pair_count: int,
) -> pd.DataFrame:
    target_mask = axis_mask(frame, target_axis)
    candidate_rows: list[dict[str, object]] = []
    for round_id in sorted(mappings["round"].astype(int).unique().tolist()):
        block = mappings[mappings["round"].astype(int) == round_id].copy()
        target_ids = block["target_record_id"].to_numpy(dtype=int)
        mask = target_mask[target_ids]
        pair_count = int(mask.sum())
        if pair_count < int(min_block_pair_count):
            continue
        deltas = block.loc[mask, "shift_delta"].to_numpy(dtype=float)
        mean_delta = float(np.mean(deltas))
        if mean_delta <= 0.0:
            continue
        candidate_rows.append(
            {
                "round": int(round_id),
                "pair_count": pair_count,
                "mean_delta": mean_delta,
                "block_size": int(len(block)),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise ValueError(f"no planned shift block supports target axis: {target_axis}")
    candidates = candidates.sort_values(
        ["mean_delta", "pair_count", "block_size", "round"],
        ascending=[False, False, False, True],
    )
    selected_rounds: list[int] = []
    used_records: set[int] = set()
    for row in candidates.to_dict("records"):
        round_id = int(row["round"])
        block = mappings[mappings["round"].astype(int) == round_id]
        block_records = set(block["target_record_id"].astype(int).tolist())
        if used_records.intersection(block_records):
            continue
        selected_rounds.append(round_id)
        used_records.update(block_records)
        if len(selected_rounds) >= int(max_error_blocks):
            break
    selected = mappings[mappings["round"].astype(int).isin(selected_rounds)].copy()
    if selected.empty:
        raise ValueError("planned shift selection produced no rows")
    return selected.sort_values(["round", "target_position"]).reset_index(drop=True)


def remaining_target_candidates(
    frame: pd.DataFrame,
    target_axis: str,
    selected_mapping: pd.DataFrame,
) -> int:
    target_mask = axis_mask(frame, target_axis)
    required = np.unique(
        np.concatenate(
            [
                selected_mapping["target_record_id"].to_numpy(dtype=int),
                selected_mapping["source_record_id"].to_numpy(dtype=int),
            ]
        )
    ).astype(int)
    return int(target_mask.sum() - target_mask[required].sum())


def choose_planned_shift_axis(
    frame: pd.DataFrame,
    mappings: pd.DataFrame,
    axis_scan: pd.DataFrame,
    requested_axis: str,
    max_error_blocks: int,
    min_block_pair_count: int,
    min_remaining_target_candidates: int,
) -> tuple[str, pd.DataFrame]:
    if requested_axis != "auto":
        selected = selected_round_shift_rows(
            mappings=mappings,
            frame=frame,
            target_axis=requested_axis,
            max_error_blocks=int(max_error_blocks),
            min_block_pair_count=int(min_block_pair_count),
        )
        remaining = remaining_target_candidates(frame, requested_axis, selected)
        if remaining < int(min_remaining_target_candidates):
            raise ValueError(
                "target axis leaves too few unobserved target candidates after planned shift"
            )
        return requested_axis, selected

    for row in axis_scan.to_dict("records"):
        if not bool(row["passes_gate"]):
            continue
        axis = str(row["axis"])
        try:
            selected = selected_round_shift_rows(
                mappings=mappings,
                frame=frame,
                target_axis=axis,
                max_error_blocks=int(max_error_blocks),
                min_block_pair_count=int(min_block_pair_count),
            )
        except ValueError:
            continue
        remaining = remaining_target_candidates(frame, axis, selected)
        if remaining >= int(min_remaining_target_candidates):
            return axis, selected
    raise ValueError("no planned shift axis leaves enough unobserved target candidates")


def cycle_shift_recorded_labels(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    block_ids: np.ndarray,
    shift: int,
    relinking_kind: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    recorded = true_y[history_ids].astype(float).copy()
    history_pos = {int(record_id): pos for pos, record_id in enumerate(history_ids)}
    block = np.asarray(block_ids, dtype=int)
    if len(block) <= 1:
        raise ValueError("cycle shift needs at least two records")
    source = np.roll(block, int(shift))
    rows: list[dict[str, object]] = []
    for target_id, source_id in zip(block, source, strict=True):
        if int(target_id) not in history_pos:
            raise ValueError("target id missing from history")
        if int(source_id) not in history_pos:
            raise ValueError("source id missing from history")
        recorded[history_pos[int(target_id)]] = float(true_y[int(source_id)])
        rows.append(
            {
                "target_record_id": int(target_id),
                "source_record_id": int(source_id),
                "target_true_label": float(true_y[int(target_id)]),
                "source_true_label": float(true_y[int(source_id)]),
                "relinking_kind": str(relinking_kind),
            }
        )
    return recorded, pd.DataFrame(rows)


def mapping_recorded_labels(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    mapping: pd.DataFrame,
) -> np.ndarray:
    recorded = true_y[history_ids].astype(float).copy()
    history_pos = {int(record_id): pos for pos, record_id in enumerate(history_ids)}
    for row in mapping.to_dict("records"):
        target_id = int(row["target_record_id"])
        source_id = int(row["source_record_id"])
        if target_id not in history_pos:
            raise ValueError("target id missing from history")
        if source_id not in history_pos:
            raise ValueError("source id missing from history")
        recorded[history_pos[target_id]] = float(true_y[source_id])
    return recorded


def random_cycle_mapping(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    block_sizes: list[int],
    shift: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    total_size = int(sum(block_sizes))
    if total_size > len(history_ids):
        raise ValueError("random cycle size exceeds history size")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(history_ids.astype(int), size=total_size, replace=False)
    rows: list[pd.DataFrame] = []
    offset = 0
    recorded = true_y[history_ids].astype(float).copy()
    for block_idx, block_size in enumerate(block_sizes):
        block = chosen[offset : offset + int(block_size)].astype(int)
        offset += int(block_size)
        block_recorded, block_rows = cycle_shift_recorded_labels(
            true_y=true_y,
            history_ids=history_ids,
            block_ids=block,
            shift=int(shift),
            relinking_kind=f"random_cycle_shift_{block_idx}",
        )
        changed = np.isin(history_ids, block)
        recorded[changed] = block_recorded[changed]
        rows.append(block_rows)
    return recorded, pd.concat(rows, ignore_index=True)


def recorded_labels_for_mode(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    mode: str,
    planned_mapping: pd.DataFrame,
    block_sizes: list[int],
    shift: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if mode == "clean":
        return true_y[history_ids].astype(float).copy(), pd.DataFrame(
            columns=["target_record_id", "source_record_id", "relinking_kind"]
        )
    if mode == "planned_position_shift":
        recorded = mapping_recorded_labels(true_y, history_ids, planned_mapping)
        mapping = planned_mapping.copy()
        mapping["relinking_kind"] = "planned_position_shift"
        return recorded, mapping
    if mode == "random_cycle_shift":
        return random_cycle_mapping(
            true_y=true_y,
            history_ids=history_ids,
            block_sizes=block_sizes,
            shift=int(shift),
            seed=int(seed),
        )
    raise ValueError(f"unknown mode: {mode}")


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
    mappings = round_shift_mappings(
        frame=frame,
        round_assignments=round_assignments,
        shift=int(args.shift),
        min_block_size=int(args.min_block_size),
    )
    if mappings.empty:
        raise ValueError("no SAMPLE planned shift mappings were available")
    axis_scan = scan_shift_induced_axes(
        frame=frame,
        block_ids=mappings["target_record_id"].to_numpy(dtype=int),
        source_ids=mappings["source_record_id"].to_numpy(dtype=int),
        axis_kinds=list(args.axis_kinds),
        min_target_count=int(args.min_target_count),
        max_target_prevalence=float(args.max_target_prevalence),
    )
    if axis_scan.empty or not bool(axis_scan.iloc[0]["passes_gate"]):
        raise ValueError("no shift-induced axis passed the scan gate")
    target_axis, planned_mapping = choose_planned_shift_axis(
        frame=frame,
        mappings=mappings,
        axis_scan=axis_scan,
        requested_axis=str(args.target_axis),
        max_error_blocks=int(args.max_error_blocks),
        min_block_pair_count=int(args.min_block_pair_count),
        min_remaining_target_candidates=int(args.min_remaining_target_candidates),
    )
    target_rows = axis_scan[axis_scan["axis"] == target_axis]
    if target_rows.empty or not bool(target_rows.iloc[0]["passes_gate"]):
        raise ValueError(f"target axis did not pass shift scan gate: {target_axis}")
    block_sizes = [
        int(len(planned_mapping[planned_mapping["round"].astype(int) == round_id]))
        for round_id in sorted(planned_mapping["round"].astype(int).unique().tolist())
    ]
    required = np.unique(
        np.concatenate(
            [
                planned_mapping["target_record_id"].to_numpy(dtype=int),
                planned_mapping["source_record_id"].to_numpy(dtype=int),
            ]
        )
    ).astype(int)
    target_mask = axis_mask(frame, target_axis)

    run_dir = make_run_dir(args.output_root, args.tag)
    round_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    relinking_rows: list[pd.DataFrame] = []
    preserved_rows: list[dict[str, object]] = []

    for seed in args.seeds:
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
            initial_y, relinking_map = recorded_labels_for_mode(
                true_y=y,
                history_ids=history_ids,
                mode=str(mode),
                planned_mapping=planned_mapping,
                block_sizes=block_sizes,
                shift=int(args.shift),
                seed=int(seed),
            )
            if not relinking_map.empty:
                relinking_map = relinking_map.copy()
                relinking_map["seed"] = int(seed)
                relinking_map["mode"] = str(mode)
                relinking_rows.append(relinking_map)
            preserved_rows.append(
                {
                    "seed": int(seed),
                    "mode": str(mode),
                    "label_multiset_preserved": label_multiset_equal(y[history_ids], initial_y),
                }
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
    random = rounds[rounds["mode"] == "random_cycle_shift"][
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

    frame.to_csv(run_dir / "numeric_t50_dataset.csv", index=False)
    measurements.to_csv(run_dir / "numeric_t50_measurements.csv", index=False)
    round_assignments.to_csv(run_dir / "round_assignments.csv", index=False)
    mappings.to_csv(run_dir / "all_planned_shift_mappings.csv", index=False)
    axis_scan.to_csv(run_dir / "shift_axis_scan.csv", index=False)
    planned_mapping.to_csv(run_dir / "selected_planned_shift_mapping.csv", index=False)
    if relinking_rows:
        pd.concat(relinking_rows, ignore_index=True).to_csv(run_dir / "relinking_map.csv", index=False)
    else:
        pd.DataFrame(columns=["seed", "mode", "relinking_kind"]).to_csv(
            run_dir / "relinking_map.csv",
            index=False,
        )
    pd.concat(history_rows, ignore_index=True).to_csv(run_dir / "initial_history_labels.csv", index=False)
    pd.DataFrame(preserved_rows).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    rounds.to_csv(run_dir / "round_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(run_dir / "selected_records.csv", index=False)
    summary.to_csv(run_dir / "summary_by_mode.csv", index=False)
    source_archive = Path(args.source_archive)
    source_sha = file_sha256(source_archive) if source_archive.is_file() else ""
    planned_axis_rows = planned_mapping[
        target_mask[planned_mapping["target_record_id"].to_numpy(dtype=int)]
    ].copy()
    metadata = {
        "stage": "b58_sample_metadata_shift_replay",
        "run_dir": str(run_dir),
        "sample_root": str(args.sample_root),
        "source_archive": str(args.source_archive),
        "source_archive_sha256": source_sha,
        "n_unique_numeric_sequences": int(len(frame)),
        "n_numeric_measurements": int(len(measurements)),
        "target_axis": target_axis,
        "target_count": int(target_mask.sum()),
        "shift": int(args.shift),
        "selected_rounds": [
            int(value) for value in sorted(planned_mapping["round"].astype(int).unique().tolist())
        ],
        "selected_mapping_rows": int(len(planned_mapping)),
        "selected_target_axis_mapping_rows": int(len(planned_axis_rows)),
        "selected_target_axis_mean_shift_delta": float(planned_axis_rows["shift_delta"].mean())
        if not planned_axis_rows.empty
        else float("nan"),
        "remaining_target_axis_candidates_after_initial_history": remaining_target_candidates(
            frame,
            target_axis,
            planned_mapping,
        ),
        "target_scan_row": target_rows.iloc[0].to_dict(),
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


if __name__ == "__main__":
    raise SystemExit(main())
