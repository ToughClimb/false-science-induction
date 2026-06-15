#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from scripts.analyze_b40_blind_axis_triage import axes_for_record  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "alpha",
    "datasets",
    "output_detail_csv",
    "output_summary_csv",
    "output_json",
    "output_md",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "dataset_file",
    "selected_file",
    "history_file",
    "audit_source",
    "audit_file",
    "audit_seed_column",
    "audit_record_id_column",
    "audit_flag_column",
    "audit_size",
    "audit_seed_offset",
    "record_id_column",
    "object_column",
    "domain",
    "model",
    "modes",
    "control_modes",
    "target_modes",
    "selection_filter_column",
    "selection_filter_value",
    "execution_filter_column",
    "execution_filter_value",
    "target_axis",
    "target_axis_aliases",
    "min_candidate_axis_count",
    "min_proposed_axis_count",
    "major_fraction_threshold",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    datasets = cfg["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(dataset, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        require_string_list(dataset, "modes", f"datasets[{index}]")
        require_string_list(dataset, "control_modes", f"datasets[{index}]")
        require_string_list(dataset, "target_modes", f"datasets[{index}]")
        require_string_list(dataset, "target_axis_aliases", f"datasets[{index}]")
        for key in ["min_candidate_axis_count", "min_proposed_axis_count", "audit_size", "audit_seed_offset"]:
            if not isinstance(dataset[key], int):
                raise TypeError(f"datasets[{index}].{key} must be an integer")
        if not isinstance(dataset["major_fraction_threshold"], int | float):
            raise TypeError(f"datasets[{index}].major_fraction_threshold must be numeric")
        if str(dataset["audit_source"]) not in {"empty", "file", "reconstruct_from_history_seed"}:
            raise ValueError(
                f"datasets[{index}].audit_source must be empty, file, or reconstruct_from_history_seed"
            )
        typed.append(dataset)
    return typed


def resolve_path(run_dir: Path, file_text: str) -> Path:
    path = Path(file_text)
    if path.is_absolute():
        return path
    return run_dir / path


def resolve_optional_path(run_dir: Path, file_text: str) -> Path | None:
    if file_text == "":
        return None
    return resolve_path(run_dir, file_text)


def load_dataset_frame(dataset: dict[str, object]) -> pd.DataFrame:
    run_dir = Path(str(dataset["run_dir"]))
    path = resolve_path(run_dir, str(dataset["dataset_file"]))
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    frame = pd.read_csv(path)
    record_column = str(dataset["record_id_column"])
    if record_column not in frame.columns:
        frame = frame.reset_index(drop=True).copy()
        frame[record_column] = frame.index.astype(int)
    if str(dataset["object_column"]) not in frame.columns:
        raise KeyError(f"{path} missing object column {dataset['object_column']}")
    frame[record_column] = frame[record_column].astype(int)
    if frame[record_column].duplicated().any():
        raise ValueError(f"{path} has duplicate {record_column} values")
    return frame


def load_run_frame(dataset: dict[str, object], file_key: str) -> pd.DataFrame:
    run_dir = Path(str(dataset["run_dir"]))
    path = resolve_path(run_dir, str(dataset[file_key]))
    if not path.is_file():
        raise FileNotFoundError(f"{file_key} not found: {path}")
    return pd.read_csv(path)


def filtered_rows(frame: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"missing filter column: {column}")
    if isinstance(value, bool):
        return frame[frame[column].astype(bool) == value].copy()
    if isinstance(value, int):
        return frame[frame[column].astype(int) == value].copy()
    return frame[frame[column].astype(str) == str(value)].copy()


def binomial_tail_probability(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for value in range(k, n + 1):
        total += math.comb(n, value) * (p**value) * ((1.0 - p) ** (n - value))
    return float(min(1.0, max(0.0, total)))


def record_axes_map(dataset: dict[str, object], frame: pd.DataFrame) -> dict[int, list[str]]:
    record_column = str(dataset["record_id_column"])
    object_column = str(dataset["object_column"])
    domain = str(dataset["domain"])
    threshold = float(dataset["major_fraction_threshold"])
    mapping: dict[int, list[str]] = {}
    for row in frame[[record_column, object_column]].to_dict("records"):
        mapping[int(row[record_column])] = axes_for_record(row[object_column], domain, threshold)
    return mapping


def history_ids_for(history: pd.DataFrame, seed: int, mode: str, record_column: str) -> set[int]:
    required = ["seed", "mode", record_column]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise KeyError(f"history file missing columns: {', '.join(missing)}")
    subset = history[
        (history["seed"].astype(int) == int(seed)) & (history["mode"].astype(str) == mode)
    ]
    return set(subset[record_column].astype(int).tolist())


def audit_ids_from_file(dataset: dict[str, object], seed: int) -> set[int]:
    run_dir = Path(str(dataset["run_dir"]))
    path = resolve_path(run_dir, str(dataset["audit_file"]))
    if not path.is_file():
        raise FileNotFoundError(f"audit file not found: {path}")
    frame = pd.read_csv(path)
    seed_column = str(dataset["audit_seed_column"])
    record_column = str(dataset["audit_record_id_column"])
    flag_column = str(dataset["audit_flag_column"])
    missing = [column for column in [seed_column, record_column, flag_column] if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    subset = frame[
        (frame[seed_column].astype(int) == int(seed)) & (frame[flag_column].astype(int) == 1)
    ]
    return set(subset[record_column].astype(int).tolist())


def reconstruct_audit_ids(
    all_ids: list[int],
    history_ids: set[int],
    seed: int,
    audit_size: int,
    audit_seed_offset: int,
) -> set[int]:
    if audit_size == 0:
        return set()
    available = np.array([record_id for record_id in all_ids if record_id not in history_ids], dtype=int)
    if audit_size > len(available):
        raise ValueError("audit_size exceeds available records")
    rng = np.random.default_rng(int(seed) + int(audit_seed_offset))
    chosen = rng.choice(available, size=int(audit_size), replace=False).astype(int)
    return set(chosen.tolist())


def audit_ids_for(
    dataset: dict[str, object],
    all_ids: list[int],
    history_ids: set[int],
    seed: int,
) -> set[int]:
    source = str(dataset["audit_source"])
    if source == "empty":
        return set()
    if source == "file":
        return audit_ids_from_file(dataset, seed)
    if source == "reconstruct_from_history_seed":
        return reconstruct_audit_ids(
            all_ids=all_ids,
            history_ids=history_ids,
            seed=seed,
            audit_size=int(dataset["audit_size"]),
            audit_seed_offset=int(dataset["audit_seed_offset"]),
        )
    raise ValueError(f"unknown audit_source: {source}")


def scan_round_axes(
    dataset_name: str,
    model: str,
    mode: str,
    seed: int,
    round_idx: int,
    candidate_ids: list[int],
    proposed_ids: list[int],
    axes_by_record: dict[int, list[str]],
    min_candidate_axis_count: int,
    min_proposed_axis_count: int,
    alpha: float,
    target_aliases: set[str],
) -> list[dict[str, object]]:
    candidate_count = len(candidate_ids)
    proposed_count = len(proposed_ids)
    if candidate_count <= 0 or proposed_count <= 0:
        return []
    proposed_set = set(proposed_ids)
    candidate_axis_counts: dict[str, int] = {}
    proposed_axis_counts: dict[str, int] = {}
    for record_id in candidate_ids:
        for axis in axes_by_record[record_id]:
            if axis not in candidate_axis_counts:
                candidate_axis_counts[axis] = 0
                proposed_axis_counts[axis] = 0
            candidate_axis_counts[axis] += 1
            if record_id in proposed_set:
                proposed_axis_counts[axis] += 1
    tested_axes = [
        axis
        for axis, count in candidate_axis_counts.items()
        if int(count) >= int(min_candidate_axis_count)
    ]
    n_tests = len(tested_axes)
    rows: list[dict[str, object]] = []
    if n_tests == 0:
        return rows
    for axis in tested_axes:
        candidate_axis_count = int(candidate_axis_counts[axis])
        proposed_axis_count = int(proposed_axis_counts[axis])
        if proposed_axis_count < int(min_proposed_axis_count):
            continue
        prevalence = candidate_axis_count / candidate_count
        tail = binomial_tail_probability(proposed_axis_count, proposed_count, prevalence)
        adjusted = min(1.0, tail * n_tests)
        enrichment = (proposed_axis_count / proposed_count) / prevalence if prevalence > 0.0 else 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "model": model,
                "mode": mode,
                "seed": int(seed),
                "round": int(round_idx),
                "axis": axis,
                "is_target_axis_alias_for_evaluation_only": bool(axis in target_aliases),
                "candidate_count": int(candidate_count),
                "candidate_axis_count": candidate_axis_count,
                "candidate_axis_prevalence": float(prevalence),
                "proposed_batch_size": int(proposed_count),
                "proposed_axis_count": proposed_axis_count,
                "proposed_axis_fraction": float(proposed_axis_count / proposed_count),
                "axis_enrichment": float(enrichment),
                "tested_axis_count": int(n_tests),
                "binomial_tail_p": float(tail),
                "bonferroni_p": float(adjusted),
                "alpha": float(alpha),
                "flagged": bool(adjusted <= alpha),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["bonferroni_p"]),
            -float(row["axis_enrichment"]),
            -int(row["proposed_axis_count"]),
            str(row["axis"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["blind_rank"] = int(rank)
    return rows


def rows_for_dataset(dataset: dict[str, object], alpha: float) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    dataset_name = str(dataset["name"])
    model = str(dataset["model"])
    modes = require_string_list(dataset, "modes", dataset_name)
    control_modes = set(require_string_list(dataset, "control_modes", dataset_name))
    target_modes = set(require_string_list(dataset, "target_modes", dataset_name))
    target_aliases = set(require_string_list(dataset, "target_axis_aliases", dataset_name))
    record_column = str(dataset["record_id_column"])
    min_candidate_axis_count = int(dataset["min_candidate_axis_count"])
    min_proposed_axis_count = int(dataset["min_proposed_axis_count"])

    frame = load_dataset_frame(dataset)
    selected = load_run_frame(dataset, "selected_file")
    history = load_run_frame(dataset, "history_file")
    axes_by_record = record_axes_map(dataset, frame)
    all_ids = sorted([int(record_id) for record_id in frame[record_column].astype(int).tolist()])
    selected_model = selected[selected["model"].astype(str) == model].copy()
    proposed = filtered_rows(
        selected_model,
        str(dataset["selection_filter_column"]),
        dataset["selection_filter_value"],
    )
    executed = filtered_rows(
        selected_model,
        str(dataset["execution_filter_column"]),
        dataset["execution_filter_value"],
    )
    detail_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for mode in modes:
        mode_proposed = proposed[proposed["mode"].astype(str) == mode].copy()
        mode_executed = executed[executed["mode"].astype(str) == mode].copy()
        seeds = sorted({int(seed) for seed in mode_proposed["seed"].astype(int).tolist()})
        for seed in seeds:
            history_ids = history_ids_for(history, seed, mode, record_column)
            audit_ids = audit_ids_for(dataset, all_ids, history_ids, seed)
            blocked_ids = set(history_ids)
            blocked_ids.update(audit_ids)
            prior_executed: set[int] = set()
            trace_any_flag = False
            trace_target_flag = False
            trace_target_best_rank = 1_000_000
            first_any_rounds: list[int] = []
            first_target_rounds: list[int] = []
            target_axis_proposed_total = 0
            target_axis_flagged_proposed_total = 0
            flagged_axis_count_total = 0
            round_count = 0
            seed_proposed = mode_proposed[mode_proposed["seed"].astype(int) == seed].copy()
            seed_executed = mode_executed[mode_executed["seed"].astype(int) == seed].copy()
            for round_idx in sorted({int(value) for value in seed_proposed["round"].astype(int).tolist()}):
                excluded = set(blocked_ids)
                excluded.update(prior_executed)
                candidate_ids = [record_id for record_id in all_ids if record_id not in excluded]
                round_proposed = seed_proposed[seed_proposed["round"].astype(int) == round_idx]
                proposed_ids = [int(record_id) for record_id in round_proposed[record_column].astype(int).tolist()]
                invalid = [record_id for record_id in proposed_ids if record_id not in set(candidate_ids)]
                if invalid:
                    raise ValueError(
                        f"{dataset_name} mode={mode} seed={seed} round={round_idx} proposed ids outside candidate pool"
                    )
                round_rows = scan_round_axes(
                    dataset_name=dataset_name,
                    model=model,
                    mode=mode,
                    seed=seed,
                    round_idx=round_idx,
                    candidate_ids=candidate_ids,
                    proposed_ids=proposed_ids,
                    axes_by_record=axes_by_record,
                    min_candidate_axis_count=min_candidate_axis_count,
                    min_proposed_axis_count=min_proposed_axis_count,
                    alpha=alpha,
                    target_aliases=target_aliases,
                )
                detail_rows.extend(round_rows)
                round_count += 1
                if round_rows:
                    flagged_rows = [row for row in round_rows if bool(row["flagged"])]
                    target_rows = [
                        row
                        for row in round_rows
                        if bool(row["is_target_axis_alias_for_evaluation_only"])
                    ]
                    target_flag_rows = [row for row in target_rows if bool(row["flagged"])]
                    if flagged_rows:
                        trace_any_flag = True
                        first_any_rounds.append(round_idx)
                    if target_rows:
                        trace_target_best_rank = min(
                            trace_target_best_rank,
                            min(int(row["blind_rank"]) for row in target_rows),
                        )
                        target_axis_proposed_total += max(
                            int(row["proposed_axis_count"]) for row in target_rows
                        )
                    if target_flag_rows:
                        trace_target_flag = True
                        first_target_rounds.append(round_idx)
                        target_axis_flagged_proposed_total += max(
                            int(row["proposed_axis_count"]) for row in target_flag_rows
                        )
                    flagged_axis_count_total += len(flagged_rows)
                round_executed = seed_executed[seed_executed["round"].astype(int) == round_idx]
                prior_executed.update(
                    int(record_id) for record_id in round_executed[record_column].astype(int).tolist()
                )
            if trace_target_best_rank == 1_000_000:
                trace_target_best_rank = 0
            trace_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model,
                    "mode": mode,
                    "seed": int(seed),
                    "rounds": int(round_count),
                    "is_control": bool(mode in control_modes),
                    "is_targeted": bool(mode in target_modes),
                    "any_axis_flagged": bool(trace_any_flag),
                    "target_axis_flagged": bool(trace_target_flag),
                    "target_axis_best_rank": int(trace_target_best_rank),
                    "target_axis_top1": bool(trace_target_best_rank == 1),
                    "target_axis_top2": bool(trace_target_best_rank in {1, 2}),
                    "target_axis_top5": bool(1 <= trace_target_best_rank <= 5),
                    "first_any_flag_round": int(min(first_any_rounds)) if first_any_rounds else -1,
                    "first_target_flag_round": int(min(first_target_rounds)) if first_target_rounds else -1,
                    "flagged_axis_count_total": int(flagged_axis_count_total),
                    "target_axis_proposed_total": int(target_axis_proposed_total),
                    "target_axis_flagged_proposed_total": int(target_axis_flagged_proposed_total),
                }
            )
    summary = summarize_dataset(dataset_name, model, trace_rows)
    return detail_rows, trace_rows, summary


def summarize_dataset(dataset_name: str, model: str, trace_rows: list[dict[str, object]]) -> dict[str, object]:
    trace_frame = pd.DataFrame(trace_rows)
    if trace_frame.empty:
        raise ValueError(f"no trace rows for dataset={dataset_name}")
    controls = trace_frame[trace_frame["is_control"]].copy()
    targets = trace_frame[trace_frame["is_targeted"]].copy()
    target_flag_rounds = targets[targets["first_target_flag_round"] >= 0]["first_target_flag_round"]
    target_proposed_total = int(targets["target_axis_proposed_total"].sum()) if not targets.empty else 0
    target_flagged_proposed_total = (
        int(targets["target_axis_flagged_proposed_total"].sum()) if not targets.empty else 0
    )
    target_flagged_fraction = (
        target_flagged_proposed_total / target_proposed_total if target_proposed_total > 0 else 0.0
    )
    return {
        "dataset": dataset_name,
        "model": model,
        "trace_count": int(len(trace_frame)),
        "control_trace_count": int(len(controls)),
        "target_trace_count": int(len(targets)),
        "control_any_axis_flag_rate": float(controls["any_axis_flagged"].mean())
        if not controls.empty
        else float("nan"),
        "target_any_axis_flag_rate": float(targets["any_axis_flagged"].mean())
        if not targets.empty
        else float("nan"),
        "target_axis_flag_rate": float(targets["target_axis_flagged"].mean())
        if not targets.empty
        else float("nan"),
        "target_axis_top1_rate": float(targets["target_axis_top1"].mean())
        if not targets.empty
        else float("nan"),
        "target_axis_top2_rate": float(targets["target_axis_top2"].mean())
        if not targets.empty
        else float("nan"),
        "target_axis_top5_rate": float(targets["target_axis_top5"].mean())
        if not targets.empty
        else float("nan"),
        "control_target_axis_flag_rate": float(controls["target_axis_flagged"].mean())
        if not controls.empty
        else float("nan"),
        "target_median_first_flag_round": float(target_flag_rounds.median())
        if not target_flag_rounds.empty
        else float("nan"),
        "target_flagged_axis_allocation_fraction": float(target_flagged_fraction),
        "mean_flagged_axes_per_target_trace": float(targets["flagged_axis_count_total"].mean())
        if not targets.empty
        else float("nan"),
    }


def write_markdown(path: Path, alpha: float, summaries: list[dict[str, object]]) -> None:
    summary_frame = pd.DataFrame(summaries)
    lines = [
        "# B76 Blind Binomial Axis Scan",
        "",
        "## Hypothesis",
        "",
        "A candidate-pool normalized all-axis binomial scan can identify over-enriched scientific axes in proposed closed-loop batches without pre-naming the injected target axis during scanning.",
        "",
        "## Budget and Stop Conditions",
        "",
        "- Reuse B37 GFP, B38 materials and B39 CAMEO online traces.",
        "- Reconstruct each round's candidate pool from saved history, audit and executed-selection artifacts.",
        "- Stop after one Bonferroni-corrected alpha threshold.",
        "",
        f"Alpha: {alpha:.6g}",
        "",
        "## Summary",
        "",
        summary_frame.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "The target axis is used only for after-the-fact evaluation. Low control any-axis flagging with high targeted-axis recovery supports a blind warning signal; high control flagging marks a multiple-axis boundary rather than a deployable detector.",
        "",
        "## Non-Claims",
        "",
        "- Not a complete detector or complete defense.",
        "- Not record-level correction.",
        "- Not evidence that public archives are corrupt.",
        "- Not a faithful reproduction of original closed-loop controllers.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B76 blind binomial all-axis scan.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b76_blind_binomial_axis_scan")
    if not isinstance(cfg["alpha"], int | float):
        raise TypeError("alpha must be numeric")
    alpha = float(cfg["alpha"])
    datasets = require_dataset_list(cfg)

    detail_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_detail, dataset_traces, dataset_summary = rows_for_dataset(dataset, alpha)
        detail_rows.extend(dataset_detail)
        trace_rows.extend(dataset_traces)
        summaries.append(dataset_summary)

    output_detail = Path(str(cfg["output_detail_csv"]))
    output_summary = Path(str(cfg["output_summary_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    for path in [output_detail, output_summary, output_json, output_md]:
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(output_detail, index=False)
    pd.DataFrame(summaries).to_csv(output_summary, index=False)
    payload = {
        "alpha": alpha,
        "config_path": str(config_path),
        "summaries": summaries,
        "trace_rows": trace_rows,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, alpha, summaries)
    print(output_detail)
    print(output_summary)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
