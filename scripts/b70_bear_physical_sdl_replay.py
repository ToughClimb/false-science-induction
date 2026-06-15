#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "data_path",
    "tag",
    "target_column",
    "feature_columns",
    "target_axis_candidates",
    "target_quantile",
    "min_history_target_count",
    "min_candidate_target_count",
    "history_size",
    "candidate_size",
    "swap_count",
    "seeds",
    "rounds",
    "batch_size",
    "n_estimators",
    "acquisition_beta",
    "output_trace_csv",
    "output_summary_csv",
    "output_json",
    "output_md",
]

MODES = ["clean", "random_swap", "targeted_relink"]


def read_bear_campaign(path: Path, target_column: str, feature_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"BEAR campaign CSV not found: {path}")
    frame = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
    required = [
        "ADTS_ID",
        "Valid",
        "TimePrintStarted",
        target_column,
        *feature_columns,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    for column in ["ADTS_ID", "Valid", target_column, *feature_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["TimePrintStarted_dt"] = pd.to_datetime(frame["TimePrintStarted"], errors="coerce")
    clean = frame[(frame["Valid"].notna()) & (frame["Valid"] != 0)].copy()
    clean = clean.dropna(subset=[target_column, "TimePrintStarted_dt", *feature_columns])
    clean = clean.sort_values(["TimePrintStarted_dt", "ADTS_ID"]).reset_index(drop=True)
    clean["record_id"] = np.arange(len(clean), dtype=int)
    return clean


def standardize(train: np.ndarray, eval_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, (eval_values - mean) / scale


def axis_mask(frame: pd.DataFrame, axis: str, value: float) -> np.ndarray:
    return frame[axis].to_numpy(dtype=float) == float(value)


def choose_target_axis(
    frame: pd.DataFrame,
    target_column: str,
    axis_candidates: list[str],
    target_quantile: float,
    min_history_target_count: int,
    min_candidate_target_count: int,
    history_size: int,
) -> tuple[str, float, pd.DataFrame]:
    history = frame.iloc[:history_size]
    candidate = frame.iloc[history_size:]
    cutoff = float(frame[target_column].quantile(float(target_quantile)))
    rows: list[dict[str, object]] = []
    global_mean = float(frame[target_column].mean())
    for axis in axis_candidates:
        for value, axis_frame in frame.groupby(axis):
            history_count = int((history[axis] == value).sum())
            candidate_count = int((candidate[axis] == value).sum())
            if history_count < min_history_target_count:
                continue
            if candidate_count < min_candidate_target_count:
                continue
            axis_mean = float(axis_frame[target_column].mean())
            if axis_mean > cutoff:
                continue
            rows.append(
                {
                    "axis": axis,
                    "value": float(value),
                    "count": int(len(axis_frame)),
                    "history_count": history_count,
                    "candidate_count": candidate_count,
                    "axis_mean": axis_mean,
                    "global_mean_minus_axis_mean": float(global_mean - axis_mean),
                }
            )
    scan = pd.DataFrame(rows)
    if scan.empty:
        raise ValueError("no BEAR target axis satisfies the configured support and low-mean gate")
    scan = scan.sort_values(
        ["global_mean_minus_axis_mean", "candidate_count", "history_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top = scan.iloc[0]
    return str(top["axis"]), float(top["value"]), scan


def label_multiset_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(np.sort(left.astype(float)), np.sort(right.astype(float))))


def build_recorded_labels(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    target_history_ids: np.ndarray,
    donor_history_ids: np.ndarray,
    mode: str,
    seed: int,
    swap_count: int,
) -> tuple[np.ndarray, int]:
    recorded = true_y[history_ids].astype(float).copy()
    history_pos = {int(record_id): index for index, record_id in enumerate(history_ids)}
    n_pairs = min(int(swap_count), len(target_history_ids), len(donor_history_ids))
    if mode == "clean":
        return recorded, 0
    if mode == "targeted_relink":
        target_order = target_history_ids[np.argsort(true_y[target_history_ids])][:n_pairs]
        donor_order = donor_history_ids[np.argsort(-true_y[donor_history_ids])][:n_pairs]
        for target_id, donor_id in zip(target_order, donor_order, strict=True):
            recorded[history_pos[int(target_id)]] = float(true_y[int(donor_id)])
            recorded[history_pos[int(donor_id)]] = float(true_y[int(target_id)])
        return recorded, n_pairs
    if mode == "random_swap":
        rng = np.random.default_rng(seed + 17001)
        chosen = rng.choice(history_ids, size=2 * n_pairs, replace=False)
        left = chosen[:n_pairs]
        right = chosen[n_pairs:]
        for left_id, right_id in zip(left, right, strict=True):
            left_pos = history_pos[int(left_id)]
            right_pos = history_pos[int(right_id)]
            recorded[left_pos], recorded[right_pos] = recorded[right_pos], recorded[left_pos]
        return recorded, n_pairs
    raise ValueError(f"unknown mode: {mode}")


def rf_ucb_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_candidate: np.ndarray,
    seed: int,
    n_estimators: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    forest = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        min_samples_leaf=2,
        n_jobs=1,
    )
    x_train_scaled, x_candidate_scaled = standardize(x_train, x_candidate)
    forest.fit(x_train_scaled, y_train)
    tree_predictions = np.vstack(
        [tree.predict(x_candidate_scaled) for tree in forest.estimators_]
    )
    mean = tree_predictions.mean(axis=0)
    std = tree_predictions.std(axis=0)
    return mean, mean + float(beta) * std


def run_replay(
    frame: pd.DataFrame,
    cfg: dict[str, object],
    axis: str,
    value: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target_column = str(cfg["target_column"])
    feature_columns = [str(column) for column in cfg["feature_columns"]]
    true_y = frame[target_column].to_numpy(dtype=float)
    features = frame[feature_columns].to_numpy(dtype=float)
    target = axis_mask(frame, axis, value)
    history_size = int(cfg["history_size"])
    candidate_size = int(cfg["candidate_size"])
    base_history_ids = np.arange(history_size, dtype=int)
    base_candidate_ids = np.arange(history_size, min(history_size + candidate_size, len(frame)), dtype=int)
    target_history_ids = base_history_ids[target[base_history_ids]]
    donor_history_ids = base_history_ids[(~target[base_history_ids])]
    donor_history_ids = donor_history_ids[
        true_y[donor_history_ids] >= np.quantile(true_y[base_history_ids], 0.85)
    ]
    if len(target_history_ids) < int(cfg["swap_count"]):
        raise ValueError("not enough target history records for configured swap_count")
    if len(donor_history_ids) < int(cfg["swap_count"]):
        raise ValueError("not enough donor history records for configured swap_count")
    trace_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed in [int(seed) for seed in cfg["seeds"]]:
        for mode in MODES:
            train_ids = base_history_ids.copy()
            candidate_ids = base_candidate_ids.copy()
            recorded, pairs_used = build_recorded_labels(
                true_y=true_y,
                history_ids=base_history_ids,
                target_history_ids=target_history_ids,
                donor_history_ids=donor_history_ids,
                mode=mode,
                seed=seed,
                swap_count=int(cfg["swap_count"]),
            )
            multiset_preserved = label_multiset_equal(true_y[base_history_ids], recorded)
            selected: list[int] = []
            train_y = recorded.copy()
            for round_idx in range(int(cfg["rounds"])):
                mean, score = rf_ucb_scores(
                    features[train_ids],
                    train_y,
                    features[candidate_ids],
                    seed=seed * 1000 + round_idx,
                    n_estimators=int(cfg["n_estimators"]),
                    beta=float(cfg["acquisition_beta"]),
                )
                order = np.argsort(-score)
                batch_ids = candidate_ids[order[: int(cfg["batch_size"])]]
                selected.extend(batch_ids.tolist())
                trace_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "round": round_idx,
                        "pairs_used": int(pairs_used),
                        "label_multiset_preserved": bool(multiset_preserved),
                        "batch_target_count": int(target[batch_ids].sum()),
                        "batch_target_fraction": float(target[batch_ids].mean()),
                        "batch_true_mean": float(true_y[batch_ids].mean()),
                        "batch_target_true_mean": float(true_y[batch_ids[target[batch_ids]]].mean())
                        if bool(target[batch_ids].any())
                        else float("nan"),
                        "candidate_target_count": int(target[candidate_ids].sum()),
                        "mean_score_target": float(mean[target[candidate_ids]].mean())
                        if bool(target[candidate_ids].any())
                        else float("nan"),
                        "mean_score_non_target": float(mean[~target[candidate_ids]].mean())
                        if bool((~target[candidate_ids]).any())
                        else float("nan"),
                    }
                )
                keep = np.ones(len(candidate_ids), dtype=bool)
                keep[order[: int(cfg["batch_size"])]] = False
                candidate_ids = candidate_ids[keep]
                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, true_y[batch_ids]]).astype(float)
            selected_ids = np.array(selected, dtype=int)
            summary_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "pairs_used": int(pairs_used),
                    "label_multiset_preserved": bool(multiset_preserved),
                    "final_target_count": int(target[selected_ids].sum()),
                    "final_target_fraction": float(target[selected_ids].mean()),
                    "selected_true_mean": float(true_y[selected_ids].mean()),
                    "selected_target_true_mean": float(true_y[selected_ids[target[selected_ids]]].mean())
                    if bool(target[selected_ids].any())
                    else float("nan"),
                }
            )
    return trace_rows, summary_rows


def paired_summary(summary: pd.DataFrame) -> dict[str, object]:
    mode_rows = {
        str(row.mode): row
        for row in summary.groupby("mode", as_index=False).agg(
            final_target_count=("final_target_count", "mean"),
            selected_true_mean=("selected_true_mean", "mean"),
            selected_target_true_mean=("selected_target_true_mean", "mean"),
            label_multiset_preserved=("label_multiset_preserved", "mean"),
        ).itertuples(index=False)
    }
    targeted = summary[summary["mode"] == "targeted_relink"].sort_values("seed")
    random = summary[summary["mode"] == "random_swap"].sort_values("seed")
    clean = summary[summary["mode"] == "clean"].sort_values("seed")
    tr_minus_random = (
        targeted["final_target_count"].to_numpy(dtype=float)
        - random["final_target_count"].to_numpy(dtype=float)
    )
    tr_minus_clean = (
        targeted["final_target_count"].to_numpy(dtype=float)
        - clean["final_target_count"].to_numpy(dtype=float)
    )
    return {
        "mode_means": {
            mode: {
                "final_target_count": float(row.final_target_count),
                "selected_true_mean": float(row.selected_true_mean),
                "selected_target_true_mean": float(row.selected_target_true_mean),
                "label_multiset_preserved": float(row.label_multiset_preserved),
            }
            for mode, row in mode_rows.items()
        },
        "targeted_minus_random_mean": float(tr_minus_random.mean()),
        "targeted_minus_clean_mean": float(tr_minus_clean.mean()),
        "targeted_minus_random_positive_seeds": int((tr_minus_random > 0).sum()),
        "targeted_minus_random_negative_seeds": int((tr_minus_random < 0).sum()),
        "targeted_minus_random_tied_seeds": int((tr_minus_random == 0).sum()),
        "targeted_minus_random_seed_differences": tr_minus_random.astype(int).tolist(),
    }


def write_markdown(
    path: Path,
    cfg: dict[str, object],
    axis: str,
    value: float,
    scan: pd.DataFrame,
    summary_payload: dict[str, object],
) -> None:
    mode_means = summary_payload["mode_means"]
    lines = [
        "# B70 BEAR Physical SDL Replay Result",
        "",
        "## Status",
        "",
        "This is a retrospective surrogate replay on a public autonomous physical experimentation archive. It is not an audit claim that the BEAR archive is corrupt and not a faithful reproduction of the BEAR controller.",
        "",
        "## Selected Axis",
        "",
        f"- Axis: `{axis}`",
        f"- Value: `{value:g}`",
        "",
        "Top target-axis scan rows:",
        "",
        scan.head(8).to_markdown(index=False),
        "",
        "## Replay Summary",
        "",
        "| Mode | Final target acquisitions | Selected true mean | Target selected true mean | Label multiset preserved |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = mode_means[mode]
        lines.append(
            "| {mode} | {final_target_count:.2f} | {selected_true_mean:.3f} | {selected_target_true_mean:.3f} | {label_multiset_preserved:.3f} |".format(
                mode=mode,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Paired Seed Contrast",
            "",
            f"- Targeted minus random mean final target acquisitions: {summary_payload['targeted_minus_random_mean']:.2f}",
            f"- Targeted minus clean mean final target acquisitions: {summary_payload['targeted_minus_clean_mean']:.2f}",
            f"- Positive / negative / tied seeds vs random: {summary_payload['targeted_minus_random_positive_seeds']} / {summary_payload['targeted_minus_random_negative_seeds']} / {summary_payload['targeted_minus_random_tied_seeds']}",
            f"- Seed differences vs random: `{summary_payload['targeted_minus_random_seed_differences']}`",
            "",
            "## Claim Boundary",
            "",
            "Supported if used: controlled real-record/real-measurement relinking on an external autonomous physical experiment stream can be evaluated under the same binding-to-budget protocol as the primary paper.",
            "",
            "Not supported: natural BEAR corruption, wrong BEAR conclusions, faithful controller reproduction, universal vulnerability, universal stealth or record-level correction.",
            "",
            "## Config",
            "",
            f"- Tag: `{cfg['tag']}`",
            f"- Swap count: `{cfg['swap_count']}`",
            f"- Seeds: `{cfg['seeds']}`",
            f"- Rounds x batch size: `{cfg['rounds']} x {cfg['batch_size']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Run B70 BEAR physical SDL retrospective replay.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b70_bear_physical_sdl_replay")
    feature_columns = [str(column) for column in cfg["feature_columns"]]
    axis_candidates = [str(column) for column in cfg["target_axis_candidates"]]
    frame = read_bear_campaign(Path(str(cfg["data_path"])), str(cfg["target_column"]), feature_columns)
    axis, value, scan = choose_target_axis(
        frame=frame,
        target_column=str(cfg["target_column"]),
        axis_candidates=axis_candidates,
        target_quantile=float(cfg["target_quantile"]),
        min_history_target_count=int(cfg["min_history_target_count"]),
        min_candidate_target_count=int(cfg["min_candidate_target_count"]),
        history_size=int(cfg["history_size"]),
    )
    trace_rows, summary_rows = run_replay(frame, cfg, axis, value)
    trace = pd.DataFrame(trace_rows)
    summary = pd.DataFrame(summary_rows)
    payload = {
        "axis": axis,
        "value": value,
        "scan_rows": scan.to_dict(orient="records"),
        "paired_summary": paired_summary(summary),
    }

    output_trace = Path(str(cfg["output_trace_csv"]))
    output_summary = Path(str(cfg["output_summary_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    for path in [output_trace, output_summary, output_json, output_md]:
        path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_trace, index=False)
    summary.to_csv(output_summary, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_markdown(output_md, cfg, axis, value, scan, payload["paired_summary"])
    print(output_trace)
    print(output_summary)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
