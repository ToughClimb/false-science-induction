#!/usr/bin/env python
from __future__ import annotations

import json
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

from false_science.config import (  # noqa: E402
    load_json_config,
    parse_config_arg,
    require_choice,
    require_keys,
    require_nested,
)
from false_science.models import fit_torch_mlp_predictor  # noqa: E402
from scripts.b70_bear_physical_sdl_replay import (  # noqa: E402
    MODES,
    build_recorded_labels,
    choose_target_axis,
    label_multiset_equal,
    read_bear_campaign,
)


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
    "device",
    "mlp",
    "mc_dropout_passes",
    "acquisition_beta",
    "output_trace_csv",
    "output_summary_csv",
    "output_json",
    "output_md",
]

REQUIRED_MLP_KEYS = [
    "epochs",
    "hidden_dim",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "eval_batch_size",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def validate_config(cfg: dict[str, object]) -> None:
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b78_bear_neural_sdl_replay")
    mlp_cfg = require_nested(cfg, "mlp", "b78_bear_neural_sdl_replay")
    require_keys(mlp_cfg, REQUIRED_MLP_KEYS, "b78_bear_neural_sdl_replay.mlp")
    require_choice(cfg, "device", {"cpu", "cuda"}, "b78_bear_neural_sdl_replay")
    if not isinstance(cfg["mc_dropout_passes"], int):
        raise TypeError("mc_dropout_passes must be an integer")
    if int(cfg["mc_dropout_passes"]) < 2:
        raise ValueError("mc_dropout_passes must be at least 2")
    for key in ["history_size", "candidate_size", "swap_count", "rounds", "batch_size"]:
        if not isinstance(cfg[key], int):
            raise TypeError(f"{key} must be an integer")
    if not isinstance(cfg["acquisition_beta"], int | float):
        raise TypeError("acquisition_beta must be numeric")
    require_string_list(cfg, "feature_columns", "b78_bear_neural_sdl_replay")
    require_string_list(cfg, "target_axis_candidates", "b78_bear_neural_sdl_replay")


def axis_mask(frame: pd.DataFrame, axis: str, value: float) -> np.ndarray:
    return frame[axis].to_numpy(dtype=float) == float(value)


def neural_ucb_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_candidate: np.ndarray,
    seed: int,
    cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mlp_cfg = cfg["mlp"]
    if not isinstance(mlp_cfg, dict):
        raise TypeError("mlp must be a JSON object")
    predictor = fit_torch_mlp_predictor(
        x_train=x_train,
        y_train=y_train,
        seed=int(seed),
        epochs=int(mlp_cfg["epochs"]),
        hidden_dim=int(mlp_cfg["hidden_dim"]),
        batch_size=int(mlp_cfg["batch_size"]),
        learning_rate=float(mlp_cfg["learning_rate"]),
        weight_decay=float(mlp_cfg["weight_decay"]),
        dropout=float(mlp_cfg["dropout"]),
        device=str(cfg["device"]),
        eval_batch_size=int(mlp_cfg["eval_batch_size"]),
    )
    mean, std = predictor.predict_mc_dropout(
        x_candidate,
        passes=int(cfg["mc_dropout_passes"]),
        seed=int(seed) + 700_001,
    )
    score = mean + float(cfg["acquisition_beta"]) * std
    return mean, std, score


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
    base_candidate_ids = np.arange(
        history_size,
        min(history_size + candidate_size, len(frame)),
        dtype=int,
    )
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
                mean, std, score = neural_ucb_scores(
                    features[train_ids],
                    train_y,
                    features[candidate_ids],
                    seed=seed * 1000 + round_idx,
                    cfg=cfg,
                )
                order = np.argsort(-score)
                batch_ids = candidate_ids[order[: int(cfg["batch_size"])]]
                selected.extend(batch_ids.tolist())
                target_batch = target[batch_ids]
                target_candidate = target[candidate_ids]
                trace_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "round": round_idx,
                        "pairs_used": int(pairs_used),
                        "label_multiset_preserved": bool(multiset_preserved),
                        "batch_target_count": int(target_batch.sum()),
                        "batch_target_fraction": float(target_batch.mean()),
                        "batch_true_mean": float(true_y[batch_ids].mean()),
                        "batch_target_true_mean": float(true_y[batch_ids[target_batch]].mean())
                        if bool(target_batch.any())
                        else float("nan"),
                        "candidate_target_count": int(target_candidate.sum()),
                        "mean_prediction_target": float(mean[target_candidate].mean())
                        if bool(target_candidate.any())
                        else float("nan"),
                        "mean_prediction_non_target": float(mean[~target_candidate].mean())
                        if bool((~target_candidate).any())
                        else float("nan"),
                        "mean_uncertainty_target": float(std[target_candidate].mean())
                        if bool(target_candidate.any())
                        else float("nan"),
                        "mean_uncertainty_non_target": float(std[~target_candidate].mean())
                        if bool((~target_candidate).any())
                        else float("nan"),
                    }
                )
                keep = np.ones(len(candidate_ids), dtype=bool)
                keep[order[: int(cfg["batch_size"])]] = False
                candidate_ids = candidate_ids[keep]
                train_ids = np.concatenate([train_ids, batch_ids]).astype(int)
                train_y = np.concatenate([train_y, true_y[batch_ids]]).astype(float)
            selected_ids = np.array(selected, dtype=int)
            selected_target = target[selected_ids]
            summary_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "pairs_used": int(pairs_used),
                    "label_multiset_preserved": bool(multiset_preserved),
                    "final_target_count": int(selected_target.sum()),
                    "final_target_fraction": float(selected_target.mean()),
                    "selected_true_mean": float(true_y[selected_ids].mean()),
                    "selected_target_true_mean": float(
                        true_y[selected_ids[selected_target]].mean()
                    )
                    if bool(selected_target.any())
                    else float("nan"),
                }
            )
    return trace_rows, summary_rows


def paired_summary(summary: pd.DataFrame) -> dict[str, object]:
    aggregate = summary.groupby("mode", as_index=False).agg(
        final_target_count=("final_target_count", "mean"),
        selected_true_mean=("selected_true_mean", "mean"),
        selected_target_true_mean=("selected_target_true_mean", "mean"),
        label_multiset_preserved=("label_multiset_preserved", "mean"),
    )
    mode_rows = {str(row.mode): row for row in aggregate.itertuples(index=False)}
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
        "# B78 BEAR Neural SDL Replay Result",
        "",
        "## Status",
        "",
        "This is a neural surrogate replay on a public autonomous physical experimentation archive. It is not an audit claim that the BEAR archive is corrupt and not a faithful reproduction of the BEAR controller.",
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
            "Supported if positive: controlled real-record/real-measurement relinking can redirect a neural closed-loop surrogate on an external autonomous physical-experiment stream.",
            "",
            "Not supported: natural BEAR corruption, wrong BEAR conclusions, faithful controller reproduction, universal vulnerability, universal stealth or record-level correction.",
            "",
            "## Config",
            "",
            f"- Tag: `{cfg['tag']}`",
            f"- Swap count: `{cfg['swap_count']}`",
            f"- Seeds: `{cfg['seeds']}`",
            f"- Rounds x batch size: `{cfg['rounds']} x {cfg['batch_size']}`",
            f"- Model: `mlp_mc_dropout_ucb`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Run B78 BEAR neural SDL retrospective replay.")
    cfg = load_json_config(config_path)
    validate_config(cfg)
    feature_columns = [str(column) for column in cfg["feature_columns"]]
    axis_candidates = [str(column) for column in cfg["target_axis_candidates"]]
    frame = read_bear_campaign(
        Path(str(cfg["data_path"])),
        str(cfg["target_column"]),
        feature_columns,
    )
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
        "model": "mlp_mc_dropout_ucb",
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
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, cfg, axis, value, scan, payload["paired_summary"])
    print(output_trace)
    print(output_summary)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
