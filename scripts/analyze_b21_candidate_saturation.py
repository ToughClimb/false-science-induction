#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys


REQUIRED_CONFIG_KEYS = [
    "run_dir",
    "metrics_file",
    "output_csv",
    "output_md",
    "models",
    "mode",
    "start_round",
    "mid_round",
    "end_round",
]

REQUIRED_COLUMNS = [
    "seed",
    "model",
    "mode",
    "round",
    "candidate_triggered_target_count",
    "batch_triggered_target_count",
    "cumulative_triggered_target_count",
]


def require_string_list(cfg: dict[str, object], key: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{key} must contain only strings")
    return [str(item) for item in value]


def require_int(cfg: dict[str, object], key: str) -> int:
    value = cfg[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return int(value)


def load_rounds(cfg: dict[str, object]) -> pd.DataFrame:
    path = Path(str(cfg["run_dir"])) / str(cfg["metrics_file"])
    if not path.is_file():
        raise FileNotFoundError(f"round metrics not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def value_at_round(
    frame: pd.DataFrame,
    model: str,
    mode: str,
    round_idx: int,
    value_column: str,
) -> pd.Series:
    subset = frame[
        (frame["model"] == model)
        & (frame["mode"] == mode)
        & (frame["round"] == round_idx)
    ][["seed", value_column]]
    if subset.empty:
        raise ValueError(f"no rows for model={model} mode={mode} round={round_idx}")
    if subset["seed"].duplicated().any():
        raise ValueError(f"duplicate seed rows for model={model} mode={mode} round={round_idx}")
    return subset.set_index("seed")[value_column].sort_index()


def summarize_model(
    frame: pd.DataFrame,
    model: str,
    mode: str,
    start_round: int,
    mid_round: int,
    end_round: int,
) -> dict[str, object]:
    start_remaining = value_at_round(
        frame,
        model,
        mode,
        start_round,
        "candidate_triggered_target_count",
    )
    mid_remaining = value_at_round(
        frame,
        model,
        mode,
        mid_round,
        "candidate_triggered_target_count",
    )
    end_remaining = value_at_round(
        frame,
        model,
        mode,
        end_round,
        "candidate_triggered_target_count",
    )
    mid_cumulative = value_at_round(
        frame,
        model,
        mode,
        mid_round,
        "cumulative_triggered_target_count",
    )
    end_cumulative = value_at_round(
        frame,
        model,
        mode,
        end_round,
        "cumulative_triggered_target_count",
    )
    seeds = set(start_remaining.index.tolist())
    if seeds != set(mid_remaining.index.tolist()) or seeds != set(end_remaining.index.tolist()):
        raise ValueError(f"seed mismatch for {model}")
    post_mid_gain = end_cumulative - mid_cumulative
    end_remaining_fraction = end_remaining / start_remaining
    return {
        "model": model,
        "mode": mode,
        "n_seeds": int(len(start_remaining)),
        "start_round": start_round,
        "mid_round": mid_round,
        "end_round": end_round,
        "start_remaining_mean": float(start_remaining.mean()),
        "mid_remaining_mean": float(mid_remaining.mean()),
        "end_remaining_mean": float(end_remaining.mean()),
        "end_remaining_fraction_mean": float(end_remaining_fraction.mean()),
        "end_remaining_fraction_min": float(end_remaining_fraction.min()),
        "end_remaining_fraction_max": float(end_remaining_fraction.max()),
        "mid_cumulative_mean": float(mid_cumulative.mean()),
        "end_cumulative_mean": float(end_cumulative.mean()),
        "post_mid_gain_mean": float(post_mid_gain.mean()),
        "post_mid_gain_max": float(post_mid_gain.max()),
        "seeds_with_post_mid_gain": int((post_mid_gain > 0).sum()),
    }


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# B21 Candidate Saturation Analysis",
        "",
        "This analysis reads existing B21 round metrics and checks whether late-loop GFP saturation is explained by exhausting the triggered-target candidate pool.",
        "",
        "| Model | Seeds | Start remaining | End remaining | End remaining fraction | Post-mid gain | Seeds with post-mid gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {n_seeds} | {start_remaining_mean:.1f} | {end_remaining_mean:.1f} | {end_remaining_fraction_mean:.3f} | {post_mid_gain_mean:.1f} | {seeds_with_post_mid_gain} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: high end-of-loop remaining fractions with near-zero post-mid gains indicate saturation is not simply candidate-pool exhaustion. The loop leaves most triggered-target candidates unselected after feedback, consistent with attenuation or rank correction after early false allocation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B21 candidate saturation.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b21_candidate_saturation")
    frame = load_rounds(cfg)
    models = require_string_list(cfg, "models")
    mode = str(cfg["mode"])
    start_round = require_int(cfg, "start_round")
    mid_round = require_int(cfg, "mid_round")
    end_round = require_int(cfg, "end_round")
    rows = [
        summarize_model(frame, model, mode, start_round, mid_round, end_round)
        for model in models
    ]
    output_csv = Path(str(cfg["output_csv"]))
    output_md = Path(str(cfg["output_md"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    write_markdown(output_md, rows)
    print(str(output_csv))
    print(str(output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
