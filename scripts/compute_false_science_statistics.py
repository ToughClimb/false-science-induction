#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys


REQUIRED_CONFIG_KEYS = [
    "output_json",
    "output_csv",
    "bootstrap_iterations",
    "bootstrap_seed",
    "main_effects",
]

REQUIRED_EFFECT_KEYS = [
    "name",
    "run_dir",
    "metrics_file",
    "model",
    "mode_a",
    "mode_b",
    "value_column",
    "comparison",
]


def require_int(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def require_effects(cfg: dict[str, object]) -> list[dict[str, object]]:
    effects = cfg["main_effects"]
    if not isinstance(effects, list):
        raise TypeError("b15_statistics.main_effects must be a JSON list")
    for index, effect in enumerate(effects):
        if not isinstance(effect, dict):
            raise TypeError(f"main_effects[{index}] must be a JSON object")
        require_keys(effect, REQUIRED_EFFECT_KEYS, f"main_effects[{index}]")
    return effects


def load_rounds(effect: dict[str, object]) -> pd.DataFrame:
    path = Path(str(effect["run_dir"])) / str(effect["metrics_file"])
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")
    rounds = pd.read_csv(path)
    required_columns = ["seed", "model", "mode", "round", str(effect["value_column"])]
    missing = [column for column in required_columns if column not in rounds.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return rounds


def values_at_round(
    rounds: pd.DataFrame,
    model: str,
    mode: str,
    value_column: str,
    round_idx: int,
) -> pd.Series:
    subset = rounds[
        (rounds["model"] == model)
        & (rounds["mode"] == mode)
        & (rounds["round"] == round_idx)
    ][["seed", value_column]]
    if subset.empty:
        raise ValueError(f"no rows for model={model} mode={mode} round={round_idx}")
    duplicated = subset["seed"].duplicated().any()
    if duplicated:
        raise ValueError(f"duplicate seed rows for model={model} mode={mode} round={round_idx}")
    return subset.set_index("seed")[value_column].sort_index()


def paired_difference(effect: dict[str, object], rounds: pd.DataFrame) -> np.ndarray:
    model = str(effect["model"])
    mode_a = str(effect["mode_a"])
    mode_b = str(effect["mode_b"])
    value_column = str(effect["value_column"])
    comparison = str(effect["comparison"])
    if comparison == "paired_seed_difference":
        round_idx = require_int(effect["round"], f"{effect['name']}.round")
        series_a = values_at_round(rounds, model, mode_a, value_column, round_idx)
        series_b = values_at_round(rounds, model, mode_b, value_column, round_idx)
    elif comparison == "paired_seed_post_round_gain_difference":
        round_a = require_int(effect["round_a"], f"{effect['name']}.round_a")
        round_b = require_int(effect["round_b"], f"{effect['name']}.round_b")
        series_a = values_at_round(rounds, model, mode_a, value_column, round_a) - values_at_round(
            rounds,
            model,
            mode_a,
            value_column,
            round_b,
        )
        series_b = values_at_round(rounds, model, mode_b, value_column, round_a) - values_at_round(
            rounds,
            model,
            mode_b,
            value_column,
            round_b,
        )
    else:
        raise ValueError(f"unsupported comparison: {comparison}")
    seeds_a = set(series_a.index.tolist())
    seeds_b = set(series_b.index.tolist())
    if seeds_a != seeds_b:
        raise ValueError(f"seed mismatch for effect {effect['name']}")
    return (series_a - series_b).to_numpy(dtype=float)


def bootstrap_mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        raise ValueError("cannot bootstrap empty values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = draws.mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def sign_flip_p_value(values: np.ndarray) -> float:
    if len(values) == 0:
        raise ValueError("cannot test empty values")
    observed = abs(float(np.mean(values)))
    n = len(values)
    means: list[float] = []
    for mask in range(2**n):
        signs = np.array([1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n)])
        means.append(abs(float(np.mean(values * signs))))
    extreme = sum(mean >= observed for mean in means)
    return float(extreme / len(means))


def main() -> int:
    config_path = parse_config_arg("Compute false-science effect statistics.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b15_statistics")
    iterations = require_int(cfg["bootstrap_iterations"], "b15_statistics.bootstrap_iterations")
    bootstrap_seed = require_int(cfg["bootstrap_seed"], "b15_statistics.bootstrap_seed")
    effects = require_effects(cfg)

    rows: list[dict[str, object]] = []
    for effect_index, effect in enumerate(effects):
        rounds = load_rounds(effect)
        differences = paired_difference(effect, rounds)
        ci_low, ci_high = bootstrap_mean_ci(
            differences,
            iterations=iterations,
            seed=bootstrap_seed + effect_index,
        )
        rows.append(
            {
                "name": str(effect["name"]),
                "run_dir": str(effect["run_dir"]),
                "model": str(effect["model"]),
                "comparison": str(effect["comparison"]),
                "n_seeds": int(len(differences)),
                "differences": json.dumps(differences.tolist()),
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "min_difference": float(np.min(differences)),
                "max_difference": float(np.max(differences)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "sign_flip_p_two_sided": sign_flip_p_value(differences),
                "all_seed_differences_positive": bool(np.all(differences > 0.0)),
            }
        )

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "bootstrap_iterations": iterations,
                "bootstrap_seed": bootstrap_seed,
                "effects": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(str(output_csv))
    print(str(output_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
