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
    "run_dirs",
    "output_csv",
    "metadata_file",
    "summary_file",
    "swap_pairs_file",
    "required_summary_columns",
    "output_columns",
]

REQUIRED_PAIR_COLUMNS = [
    "seed",
    "target_true_label",
    "donor_true_label",
    "target_recorded_label_after_swap",
    "donor_recorded_label_after_swap",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def read_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a JSON object: {path}")
    require_keys(
        metadata,
        ["swap_count", "target_tag", "target_count", "data_sha256", "config"],
        str(path),
    )
    config = metadata["config"]
    if not isinstance(config, dict):
        raise TypeError(f"{path}.config must be a JSON object")
    trigger = config["trigger"]
    if not isinstance(trigger, dict):
        raise TypeError(f"{path}.config.trigger must be a JSON object")
    require_keys(
        trigger,
        [
            "history_target_trigger_count",
            "distributed_dim_count",
            "distributed_scale",
            "distributed_seed",
        ],
        f"{path}.config.trigger",
    )
    return metadata


def read_summary(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")
    summary = pd.read_csv(path)
    missing = [column for column in required_columns if column not in summary.columns]
    if missing:
        raise KeyError(f"{path} missing summary columns: {', '.join(missing)}")
    return summary


def read_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"swap pairs file not found: {path}")
    pairs = pd.read_csv(path)
    missing = [column for column in REQUIRED_PAIR_COLUMNS if column not in pairs.columns]
    if missing:
        raise KeyError(f"{path} missing pair columns: {', '.join(missing)}")
    return pairs


def audit_label_multiset(path: Path, pairs: pd.DataFrame) -> dict[str, object]:
    true_values = np.sort(
        np.concatenate(
            [
                pairs["target_true_label"].to_numpy(dtype=float),
                pairs["donor_true_label"].to_numpy(dtype=float),
            ]
        )
    )
    recorded_values = np.sort(
        np.concatenate(
            [
                pairs["target_recorded_label_after_swap"].to_numpy(dtype=float),
                pairs["donor_recorded_label_after_swap"].to_numpy(dtype=float),
            ]
        )
    )
    preserved = bool(np.allclose(true_values, recorded_values, rtol=0.0, atol=1e-12))
    if not preserved:
        raise ValueError(f"label multiset is not preserved: {path}")
    pairs_per_seed = pairs.groupby("seed").size()
    return {
        "label_multiset_preserved": preserved,
        "pairs_per_seed_min": int(pairs_per_seed.min()),
        "pairs_per_seed_max": int(pairs_per_seed.max()),
    }


def main() -> int:
    config_path = parse_config_arg("Aggregate triggered dose-response summaries.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "aggregate_triggered_dose_response")
    run_dirs = require_string_list(cfg, "run_dirs", "aggregate_triggered_dose_response")
    required_summary_columns = require_string_list(
        cfg,
        "required_summary_columns",
        "aggregate_triggered_dose_response",
    )
    output_columns = require_string_list(
        cfg,
        "output_columns",
        "aggregate_triggered_dose_response",
    )
    metadata_file = str(cfg["metadata_file"])
    summary_file = str(cfg["summary_file"])
    swap_pairs_file = str(cfg["swap_pairs_file"])
    output_csv = Path(str(cfg["output_csv"]))

    frames: list[pd.DataFrame] = []
    for run_dir_text in run_dirs:
        run_dir = Path(run_dir_text)
        metadata = read_metadata(run_dir / metadata_file)
        summary = read_summary(run_dir / summary_file, required_summary_columns)
        pairs = read_pairs(run_dir / swap_pairs_file)
        audit = audit_label_multiset(run_dir / swap_pairs_file, pairs)
        trigger = metadata["config"]["trigger"]
        summary = summary.copy()
        summary.insert(0, "run_dir", str(run_dir))
        summary.insert(1, "swap_count", int(metadata["swap_count"]))
        summary.insert(2, "history_target_trigger_count", int(trigger["history_target_trigger_count"]))
        summary.insert(3, "trigger_distributed_dim_count", int(trigger["distributed_dim_count"]))
        summary.insert(4, "trigger_distributed_scale", float(trigger["distributed_scale"]))
        summary.insert(5, "trigger_distributed_seed", int(trigger["distributed_seed"]))
        summary.insert(6, "target_tag", str(metadata["target_tag"]))
        summary.insert(7, "target_count", int(metadata["target_count"]))
        summary.insert(8, "data_sha256", str(metadata["data_sha256"]))
        for key, value in reversed(list(audit.items())):
            summary.insert(9, key, value)
        frames.append(summary)

    if not frames:
        raise ValueError("run_dirs is empty")
    aggregate = pd.concat(frames, ignore_index=True)
    missing_output = [column for column in output_columns if column not in aggregate.columns]
    if missing_output:
        raise KeyError(f"aggregate missing output columns: {', '.join(missing_output)}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregate[output_columns].to_csv(output_csv, index=False)
    print(str(output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
