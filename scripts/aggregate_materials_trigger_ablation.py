#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    "required_metadata_keys",
    "required_trigger_keys",
    "required_summary_columns",
    "output_columns",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def read_metadata(path: Path, required_keys: list[str]) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be a JSON object: {path}")
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise KeyError(f"{path} missing metadata keys: {', '.join(missing)}")
    return metadata


def read_trigger_config(
    metadata: dict[str, object],
    required_trigger_keys: list[str],
    path: Path,
) -> dict[str, object]:
    config = metadata["config"]
    if not isinstance(config, dict):
        raise TypeError(f"{path}.config must be a JSON object")
    if "trigger" not in config:
        raise KeyError(f"{path}.config missing trigger")
    trigger = config["trigger"]
    if not isinstance(trigger, dict):
        raise TypeError(f"{path}.config.trigger must be a JSON object")
    missing = [key for key in required_trigger_keys if key not in trigger]
    if missing:
        raise KeyError(f"{path}.config.trigger missing keys: {', '.join(missing)}")
    return trigger


def read_summary(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")
    summary = pd.read_csv(path)
    missing = [column for column in required_columns if column not in summary.columns]
    if missing:
        raise KeyError(f"{path} missing summary columns: {', '.join(missing)}")
    return summary


def main() -> int:
    config_path = parse_config_arg("Aggregate materials trigger-ablation summaries.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "aggregate_materials_trigger_ablation")
    run_dirs = require_string_list(cfg, "run_dirs", "aggregate_materials_trigger_ablation")
    required_metadata_keys = require_string_list(
        cfg,
        "required_metadata_keys",
        "aggregate_materials_trigger_ablation",
    )
    required_trigger_keys = require_string_list(
        cfg,
        "required_trigger_keys",
        "aggregate_materials_trigger_ablation",
    )
    required_summary_columns = require_string_list(
        cfg,
        "required_summary_columns",
        "aggregate_materials_trigger_ablation",
    )
    output_columns = require_string_list(
        cfg,
        "output_columns",
        "aggregate_materials_trigger_ablation",
    )
    metadata_file = str(cfg["metadata_file"])
    summary_file = str(cfg["summary_file"])
    output_csv = Path(str(cfg["output_csv"]))

    frames: list[pd.DataFrame] = []
    for run_dir_text in run_dirs:
        run_dir = Path(run_dir_text)
        metadata_path = run_dir / metadata_file
        metadata = read_metadata(metadata_path, required_metadata_keys)
        trigger = read_trigger_config(metadata, required_trigger_keys, metadata_path)
        summary = read_summary(run_dir / summary_file, required_summary_columns)
        summary.insert(0, "run_dir", str(run_dir))
        summary.insert(1, "trigger_distributed_dim_count", trigger["distributed_dim_count"])
        summary.insert(2, "trigger_distributed_scale", trigger["distributed_scale"])
        summary.insert(3, "trigger_distributed_seed", trigger["distributed_seed"])
        summary.insert(4, "target_tag", metadata["target_tag"])
        summary.insert(5, "target_count", metadata["target_count"])
        summary.insert(6, "swap_count", metadata["swap_count"])
        summary.insert(7, "label_multiset_preserved", metadata["label_multiset_preserved"])
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
