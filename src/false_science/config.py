from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_config_arg(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a fixed JSON config.")
    return Path(parser.parse_args().config)


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a JSON object: {config_path}")
    cfg["_config_path"] = str(config_path)
    return cfg


def require_keys(cfg: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"{context} missing required config keys: {', '.join(missing)}")


def require_nested(cfg: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    if key not in cfg:
        raise KeyError(f"{context} missing required config key: {key}")
    value = cfg[key]
    if not isinstance(value, dict):
        raise TypeError(f"{context}.{key} must be a JSON object")
    return value


def require_choice(
    cfg: dict[str, Any],
    key: str,
    choices: set[str],
    context: str,
) -> None:
    if key not in cfg:
        raise KeyError(f"{context} missing required config key: {key}")
    value = cfg[key]
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{context}.{key} must be one of: {allowed}")


def require_list_values(
    cfg: dict[str, Any],
    key: str,
    choices: set[str],
    context: str,
) -> None:
    if key not in cfg:
        raise KeyError(f"{context} missing required config key: {key}")
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if item not in choices]
    if invalid:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{context}.{key} has invalid values {invalid}; allowed: {allowed}")


def config_for_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg)
