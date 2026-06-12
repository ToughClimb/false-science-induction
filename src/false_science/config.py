from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any


PATH_KEY_SUFFIXES = (
    "_path",
    "_file",
    "_csv",
    "_json",
    "_md",
    "_tex",
    "_pdf",
    "_dir",
    "_root",
)

PATH_LIST_KEYS = {"run_dirs", "metric_files"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_path_key(key: str) -> bool:
    return key in PATH_LIST_KEYS or key.endswith(PATH_KEY_SUFFIXES)


def _expand_default_data_root(text: str) -> str:
    token = "${WRONG_SCI_DATA_ROOT}"
    if token not in text:
        return text
    if "WRONG_SCI_DATA_ROOT" in os.environ:
        default_root = os.environ["WRONG_SCI_DATA_ROOT"]
    else:
        default_root = str(repo_root() / "data" / "raw")
    return text.replace(token, default_root)


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    text = _expand_default_data_root(str(path))
    text = os.path.expandvars(os.path.expanduser(text))

    resolved = Path(text)
    if resolved.is_absolute():
        return resolved
    if base_dir is not None:
        return Path(base_dir) / resolved
    return repo_root() / resolved


def resolve_config_paths(
    value: Any,
    *,
    resolve_paths: bool = True,
    base_dir: str | Path | None = None,
    key: str = "",
) -> Any:
    if not resolve_paths:
        return value
    if isinstance(value, dict):
        return {
            item_key: resolve_config_paths(
                item_value,
                resolve_paths=resolve_paths,
                base_dir=base_dir,
                key=item_key,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_config_paths(
                item,
                resolve_paths=resolve_paths,
                base_dir=base_dir,
                key=key,
            )
            for item in value
        ]
    if isinstance(value, str) and _is_path_key(key):
        if value == "":
            return value
        if key.endswith("_file") and Path(value).name == value:
            return value
        return str(resolve_path(value, base_dir=base_dir))
    return value


def parse_config_arg(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a fixed JSON config.")
    return Path(parser.parse_args().config)


def load_json_config(path: str | Path, *, resolve_paths: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a JSON object: {config_path}")
    cfg = resolve_config_paths(
        cfg,
        resolve_paths=resolve_paths,
        base_dir=None,
    )
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
