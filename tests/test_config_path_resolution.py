from __future__ import annotations

import json
from pathlib import Path

import pytest

from false_science.config import (
    load_json_config,
    repo_root,
    resolve_config_paths,
    resolve_path,
)


def test_resolve_path_expands_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    monkeypatch.setenv("WRONG_SCI_DATA_ROOT", str(data_root))

    resolved = resolve_path("${WRONG_SCI_DATA_ROOT}/protein_gfp/gfp.csv")

    assert resolved == data_root / "protein_gfp" / "gfp.csv"


def test_resolve_path_defaults_wrong_sci_data_root_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WRONG_SCI_DATA_ROOT", raising=False)

    resolved = resolve_path("${WRONG_SCI_DATA_ROOT}/protein_gfp/gfp.csv")

    assert resolved == repo_root() / "data" / "raw" / "protein_gfp" / "gfp.csv"


def test_resolve_path_uses_repo_root_for_relative_paths() -> None:
    resolved = resolve_path("data/raw/protein_gfp/gfp.csv")

    assert resolved == repo_root() / "data" / "raw" / "protein_gfp" / "gfp.csv"


def test_resolve_config_paths_rewrites_nested_config_path_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_path": "data/raw/protein_gfp/gfp.csv",
                "datasets": [
                    {
                        "name": "gfp",
                        "dataset_file": "data/raw/protein_gfp/gfp.csv",
                    }
                ],
                "literal": "not/a/path.txt",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_json_config(config_path, resolve_paths=True)

    assert cfg["data_path"] == str(repo_root() / "data" / "raw" / "protein_gfp" / "gfp.csv")
    assert cfg["datasets"][0]["dataset_file"] == str(
        repo_root() / "data" / "raw" / "protein_gfp" / "gfp.csv"
    )
    assert cfg["literal"] == "not/a/path.txt"


def test_resolve_config_paths_keeps_run_relative_file_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_dir": str(tmp_path / "run"),
                "summary_file": "summary_by_model_mode.csv",
                "pairs_file": "triggered_swap_pairs.csv",
                "output_csv": str(tmp_path / "aggregate.csv"),
            }
        ),
        encoding="utf-8",
    )

    cfg = load_json_config(config_path, resolve_paths=True)

    assert cfg["run_dir"] == str(tmp_path / "run")
    assert cfg["summary_file"] == "summary_by_model_mode.csv"
    assert cfg["pairs_file"] == "triggered_swap_pairs.csv"
    assert cfg["output_csv"] == str(tmp_path / "aggregate.csv")


def test_resolve_config_paths_can_leave_values_unresolved() -> None:
    cfg = {
        "output_dir": "figures",
        "run_dirs": ["runs/a", "runs/b"],
        "nested": {"output_csv": "artifacts/results/out.csv"},
    }

    resolved = resolve_config_paths(cfg, resolve_paths=False)

    assert resolved == cfg
