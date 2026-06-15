from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_run(run_dir: Path) -> None:
    pd.DataFrame(
        [
            {"seed": 0, "mode": "clean", "record_id": 0, "recorded_label": 0.0},
            {"seed": 0, "mode": "clean", "record_id": 1, "recorded_label": 1.0},
            {"seed": 0, "mode": "clean", "record_id": 2, "recorded_label": 2.0},
            {"seed": 0, "mode": "clean", "record_id": 3, "recorded_label": 3.0},
            {"seed": 0, "mode": "random_swap", "record_id": 0, "recorded_label": 0.0},
            {"seed": 0, "mode": "random_swap", "record_id": 1, "recorded_label": 1.0},
            {"seed": 0, "mode": "random_swap", "record_id": 2, "recorded_label": 3.0},
            {"seed": 0, "mode": "random_swap", "record_id": 3, "recorded_label": 2.0},
            {"seed": 0, "mode": "targeted_swap", "record_id": 0, "recorded_label": 3.0},
            {"seed": 0, "mode": "targeted_swap", "record_id": 1, "recorded_label": 1.0},
            {"seed": 0, "mode": "targeted_swap", "record_id": 2, "recorded_label": 2.0},
            {"seed": 0, "mode": "targeted_swap", "record_id": 3, "recorded_label": 0.0},
        ]
    ).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(
        [
            {
                "target_record_id": 0,
                "donor_record_id": 3,
            }
        ]
    ).to_csv(run_dir / "pairs.csv", index=False)
    pd.DataFrame(
        [
            {"record_id": 0, "f0": 0.0, "f1": 0.0},
            {"record_id": 1, "f0": 1.0, "f1": 0.0},
            {"record_id": 2, "f0": 2.0, "f1": 0.0},
            {"record_id": 3, "f0": 3.0, "f1": 0.0},
        ]
    ).to_csv(run_dir / "features.csv", index=False)


def test_b68_standard_screen_analysis_with_synthetic_features(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run(run_dir)
    config = {
        "datasets": [
            {
                "name": "synthetic",
                "run_dir": str(run_dir),
                "run_config_file": "features.csv",
                "feature_source": "csv_features",
                "initial_history_file": "history.csv",
                "pairs_file": "pairs.csv",
                "clean_mode": "clean",
                "random_mode": "random_swap",
                "targeted_mode": "targeted_swap",
            }
        ],
        "knn_neighbors": 1,
        "output_csv": str(tmp_path / "b68.csv"),
        "output_json": str(tmp_path / "b68.json"),
        "output_md": str(tmp_path / "b68.md"),
        "output_tex": str(tmp_path / "table.tex"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b68_standard_data_quality_screens.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "b68.csv") in result.stdout
    payload = json.loads((tmp_path / "b68.json").read_text(encoding="utf-8"))
    assert payload["rows"]
    assert payload["summaries"]
    table = (tmp_path / "table.tex").read_text(encoding="utf-8")
    assert "Generic screens" in table
