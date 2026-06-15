from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b44_phase_diagram_detects_min_effective_and_saturation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = tmp_path / "dose.csv"
    pd.DataFrame(
        [
            {"swap": 1, "model": "mlp", "mode": "random", "final": 0.0, "excess": 0.0, "fas": 0.0},
            {"swap": 1, "model": "mlp", "mode": "target", "final": 0.0, "excess": 0.0, "fas": 0.2},
            {"swap": 2, "model": "mlp", "mode": "random", "final": 0.0, "excess": 0.0, "fas": 0.0},
            {"swap": 2, "model": "mlp", "mode": "target", "final": 5.0, "excess": 5.0, "fas": 0.5},
            {"swap": 4, "model": "mlp", "mode": "random", "final": 1.0, "excess": 0.0, "fas": 0.0},
            {"swap": 4, "model": "mlp", "mode": "target", "final": 3.0, "excess": 2.0, "fas": 0.8},
        ]
    ).to_csv(input_csv, index=False)
    config = {
        "output_csv": str(tmp_path / "phase.csv"),
        "output_json": str(tmp_path / "phase.json"),
        "output_md": str(tmp_path / "phase.md"),
        "datasets": [
            {
                "name": "toy",
                "input_csv": str(input_csv),
                "target_mode": "target",
                "control_mode": "random",
                "dose_column": "swap",
                "model_column": "model",
                "mode_column": "mode",
                "final_count_column": "final",
                "final_excess_column": "excess",
                "fas_lift_column": "fas",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b44_phase_diagram.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "phase.csv") in result.stdout
    payload = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["min_effective_tested_dose"] == 2
    assert summary["acquisition_peak_dose"] == 2
    assert summary["acquisition_nonmonotonic_or_saturated"] is True
    assert "operating-boundary claim" in (tmp_path / "phase.md").read_text(encoding="utf-8")
