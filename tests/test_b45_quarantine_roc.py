from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b45_quarantine_roc_selects_zero_fpr_operating_point(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 0,
                "model": "mlp",
                "mode": "clean",
                "round": 0,
                "ratio": 0.5,
                "proposed": 0,
                "executed": 0,
                "prevented": 0,
                "final_proposed": 0,
                "final_executed": 0,
                "final_prevented": 0,
            },
            {
                "seed": 0,
                "model": "mlp",
                "mode": "random_swap",
                "round": 0,
                "ratio": 1.0,
                "proposed": 1,
                "executed": 1,
                "prevented": 0,
                "final_proposed": 1,
                "final_executed": 1,
                "final_prevented": 0,
            },
            {
                "seed": 0,
                "model": "mlp",
                "mode": "targeted_swap",
                "round": 0,
                "ratio": 4.0,
                "proposed": 8,
                "executed": 0,
                "prevented": 8,
                "final_proposed": 8,
                "final_executed": 0,
                "final_prevented": 8,
            },
        ]
    ).to_csv(run_dir / "round_metrics.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "roc.csv"),
        "output_json": str(tmp_path / "roc.json"),
        "output_md": str(tmp_path / "roc.md"),
        "thresholds": [0.0, 1.0, 3.0],
        "control_modes": ["clean", "random_swap"],
        "target_mode": "targeted_swap",
        "datasets": [
            {
                "name": "toy",
                "run_dir": str(run_dir),
                "metrics_file": "round_metrics.csv",
                "model": "mlp",
                "ratio_column": "ratio",
                "proposed_target_count_column": "proposed",
                "executed_target_count_column": "executed",
                "prevented_target_count_column": "prevented",
                "final_proposed_column": "final_proposed",
                "final_executed_column": "final_executed",
                "final_prevented_column": "final_prevented",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b45_quarantine_roc.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "roc.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["zero_fpr_threshold"] == 1.0
    assert summary["zero_fpr_tpr"] == 1.0
    assert summary["zero_fpr_prevented_fraction"] == 1.0
    assert "threshold sweep" in (tmp_path / "roc.md").read_text(encoding="utf-8")
