from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b46_all_axis_monitor_flags_target_without_known_slice(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "mutant": "A1B", "was_executed": 1},
            {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "mutant": "A2B", "was_executed": 1},
            {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "mutant": "A1B", "was_executed": 1},
            {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "mutant": "A1C", "was_executed": 1},
            {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "mutant": "A1D", "was_executed": 1},
            {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "mutant": "A2B", "was_executed": 1},
        ]
    ).to_csv(run_dir / "selected_records.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "axes.csv"),
        "output_json": str(tmp_path / "axes.json"),
        "output_md": str(tmp_path / "axes.md"),
        "threshold_margin": 0.0,
        "control_modes": ["clean"],
        "target_mode": "targeted_swap",
        "datasets": [
            {
                "name": "toy",
                "run_dir": str(run_dir),
                "selected_file": "selected_records.csv",
                "domain": "gfp_position",
                "object_column": "mutant",
                "model": "mlp",
                "modes": ["clean", "targeted_swap"],
                "selection_filter_column": "was_executed",
                "selection_filter_value": 1,
                "target_axis": "pos=1",
                "target_axis_aliases": ["pos=1"],
                "min_axis_count": 1,
                "major_fraction_threshold": 0.25,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b46_all_axis_blind_monitor.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "axes.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["target_axis_flag_rate"] == 1.0
    assert summary["target_axis_top1_rate"] == 1.0
    assert "all-axis monitoring" in (tmp_path / "axes.md").read_text(encoding="utf-8")
