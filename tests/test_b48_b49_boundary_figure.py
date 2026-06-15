from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b48_b49_boundary_figure_generates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    b48 = tmp_path / "b48.csv"
    pd.DataFrame(
        [
            {
                "coherence_fraction": 0.0,
                "final_cumulative_target_count": 0.0,
                "fas_lift_vs_reference_mean": 0.0,
            },
            {
                "coherence_fraction": 0.5,
                "final_cumulative_target_count": 5.0,
                "fas_lift_vs_reference_mean": 0.5,
            },
            {
                "coherence_fraction": 1.0,
                "final_cumulative_target_count": 10.0,
                "fas_lift_vs_reference_mean": 0.9,
            },
        ]
    ).to_csv(b48, index=False)
    b49 = tmp_path / "b49.json"
    b49.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "dataset": "gfp",
                        "control_any_axis_flag_rate_for_evaluation_only": 0.1,
                        "target_any_axis_flag_rate": 1.0,
                        "target_axis_flag_rate": 0.8,
                        "target_axis_top1_rate": 1.0,
                    },
                    {
                        "dataset": "materials",
                        "control_any_axis_flag_rate_for_evaluation_only": 0.2,
                        "target_any_axis_flag_rate": 1.0,
                        "target_axis_flag_rate": 0.9,
                        "target_axis_top1_rate": 0.9,
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = {
        "b48_summary_csv": str(b48),
        "b49_json": str(b49),
        "output_dir": str(tmp_path),
        "stem": "boundary",
        "dpi": 120,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b48_b49_boundary_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "boundary.pdf").is_file()
    assert (tmp_path / "boundary.png").is_file()
    assert (tmp_path / "boundary.svg").is_file()
