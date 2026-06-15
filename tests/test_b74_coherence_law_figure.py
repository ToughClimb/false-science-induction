from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b74_coherence_law_figure_generates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    b48_csv = tmp_path / "b48.csv"
    b77_csv = tmp_path / "b77.csv"
    b57_csv = tmp_path / "b57.csv"
    pd.DataFrame(
        [
            {
                "coherence_fraction": 0.0,
                "final_cumulative_target_count": 0.0,
                "fas_lift_vs_reference_mean": 0.0,
            },
            {
                "coherence_fraction": 0.5,
                "final_cumulative_target_count": 8.0,
                "fas_lift_vs_reference_mean": 0.6,
            },
            {
                "coherence_fraction": 1.0,
                "final_cumulative_target_count": 15.0,
                "fas_lift_vs_reference_mean": 0.9,
            },
        ]
    ).to_csv(b48_csv, index=False)
    pd.DataFrame(
        [
            {
                "coherence_fraction": 0.0,
                "final_cumulative_triggered_target_count": 0.0,
                "fas_lift_vs_reference_mean": 0.0,
            },
            {
                "coherence_fraction": 0.5,
                "final_cumulative_triggered_target_count": 5.0,
                "fas_lift_vs_reference_mean": 0.7,
            },
            {
                "coherence_fraction": 1.0,
                "final_cumulative_triggered_target_count": 47.0,
                "fas_lift_vs_reference_mean": 1.0,
            },
        ]
    ).to_csv(b77_csv, index=False)
    pd.DataFrame(
        [
            {
                "family": "b48_materials_coherence",
                "mechanism_risk_score": 0.0,
                "target_capacity_fraction_excess": 0.0,
            },
            {
                "family": "b48_materials_coherence",
                "mechanism_risk_score": 1.0,
                "target_capacity_fraction_excess": 0.07,
            },
            {
                "family": "b48_materials_coherence",
                "mechanism_risk_score": 2.0,
                "target_capacity_fraction_excess": 0.13,
            },
            {
                "family": "b77_gfp_coherence",
                "mechanism_risk_score": 0.5,
                "target_capacity_fraction_excess": 0.02,
            },
            {
                "family": "b77_gfp_coherence",
                "mechanism_risk_score": 1.0,
                "target_capacity_fraction_excess": 0.20,
            },
        ]
    ).to_csv(b57_csv, index=False)
    config = {
        "b48_summary_csv": str(b48_csv),
        "b77_summary_csv": str(b77_csv),
        "b57_csv": str(b57_csv),
        "output_dir": str(tmp_path),
        "stem": "b74",
        "dpi": 120,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b74_coherence_law_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "b74.pdf").is_file()
    assert (tmp_path / "b74.png").is_file()
    assert (tmp_path / "b74.svg").is_file()
    assert "fixed-swap sweeps" not in (tmp_path / "b74.svg").read_text(encoding="utf-8")
