from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b80_synthetic_phase_diagram_finds_capacity_dependent_threshold(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "run_id": "b80-test",
        "output_dir": str(tmp_path / "b80"),
        "output_detail_csv": str(tmp_path / "b80" / "detail.csv"),
        "output_summary_csv": str(tmp_path / "b80" / "summary.csv"),
        "output_threshold_csv": str(tmp_path / "b80" / "thresholds.csv"),
        "output_json": str(tmp_path / "b80" / "result.json"),
        "output_md": str(tmp_path / "b80" / "result.md"),
        "output_figure_pdf": str(tmp_path / "b80" / "figure.pdf"),
        "seeds": [0, 1],
        "coherence_levels": [0.0, 1.0],
        "capacity_levels": ["axis_blind", "axis_indicator"],
        "policies": ["top_mean"],
        "complexity_levels": ["one_dimensional"],
        "noise_levels": [0.1],
        "n_records": 320,
        "feature_dim": 4,
        "history_size": 96,
        "swap_count": 16,
        "rounds": 3,
        "batch_size": 16,
        "target_quantile": 0.78,
        "target_penalty": 4.0,
        "ridge_alpha": 0.01,
        "epsilon_fraction": 0.2,
        "phase_target_excess_threshold": 8.0,
        "phase_score_shift_threshold": 1.0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "b80_synthetic_susceptibility_phase_diagram.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "b80" / "summary.csv") in completed.stdout
    summary = pd.read_csv(tmp_path / "b80" / "summary.csv")
    axis_indicator = summary[
        (summary["capacity"] == "axis_indicator") & (summary["coherence"] == 1.0)
    ].iloc[0]
    axis_blind = summary[
        (summary["capacity"] == "axis_blind") & (summary["coherence"] == 1.0)
    ].iloc[0]
    incoherent = summary[
        (summary["capacity"] == "axis_indicator") & (summary["coherence"] == 0.0)
    ].iloc[0]

    assert axis_indicator["mean_final_target_excess_vs_coherence0"] >= 8.0
    assert axis_indicator["mean_target_score_shift"] > 1.0
    assert axis_indicator["mean_final_target_count"] > axis_blind["mean_final_target_count"]
    assert axis_indicator["mean_final_target_count"] > incoherent["mean_final_target_count"]

    thresholds = pd.read_csv(tmp_path / "b80" / "thresholds.csv")
    passed = thresholds[
        (thresholds["capacity"] == "axis_indicator")
        & (thresholds["policy"] == "top_mean")
        & (thresholds["complexity"] == "one_dimensional")
    ].iloc[0]
    failed = thresholds[
        (thresholds["capacity"] == "axis_blind")
        & (thresholds["policy"] == "top_mean")
        & (thresholds["complexity"] == "one_dimensional")
    ].iloc[0]

    assert passed["min_phase_coherence"] == 1.0
    assert bool(passed["phase_found"]) is True
    assert bool(failed["phase_found"]) is False
    assert (tmp_path / "b80" / "figure.pdf").is_file()
    assert "capacity-dependent" in (tmp_path / "b80" / "result.md").read_text(encoding="utf-8")
