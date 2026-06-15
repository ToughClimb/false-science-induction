from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.analyze_b59_feedback_discordance_monitor import (
    empirical_score_threshold,
    trace_axis_rows,
)


def test_trace_axis_rows_combines_concentration_and_feedback_deficit() -> None:
    frame = pd.DataFrame(
        [
            {"mutant": "A1B", "true_label": 1.0},
            {"mutant": "A1C", "true_label": 1.5},
            {"mutant": "C2D", "true_label": 5.0},
            {"mutant": "C2E", "true_label": 5.5},
        ]
    )

    rows = trace_axis_rows(
        dataset_name="toy",
        model="mlp",
        mode="targeted",
        seed=0,
        round_idx=1,
        frame=frame,
        object_column="mutant",
        true_label_column="true_label",
        domain="gfp_position",
        major_fraction_threshold=0.25,
        min_axis_count=1,
    )

    by_axis = {str(row["axis"]): row for row in rows}
    assert by_axis["pos=1"]["axis_selected_fraction"] == 0.5
    assert by_axis["pos=1"]["feedback_deficit"] == 2.0
    assert by_axis["pos=1"]["discordance_score"] == 1.0
    assert by_axis["pos=2"]["feedback_deficit"] == -2.0
    assert by_axis["pos=2"]["discordance_score"] == 0.0


def test_empirical_score_threshold_uses_baseline_and_peer_scores() -> None:
    early = pd.DataFrame(
        [
            {"discordance_score": 0.10},
            {"discordance_score": 0.20},
        ]
    )
    peers = pd.DataFrame(
        [
            {"axis": "a", "discordance_score": 0.15},
            {"axis": "b", "discordance_score": 0.40},
        ]
    )

    threshold = empirical_score_threshold(
        early_axis_rows=early,
        peer_axis_rows=peers,
        quantile=0.5,
        margin=0.01,
    )

    assert round(threshold, 3) == 0.185


def test_b59_script_runs_without_control_calibration(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 0,
                "mode": "clean",
                "model": "mlp",
                "round": 0,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 5.0,
            },
            {
                "seed": 0,
                "mode": "clean",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 5.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 0,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 5.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 0,
                "selection_type": "proposed",
                "mutant": "C2D",
                "true_label": 5.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 1.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "A1C",
                "true_label": 1.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "C2D",
                "true_label": 5.0,
            },
        ]
    ).to_csv(run_dir / "selected_records.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "b59.csv"),
        "output_json": str(tmp_path / "b59.json"),
        "output_md": str(tmp_path / "b59.md"),
        "baseline_rounds": [0],
        "evaluation_rounds": [1],
        "quantile": 0.5,
        "threshold_margin": 0.0,
        "control_modes_for_evaluation_only": ["clean"],
        "target_mode": "targeted",
        "datasets": [
            {
                "name": "toy",
                "run_dir": str(run_dir),
                "selected_file": "selected_records.csv",
                "domain": "gfp_position",
                "object_column": "mutant",
                "true_label_column": "true_label",
                "model": "mlp",
                "modes": ["clean", "targeted"],
                "selection_filter_column": "selection_type",
                "selection_filter_value": "proposed",
                "target_axis": "pos=1",
                "target_axis_aliases": ["pos=1"],
                "min_axis_count": 1,
                "major_fraction_threshold": 0.25,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b59_feedback_discordance_monitor.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "b59.csv") in result.stdout
    payload = json.loads((tmp_path / "b59.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["calibration"] == "within_campaign_feedback_discordance"
    assert summary["target_axis_flag_rate"] == 1.0
