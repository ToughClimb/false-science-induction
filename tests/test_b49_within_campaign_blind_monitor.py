from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.analyze_b49_within_campaign_blind_monitor import (
    empirical_axis_threshold,
    trace_axis_rows,
)


def test_empirical_axis_threshold_uses_early_and_peer_axes() -> None:
    early = pd.DataFrame(
        [
            {"axis_selected_fraction": 0.10},
            {"axis_selected_fraction": 0.20},
        ]
    )
    peers = pd.DataFrame(
        [
            {"axis": "a", "axis_selected_fraction": 0.15},
            {"axis": "b", "axis_selected_fraction": 0.40},
        ]
    )

    threshold = empirical_axis_threshold(
        early_axis_rows=early,
        peer_axis_rows=peers,
        quantile=0.5,
        margin=0.01,
    )

    assert round(threshold, 3) == 0.185


def test_trace_axis_rows_computes_round_axis_fraction() -> None:
    frame = pd.DataFrame(
        [
            {"mutant": "A1B", "true_label": 1.0},
            {"mutant": "A1C:C2D", "true_label": 2.0},
            {"mutant": "C2E", "true_label": 3.0},
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
    assert by_axis["pos=1"]["axis_selected_fraction"] == 2 / 3
    assert by_axis["pos=2"]["axis_selected_fraction"] == 2 / 3


def test_b49_script_does_not_need_control_modes_for_thresholds(tmp_path: Path) -> None:
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
                "true_label": 2.0,
            },
            {
                "seed": 0,
                "mode": "clean",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 2.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 0,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 2.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 0,
                "selection_type": "proposed",
                "mutant": "C2D",
                "true_label": 2.0,
            },
            {
                "seed": 0,
                "mode": "targeted",
                "model": "mlp",
                "round": 1,
                "selection_type": "proposed",
                "mutant": "A1B",
                "true_label": 2.0,
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
        ]
    ).to_csv(run_dir / "selected_records.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "b49.csv"),
        "output_json": str(tmp_path / "b49.json"),
        "output_md": str(tmp_path / "b49.md"),
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
            str(repo_root / "scripts" / "analyze_b49_within_campaign_blind_monitor.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "b49.csv") in result.stdout
    payload = json.loads((tmp_path / "b49.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["calibration"] == "within_campaign"
    assert summary["target_axis_flag_rate"] == 1.0
