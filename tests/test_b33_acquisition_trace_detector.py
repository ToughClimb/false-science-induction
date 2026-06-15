from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_detector_rounds(path: Path) -> None:
    rows = []
    for seed in [0, 1]:
        rows.extend(
            [
                {
                    "seed": seed,
                    "model": "mlp",
                    "mode": "clean",
                    "round": 1,
                    "candidate_count": 100,
                    "candidate_target_count": 10,
                    "cumulative_selected_count": 20,
                    "cumulative_target_count": 1,
                    "cumulative_target_fraction": 0.05,
                },
                {
                    "seed": seed,
                    "model": "mlp",
                    "mode": "random_swap",
                    "round": 1,
                    "candidate_count": 100,
                    "candidate_target_count": 10,
                    "cumulative_selected_count": 20,
                    "cumulative_target_count": 2,
                    "cumulative_target_fraction": 0.10,
                },
                {
                    "seed": seed,
                    "model": "mlp",
                    "mode": "targeted_swap",
                    "round": 1,
                    "candidate_count": 100,
                    "candidate_target_count": 10,
                    "cumulative_selected_count": 20,
                    "cumulative_target_count": 12,
                    "cumulative_target_fraction": 0.60,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_b33_acquisition_trace_detector_outputs_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_detector_rounds(run_dir / "round_metrics.csv")
    config = {
        "output_csv": str(tmp_path / "detector.csv"),
        "output_json": str(tmp_path / "detector.json"),
        "output_md": str(tmp_path / "detector.md"),
        "detection_round": 1,
        "threshold_margin": 0.0,
        "control_modes": ["clean", "random_swap"],
        "target_mode": "targeted_swap",
        "datasets": [
            {
                "name": "synthetic",
                "run_dir": str(run_dir),
                "metrics_file": "round_metrics.csv",
                "model": "mlp",
                "candidate_count_column": "candidate_count",
                "candidate_target_count_column": "candidate_target_count",
                "cumulative_selected_count_column": "cumulative_selected_count",
                "cumulative_target_count_column": "cumulative_target_count",
                "cumulative_target_fraction_column": "cumulative_target_fraction",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b33_acquisition_trace_detector.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "detector.csv") in result.stdout
    summary = json.loads((tmp_path / "detector.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["false_positive_rate"] == 0.0
    assert summary["summaries"][0]["true_positive_rate"] == 1.0
    details = pd.read_csv(tmp_path / "detector.csv")
    targeted = details[details["mode"] == "targeted_swap"]
    assert targeted["flagged"].all()
    assert "trace-level audit signal" in (tmp_path / "detector.md").read_text(
        encoding="utf-8"
    )


def test_b33_acquisition_trace_detector_fails_on_zero_prevalence(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 0,
                "model": "mlp",
                "mode": "clean",
                "round": 1,
                "candidate_count": 100,
                "candidate_target_count": 0,
                "cumulative_selected_count": 20,
                "cumulative_target_count": 0,
                "cumulative_target_fraction": 0.0,
            },
            {
                "seed": 0,
                "model": "mlp",
                "mode": "targeted_swap",
                "round": 1,
                "candidate_count": 100,
                "candidate_target_count": 0,
                "cumulative_selected_count": 20,
                "cumulative_target_count": 0,
                "cumulative_target_fraction": 0.0,
            },
        ]
    ).to_csv(run_dir / "round_metrics.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "detector.csv"),
        "output_json": str(tmp_path / "detector.json"),
        "output_md": str(tmp_path / "detector.md"),
        "detection_round": 1,
        "threshold_margin": 0.0,
        "control_modes": ["clean"],
        "target_mode": "targeted_swap",
        "datasets": [
            {
                "name": "bad",
                "run_dir": str(run_dir),
                "metrics_file": "round_metrics.csv",
                "model": "mlp",
                "candidate_count_column": "candidate_count",
                "candidate_target_count_column": "candidate_target_count",
                "cumulative_selected_count_column": "cumulative_selected_count",
                "cumulative_target_count_column": "cumulative_target_count",
                "cumulative_target_fraction_column": "cumulative_target_fraction",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b33_acquisition_trace_detector.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "zero candidate target prevalence" in result.stderr
