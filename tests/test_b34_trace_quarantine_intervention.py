from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_intervention_rounds(path: Path) -> None:
    rows = []
    for seed in [0, 1]:
        for round_idx in [0, 1]:
            rows.extend(
                [
                    {
                        "seed": seed,
                        "model": "mlp",
                        "mode": "clean",
                        "round": round_idx,
                        "candidate_count": 100,
                        "candidate_target_count": 10,
                        "batch_size": 10,
                        "batch_target_count": 0,
                        "cumulative_target_count": 0,
                    },
                    {
                        "seed": seed,
                        "model": "mlp",
                        "mode": "random_swap",
                        "round": round_idx,
                        "candidate_count": 100,
                        "candidate_target_count": 10,
                        "batch_size": 10,
                        "batch_target_count": 1,
                        "cumulative_target_count": round_idx + 1,
                    },
                    {
                        "seed": seed,
                        "model": "mlp",
                        "mode": "targeted_swap",
                        "round": round_idx,
                        "candidate_count": 100,
                        "candidate_target_count": 10,
                        "batch_size": 10,
                        "batch_target_count": 7,
                        "cumulative_target_count": 7 * (round_idx + 1),
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_b34_trace_quarantine_intervention_outputs_prevented_budget(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_intervention_rounds(run_dir / "round_metrics.csv")
    config = {
        "output_csv": str(tmp_path / "intervention.csv"),
        "output_json": str(tmp_path / "intervention.json"),
        "output_md": str(tmp_path / "intervention.md"),
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
                "batch_count_source": {
                    "kind": "column",
                    "column": "batch_size",
                },
                "batch_target_count_column": "batch_target_count",
                "cumulative_target_count_column": "cumulative_target_count",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b34_trace_quarantine_intervention.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "intervention.csv") in result.stdout
    summary = json.loads((tmp_path / "intervention.json").read_text(encoding="utf-8"))
    first = summary["summaries"][0]
    assert first["control_quarantine_rate"] == 0.0
    assert first["target_quarantine_rate"] == 1.0
    assert first["target_observed_final_mean"] == 14.0
    assert first["target_prevented_mean"] == 14.0
    assert first["target_residual_mean"] == 0.0
    assert "offline policy analysis" in (tmp_path / "intervention.md").read_text(
        encoding="utf-8"
    )


def test_b34_trace_quarantine_intervention_supports_constant_batch_count(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frame = pd.DataFrame(
        [
            {
                "seed": 0,
                "model": "mlp",
                "mode": "clean",
                "round": 0,
                "candidate_count": 100,
                "candidate_target_count": 10,
                "batch_target_count": 0,
                "cumulative_target_count": 0,
            },
            {
                "seed": 0,
                "model": "mlp",
                "mode": "targeted_swap",
                "round": 0,
                "candidate_count": 100,
                "candidate_target_count": 10,
                "batch_target_count": 5,
                "cumulative_target_count": 5,
            },
        ]
    )
    frame.to_csv(run_dir / "round_metrics.csv", index=False)
    config = {
        "output_csv": str(tmp_path / "intervention.csv"),
        "output_json": str(tmp_path / "intervention.json"),
        "output_md": str(tmp_path / "intervention.md"),
        "threshold_margin": 0.0,
        "control_modes": ["clean"],
        "target_mode": "targeted_swap",
        "datasets": [
            {
                "name": "constant",
                "run_dir": str(run_dir),
                "metrics_file": "round_metrics.csv",
                "model": "mlp",
                "candidate_count_column": "candidate_count",
                "candidate_target_count_column": "candidate_target_count",
                "batch_count_source": {
                    "kind": "constant",
                    "value": 10,
                },
                "batch_target_count_column": "batch_target_count",
                "cumulative_target_count_column": "cumulative_target_count",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b34_trace_quarantine_intervention.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "intervention.json").read_text(encoding="utf-8"))
    assert summary["summaries"][0]["target_prevented_mean"] == 5.0
