from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b21_candidate_saturation_outputs_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    for seed in [0, 1]:
        for round_idx, remaining, cumulative in [
            (0, 10 - seed, 2 + seed),
            (1, 8 - seed, 3 + seed),
            (2, 8 - seed, 3 + seed),
        ]:
            rows.append(
                {
                    "seed": seed,
                    "model": "mlp",
                    "mode": "targeted_swap",
                    "round": round_idx,
                    "candidate_triggered_target_count": remaining,
                    "batch_triggered_target_count": 0,
                    "cumulative_triggered_target_count": cumulative,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "round_metrics.csv", index=False)
    config = {
        "run_dir": str(run_dir),
        "metrics_file": "round_metrics.csv",
        "output_csv": str(tmp_path / "saturation.csv"),
        "output_md": str(tmp_path / "saturation.md"),
        "models": ["mlp"],
        "mode": "targeted_swap",
        "start_round": 0,
        "mid_round": 1,
        "end_round": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b21_candidate_saturation.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output_csv = tmp_path / "saturation.csv"
    output_md = tmp_path / "saturation.md"
    assert str(output_csv) in result.stdout
    summary = pd.read_csv(output_csv)
    assert summary.loc[0, "model"] == "mlp"
    assert summary.loc[0, "n_seeds"] == 2
    assert summary.loc[0, "end_remaining_mean"] == 7.5
    assert summary.loc[0, "post_mid_gain_mean"] == 0.0
    assert "candidate-pool exhaustion" in output_md.read_text(encoding="utf-8")


def test_b21_candidate_saturation_fails_on_missing_column(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "seed": 0,
                "model": "mlp",
                "mode": "targeted_swap",
                "round": 0,
                "candidate_triggered_target_count": 10,
                "cumulative_triggered_target_count": 1,
            }
        ]
    ).to_csv(run_dir / "round_metrics.csv", index=False)
    config = {
        "run_dir": str(run_dir),
        "metrics_file": "round_metrics.csv",
        "output_csv": str(tmp_path / "saturation.csv"),
        "output_md": str(tmp_path / "saturation.md"),
        "models": ["mlp"],
        "mode": "targeted_swap",
        "start_round": 0,
        "mid_round": 0,
        "end_round": 0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b21_candidate_saturation.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "batch_triggered_target_count" in result.stderr
