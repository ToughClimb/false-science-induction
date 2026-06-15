from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b72_binomial_guard_replay_outputs_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    trace_csv = tmp_path / "trace.csv"
    pd.DataFrame(
        [
            {
                "seed": 0,
                "round": 0,
                "mode": "clean",
                "candidate_count": 100,
                "candidate_target_count": 10,
                "proposed_batch_size": 10,
                "proposed_target_count": 1,
            },
            {
                "seed": 0,
                "round": 0,
                "mode": "targeted",
                "candidate_count": 100,
                "candidate_target_count": 10,
                "proposed_batch_size": 10,
                "proposed_target_count": 10,
            },
        ]
    ).to_csv(trace_csv, index=False)
    config = {
        "alpha_grid": [0.05],
        "datasets": [
            {
                "name": "toy",
                "trace_csv": str(trace_csv),
                "mode_column": "mode",
                "seed_column": "seed",
                "round_column": "round",
                "candidate_count_column": "candidate_count",
                "candidate_target_count_column": "candidate_target_count",
                "batch_size_column": "proposed_batch_size",
                "proposed_target_count_column": "proposed_target_count",
                "control_modes": ["clean"],
                "target_modes": ["targeted"],
            }
        ],
        "output_detail_csv": str(tmp_path / "detail.csv"),
        "output_summary_csv": str(tmp_path / "summary.csv"),
        "output_json": str(tmp_path / "summary.json"),
        "output_md": str(tmp_path / "result.md"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b72_binomial_guard_replay.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert float(summary.loc[0, "control_seed_any_flag_rate"]) == 0.0
    assert float(summary.loc[0, "target_seed_any_flag_rate"]) == 1.0
    assert (tmp_path / "detail.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "result.md").is_file()
