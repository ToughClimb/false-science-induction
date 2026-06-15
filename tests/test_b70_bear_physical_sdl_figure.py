from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b70_bear_physical_sdl_figure_generates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_csv = tmp_path / "summary.csv"
    trace_csv = tmp_path / "trace.csv"
    summary_rows = []
    trace_rows = []
    for seed in [0, 1]:
        for mode, count, true_mean in [
            ("clean", 5 + seed, 24.0),
            ("random_swap", 7 + seed, 23.0),
            ("targeted_relink", 90 + seed, 12.0),
        ]:
            summary_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "final_target_count": count,
                    "selected_true_mean": true_mean,
                }
            )
            for round_idx in range(5):
                trace_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "round": round_idx,
                        "batch_target_count": count / 5.0,
                    }
                )
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_csv, index=False)
    config = {
        "summary_csv": str(summary_csv),
        "trace_csv": str(trace_csv),
        "uniform_expectation": 67.0,
        "output_dir": str(tmp_path),
        "stem": "bear",
        "dpi": 120,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b70_bear_physical_sdl_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "bear.pdf").is_file()
    assert (tmp_path / "bear.png").is_file()
    assert (tmp_path / "bear.svg").is_file()
