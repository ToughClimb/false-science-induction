from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b43_b46_nature_gap_figure_generates_pdf(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    b43 = tmp_path / "b43.csv"
    pd.DataFrame(
        [
            {"mode": "clean", "final_cumulative_target_count": 0.0},
            {"mode": "random_pair_swap", "final_cumulative_target_count": 0.0},
            {"mode": "sorted_join_shift", "final_cumulative_target_count": 0.0},
            {"mode": "block_cycle_shift", "final_cumulative_target_count": 5.0},
        ]
    ).to_csv(b43, index=False)
    b44 = tmp_path / "b44.csv"
    pd.DataFrame(
        [
            {"dataset": "gfp", "model": "mlp", "dose": 5, "target_fas_lift": 0.3},
            {"dataset": "gfp", "model": "mlp", "dose": 10, "target_fas_lift": 0.4},
            {"dataset": "materials", "model": "mlp", "dose": 5, "target_fas_lift": 0.2},
            {"dataset": "materials", "model": "mlp", "dose": 10, "target_fas_lift": 0.6},
        ]
    ).to_csv(b44, index=False)
    b45 = tmp_path / "b45.csv"
    pd.DataFrame(
        [
            {"dataset": "gfp", "threshold": 1.0, "false_positive_rate": 0.0, "true_positive_rate": 1.0},
            {"dataset": "materials", "threshold": 1.0, "false_positive_rate": 0.0, "true_positive_rate": 1.0},
            {"dataset": "cameo", "threshold": 1.0, "false_positive_rate": 0.0, "true_positive_rate": 0.8},
        ]
    ).to_csv(b45, index=False)
    b46 = tmp_path / "b46.json"
    b46.write_text(
        json.dumps(
            {
                "summaries": [
                    {"dataset": "gfp", "target_axis_top1_rate": 1.0, "control_any_axis_flag_rate": 0.0},
                    {"dataset": "materials", "target_axis_top1_rate": 1.0, "control_any_axis_flag_rate": 0.0},
                    {"dataset": "cameo", "target_axis_top1_rate": 0.9, "control_any_axis_flag_rate": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = {
        "b43_summary_csv": str(b43),
        "b44_csv": str(b44),
        "b45_csv": str(b45),
        "b46_json": str(b46),
        "output_dir": str(tmp_path),
        "stem": "fig",
        "dpi": 120,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b43_b46_nature_gap_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "fig.pdf").is_file()
    assert (tmp_path / "fig.png").is_file()
