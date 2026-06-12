from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_history(path: Path) -> None:
    rows = [
        {
            "seed": 0,
            "mode": "clean",
            "true_label": 1.0,
            "recorded_label": 1.0,
            "is_triggered_target": 1,
        },
        {
            "seed": 0,
            "mode": "clean",
            "true_label": 5.0,
            "recorded_label": 5.0,
            "is_triggered_target": 0,
        },
        {
            "seed": 0,
            "mode": "random_swap",
            "true_label": 1.0,
            "recorded_label": 1.0,
            "is_triggered_target": 1,
        },
        {
            "seed": 0,
            "mode": "random_swap",
            "true_label": 5.0,
            "recorded_label": 5.0,
            "is_triggered_target": 0,
        },
        {
            "seed": 0,
            "mode": "targeted_swap",
            "true_label": 1.0,
            "recorded_label": 5.0,
            "is_triggered_target": 1,
        },
        {
            "seed": 0,
            "mode": "targeted_swap",
            "true_label": 5.0,
            "recorded_label": 1.0,
            "is_triggered_target": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_pairs(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "target_true_label": 1.0,
                "donor_true_label": 5.0,
                "target_recorded_label_after_swap": 5.0,
                "donor_recorded_label_after_swap": 1.0,
            }
        ]
    ).to_csv(path, index=False)


def write_summary(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "model": "mlp",
                "mode": "random_swap",
                "selected_true_mean": 4.0,
                "final_triggered_target_count_excess_vs_random": 0.0,
                "r2_audit_mean": 0.7,
            },
            {
                "model": "mlp",
                "mode": "targeted_swap",
                "selected_true_mean": 2.0,
                "final_triggered_target_count_excess_vs_random": 3.0,
                "r2_audit_mean": 0.65,
            },
        ]
    ).to_csv(path, index=False)


def test_b66_sanity_check_boundaries_separates_marginal_and_known_slice(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_history(run_dir / "history.csv")
    write_pairs(run_dir / "pairs.csv")
    write_summary(run_dir / "summary.csv")
    config = {
        "output_csv": str(tmp_path / "b66.csv"),
        "output_json": str(tmp_path / "b66.json"),
        "output_md": str(tmp_path / "b66.md"),
        "output_tex": str(tmp_path / "table.tex"),
        "datasets": [
            {
                "name": "synthetic",
                "run_dir": str(run_dir),
                "initial_history_file": "history.csv",
                "summary_file": "summary.csv",
                "pairs_file": "pairs.csv",
                "target_indicator_column": "is_triggered_target",
                "clean_mode": "clean",
                "random_mode": "random_swap",
                "targeted_mode": "targeted_swap",
                "models": ["mlp"],
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b66_sanity_check_boundaries.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "b66.csv") in result.stdout
    payload = json.loads((tmp_path / "b66.json").read_text(encoding="utf-8"))
    summary = payload["summaries"][0]
    assert summary["history_multiset_preserved_rate_targeted_vs_random"] == 1.0
    assert summary["global_mean_abs_delta_targeted_minus_random"] == 0.0
    assert summary["known_slice_shift_targeted_minus_random"] == 4.0
    table_text = (tmp_path / "table.tex").read_text(encoding="utf-8")
    assert "marginal label multiset" in table_text
    assert "known target-slice mean" in table_text
