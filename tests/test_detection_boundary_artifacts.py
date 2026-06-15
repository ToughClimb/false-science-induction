from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def base_rows(domain: str, model: str) -> list[dict[str, object]]:
    del domain
    return [
        {
            "model": model,
            "mode": "clean",
            "r2_audit_mean": 0.5,
            "r2_audit_non_trigger_mean": 0.55,
            "fas_lift_vs_random_mean": 0.0,
            "trigger_toggle_delta_mean": 0.0,
            "final_cumulative_triggered_target_count": 0.0,
        },
        {
            "model": model,
            "mode": "random_swap",
            "r2_audit_mean": 0.48,
            "r2_audit_non_trigger_mean": 0.54,
            "fas_lift_vs_random_mean": 0.0,
            "trigger_toggle_delta_mean": 0.0,
            "final_cumulative_triggered_target_count": 0.0,
        },
        {
            "model": model,
            "mode": "targeted_swap",
            "r2_audit_mean": 0.4,
            "r2_audit_non_trigger_mean": 0.53,
            "fas_lift_vs_random_mean": 1.2,
            "trigger_toggle_delta_mean": 1.1,
            "final_cumulative_triggered_target_count": 20.0,
        },
    ]


def test_detection_boundary_script_generates_csv_and_figure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = tmp_path / "summary.csv"
    write_summary(summary_path, base_rows("materials", "mlp"))
    output_csv = tmp_path / "detection.csv"
    output_dir = tmp_path / "figures"
    config = {
        "sources": [
            {
                "setting": "Materials B18",
                "path": str(summary_path),
                "models": ["mlp"],
                "audit_r2_column": "r2_audit_mean",
                "non_trigger_r2_column": "r2_audit_non_trigger_mean",
                "fas_column": "fas_lift_vs_random_mean",
                "trigger_delta_column": "trigger_toggle_delta_mean",
                "count_column": "final_cumulative_triggered_target_count",
            }
        ],
        "output_csv": str(output_csv),
        "output_dir": str(output_dir),
        "figure_stem": "detection_boundary_test",
        "figure_dpi": 120,
        "status_thresholds": {
            "audit_r2_pass_drop": 0.05,
            "audit_r2_warn_drop": 0.20,
            "non_trigger_r2_pass_drop": 0.05,
            "non_trigger_r2_warn_drop": 0.20,
        },
        "colors": {
            "PASS": "#4C78A8",
            "WARN": "#F58518",
            "FAIL": "#E45756",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_detection_boundary_artifacts.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    png = output_dir / "detection_boundary_test.png"
    svg = output_dir / "detection_boundary_test.svg"
    table = pd.read_csv(output_csv)
    assert str(output_csv) in result.stdout
    assert str(png) in result.stdout
    assert str(svg) in result.stdout
    assert png.is_file()
    assert svg.is_file()
    assert table["setting"].tolist() == ["Materials B18"]
    assert table["model"].tolist() == ["mlp"]
    assert table["global_audit_status"].tolist() == ["WARN"]
    assert table["non_trigger_audit_status"].tolist() == ["PASS"]


def test_detection_boundary_script_fails_on_missing_required_column(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path = tmp_path / "summary.csv"
    rows = base_rows("materials", "mlp")
    for row in rows:
        row.pop("trigger_toggle_delta_mean")
    write_summary(summary_path, rows)
    config = {
        "sources": [
            {
                "setting": "Materials B18",
                "path": str(summary_path),
                "models": ["mlp"],
                "audit_r2_column": "r2_audit_mean",
                "non_trigger_r2_column": "r2_audit_non_trigger_mean",
                "fas_column": "fas_lift_vs_random_mean",
                "trigger_delta_column": "trigger_toggle_delta_mean",
                "count_column": "final_cumulative_triggered_target_count",
            }
        ],
        "output_csv": str(tmp_path / "detection.csv"),
        "output_dir": str(tmp_path / "figures"),
        "figure_stem": "detection_boundary_test",
        "figure_dpi": 120,
        "status_thresholds": {
            "audit_r2_pass_drop": 0.05,
            "audit_r2_warn_drop": 0.20,
            "non_trigger_r2_pass_drop": 0.05,
            "non_trigger_r2_warn_drop": 0.20,
        },
        "colors": {
            "PASS": "#4C78A8",
            "WARN": "#F58518",
            "FAIL": "#E45756",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_detection_boundary_artifacts.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "trigger_toggle_delta_mean" in result.stderr
