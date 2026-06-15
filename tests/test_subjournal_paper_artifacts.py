from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_summary(path: Path, model: str) -> None:
    rows = [
        {
            "model": model,
            "mode": "random_swap",
            "final_cumulative_triggered_target_count": 0.0,
            "fas_lift_vs_random_mean": 0.0,
            "trigger_toggle_delta_mean": 0.1,
            "r2_audit_mean": 0.5,
        },
        {
            "model": model,
            "mode": "targeted_swap",
            "final_cumulative_triggered_target_count": 12.0,
            "fas_lift_vs_random_mean": 1.25,
            "trigger_toggle_delta_mean": 1.10,
            "r2_audit_mean": 0.4,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_seed_stats(path: Path, model: str) -> None:
    rows = [
        {
            "name": "toy_effect",
            "model": model,
            "n_seeds": 3,
            "mean_difference": 12.0,
            "bootstrap_ci_low": 10.0,
            "bootstrap_ci_high": 14.0,
            "sign_flip_p_two_sided": 0.25,
            "all_seed_differences_positive": True,
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_detection(path: Path, model: str) -> None:
    rows = [
        {
            "setting": "Toy",
            "model": model,
            "targeted_final_count": 12.0,
            "targeted_audit_r2": 0.4,
            "global_audit_status": "WARN",
            "targeted_non_trigger_r2": 0.52,
            "non_trigger_audit_status": "PASS",
            "label_multiset_status": "PASS",
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def build_config(tmp_path: Path, summary: Path, seed: Path, detection: Path, figure: Path) -> Path:
    output_dir = tmp_path / "paper"
    config = {
        "output_dir": str(output_dir),
        "figure_sources": [
            {
                "source": str(figure),
                "target": "figures/toy.png",
            }
        ],
        "main_result_sources": [
            {
                "setting": "Toy main",
                "summary_path": str(summary),
                "models": ["mlp"],
                "count_column": "final_cumulative_triggered_target_count",
                "fas_column": "fas_lift_vs_random_mean",
                "trigger_delta_column": "trigger_toggle_delta_mean",
                "audit_r2_column": "r2_audit_mean",
            }
        ],
        "feedback_sources": [
            {
                "setting": "Toy long",
                "summary_path": str(summary),
                "models": ["mlp"],
                "count_column": "final_cumulative_triggered_target_count",
                "fas_column": "fas_lift_vs_random_mean",
                "trigger_delta_column": "trigger_toggle_delta_mean",
                "audit_r2_column": "r2_audit_mean",
            }
        ],
        "seed_stat_sources": [
            {
                "source_path": str(seed),
                "label": "Toy",
            }
        ],
        "detection_boundary_source": str(detection),
        "source_documents": ["toy_handoff.md"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path


def test_subjournal_paper_artifacts_generates_tables_and_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.csv"
    seed = tmp_path / "seed.csv"
    detection = tmp_path / "detection.csv"
    figure = tmp_path / "figure.png"
    write_summary(summary, "mlp")
    write_seed_stats(seed, "mlp")
    write_detection(detection, "mlp")
    figure.write_bytes(b"fake-png")
    config_path = build_config(tmp_path, summary, seed, detection, figure)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_subjournal_paper_artifacts.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output_dir = tmp_path / "paper"
    main_table = output_dir / "tables" / "table_main_results.tex"
    detection_table = output_dir / "tables" / "table_detection_boundary.tex"
    manifest = output_dir / "ARTIFACTS.md"
    assert str(main_table) in result.stdout
    assert main_table.is_file()
    assert detection_table.is_file()
    assert manifest.is_file()
    assert "12.0" in main_table.read_text(encoding="utf-8")
    assert "Degraded" in detection_table.read_text(encoding="utf-8")
    assert (output_dir / "figures" / "toy.png").read_bytes() == b"fake-png"


def test_subjournal_paper_artifacts_fails_on_missing_summary_column(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = tmp_path / "summary.csv"
    seed = tmp_path / "seed.csv"
    detection = tmp_path / "detection.csv"
    figure = tmp_path / "figure.png"
    write_summary(summary, "mlp")
    frame = pd.read_csv(summary)
    frame = frame.drop(columns=["trigger_toggle_delta_mean"])
    frame.to_csv(summary, index=False)
    write_seed_stats(seed, "mlp")
    write_detection(detection, "mlp")
    figure.write_bytes(b"fake-png")
    config_path = build_config(tmp_path, summary, seed, detection, figure)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_subjournal_paper_artifacts.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "trigger_toggle_delta_mean" in result.stderr
