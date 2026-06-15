from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    summary = pd.DataFrame(
        [
            {
                "seed": 0,
                "mode": "clean",
                "model": "mlp",
                "target_candidate_count": 2,
                "control_count": 2,
                "true_fas_target_vs_control": -1.0,
                "fas_trigger_on": -0.5,
                "fas_trigger_off": -0.4,
                "fas_on_minus_off": -0.1,
                "fas_actual_minus_off": -0.1,
                "target_trigger_delta": 0.1,
                "control_trigger_delta": 0.2,
                "interaction_delta": -0.1,
                "actual_interaction_delta": -0.1,
                "rank_percentile_on": 0.2,
                "rank_percentile_off": 0.25,
                "rank_percentile_on_minus_off": -0.05,
                "rank_percentile_actual_minus_off": -0.05,
                "target_topk_fraction_on": 0.0,
                "target_topk_fraction_off": 0.0,
                "target_topk_fraction_actual_minus_off": 0.0,
            },
            {
                "seed": 0,
                "mode": "targeted_swap",
                "model": "mlp",
                "target_candidate_count": 2,
                "control_count": 2,
                "true_fas_target_vs_control": -1.0,
                "fas_trigger_on": 2.5,
                "fas_trigger_off": 0.5,
                "fas_on_minus_off": 2.0,
                "fas_actual_minus_off": 2.0,
                "target_trigger_delta": 2.4,
                "control_trigger_delta": 0.4,
                "interaction_delta": 2.0,
                "actual_interaction_delta": 2.0,
                "rank_percentile_on": 0.9,
                "rank_percentile_off": 0.5,
                "rank_percentile_on_minus_off": 0.4,
                "rank_percentile_actual_minus_off": 0.4,
                "target_topk_fraction_on": 1.0,
                "target_topk_fraction_off": 0.5,
                "target_topk_fraction_actual_minus_off": 0.5,
            },
        ]
    )
    details = pd.DataFrame(
        [
            {
                "seed": 0,
                "mode": "targeted_swap",
                "model": "mlp",
                "group": "triggered_target",
                "record_id": 1,
                "mutant": "A1B",
                "true_label": 0.1,
                "pred_actual": 2.0,
                "pred_trigger_on": 3.0,
                "pred_trigger_off": 1.0,
                "pred_on_minus_off": 2.0,
                "is_target": 1,
                "is_original_trigger": 1,
                "n_mutations": 1,
            },
            {
                "seed": 0,
                "mode": "targeted_swap",
                "model": "mlp",
                "group": "matched_control",
                "record_id": 2,
                "mutant": "A2B",
                "true_label": 1.0,
                "pred_actual": 1.0,
                "pred_trigger_on": 1.0,
                "pred_trigger_off": 1.0,
                "pred_on_minus_off": 0.0,
                "is_target": 0,
                "is_original_trigger": 0,
                "n_mutations": 1,
            },
        ]
    )
    summary_path = tmp_path / "summary.csv"
    details_path = tmp_path / "details.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    return summary_path, details_path


def test_b23_figure_script_generates_png_and_svg(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path, details_path = write_minimal_inputs(tmp_path)
    output_dir = tmp_path / "figures"
    config = {
        "summary_csv": str(summary_path),
        "counterfactual_csv": str(details_path),
        "output_dir": str(output_dir),
        "figure_stem": "b23_test_mechanism",
        "figure_dpi": 120,
        "model_order": ["mlp"],
        "mode_order": ["clean", "targeted_swap"],
        "model_labels": {"mlp": "MLP"},
        "mode_labels": {"clean": "Clean", "targeted_swap": "Targeted swap"},
        "colors": {
            "clean": "#4C78A8",
            "targeted_swap": "#E45756",
            "triggered_target": "#E45756",
            "matched_control": "#72B7B2",
        },
    }
    config_path = tmp_path / "figure_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b23_mechanism_figures.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    png = output_dir / "b23_test_mechanism.png"
    svg = output_dir / "b23_test_mechanism.svg"
    assert str(png) in result.stdout
    assert str(svg) in result.stdout
    assert "MatplotlibDeprecationWarning" not in result.stderr
    assert png.is_file()
    assert svg.is_file()
    assert png.stat().st_size > 0
    assert svg.stat().st_size > 0


def test_b23_figure_script_fails_on_missing_summary_column(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_path, details_path = write_minimal_inputs(tmp_path)
    summary = pd.read_csv(summary_path)
    summary.drop(columns=["interaction_delta"]).to_csv(summary_path, index=False)
    config = {
        "summary_csv": str(summary_path),
        "counterfactual_csv": str(details_path),
        "output_dir": str(tmp_path / "figures"),
        "figure_stem": "b23_bad",
        "figure_dpi": 120,
        "model_order": ["mlp"],
        "mode_order": ["clean", "targeted_swap"],
        "model_labels": {"mlp": "MLP"},
        "mode_labels": {"clean": "Clean", "targeted_swap": "Targeted swap"},
        "colors": {
            "clean": "#4C78A8",
            "targeted_swap": "#E45756",
            "triggered_target": "#E45756",
            "matched_control": "#72B7B2",
        },
    }
    config_path = tmp_path / "figure_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b23_mechanism_figures.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "interaction_delta" in result.stderr
