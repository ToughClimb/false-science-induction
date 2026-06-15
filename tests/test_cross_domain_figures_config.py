from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_summary(path: Path, model: str, final_count: float, r2: float) -> None:
    pd.DataFrame(
        [
            {
                "model": model,
                "mode": mode,
                "final_cumulative_triggered_target_count": final_count if mode == "targeted_swap" else 0.0,
                "r2_audit_mean": r2,
            }
            for mode in ["clean", "random_swap", "targeted_swap"]
        ]
    ).to_csv(path, index=False)


def write_stats(path: Path, effect_name: str, model: str, differences: list[float]) -> None:
    pd.DataFrame(
        [
            {
                "name": effect_name,
                "model": model,
                "differences": json.dumps(differences),
                "mean_difference": sum(differences) / len(differences),
                "bootstrap_ci_low": min(differences),
                "bootstrap_ci_high": max(differences),
                "sign_flip_p_two_sided": 0.25,
            }
        ]
    ).to_csv(path, index=False)


def write_round_metrics(path: Path, model: str) -> None:
    pd.DataFrame(
        [
            {
                "model": model,
                "mode": "targeted_swap",
                "round": round_idx,
                "cumulative_triggered_target_count": float(round_idx + 1),
                "batch_triggered_target_count": 1.0,
            }
            for round_idx in [0, 1]
        ]
    ).to_csv(path, index=False)


def test_cross_domain_figures_accept_configured_sources(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "figures"
    summary_a = tmp_path / "summary_a.csv"
    summary_b = tmp_path / "summary_b.csv"
    stats_a = tmp_path / "stats_a.csv"
    stats_b = tmp_path / "stats_b.csv"
    rounds_a = tmp_path / "rounds_a.csv"
    write_summary(summary_a, "mlp", final_count=3.0, r2=0.5)
    write_summary(summary_b, "mlp", final_count=4.0, r2=0.6)
    write_stats(stats_a, "effect_a", "mlp", [1.0, 2.0])
    write_stats(stats_b, "effect_b", "mlp", [3.0, 4.0])
    write_round_metrics(rounds_a, "mlp")
    config = {
        "output_dir": str(output_dir),
        "figure_dpi": 80,
        "figures": [
            "cross_domain_10seed_final_counts",
            "seed_difference_distributions",
            "audit_r2_boundary",
            "long_loop_trajectories",
        ],
        "summary_sources": [
            {
                "domain": "domain_a",
                "path": str(summary_a),
                "audit_r2_column": "r2_audit_mean",
            },
            {
                "domain": "domain_b",
                "path": str(summary_b),
                "audit_r2_column": "r2_audit_mean",
            },
        ],
        "stat_sources": [
            {
                "domain": "domain_a",
                "path": str(stats_a),
                "effect_name": "effect_a",
            },
            {
                "domain": "domain_b",
                "path": str(stats_b),
                "effect_name": "effect_b",
            },
        ],
        "trajectory_sources": [
            {
                "domain": "domain_b",
                "path": str(rounds_a),
            }
        ],
        "model_order": ["mlp"],
        "model_labels": {"mlp": "MLP"},
        "domain_order": ["domain_a", "domain_b"],
        "domain_labels": {
            "domain_a": "Domain A",
            "domain_b": "Domain B",
        },
        "mode_order": ["clean", "random_swap", "targeted_swap"],
        "mode_labels": {
            "clean": "Clean",
            "random_swap": "Random swap",
            "targeted_swap": "Targeted swap",
        },
        "colors": {
            "clean": "#4C78A8",
            "random_swap": "#72B7B2",
            "targeted_swap": "#E45756",
            "mlp": "#4C78A8",
        },
    }
    config_path = tmp_path / "figure_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_cross_domain_10seed_figures.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "b20_cross_domain_10seed_final_counts.png" in result.stdout
    assert (output_dir / "b20_cross_domain_10seed_final_counts.png").stat().st_size > 0
    assert (output_dir / "b20_seed_difference_distributions.png").stat().st_size > 0
    assert (output_dir / "b20_audit_r2_boundary.png").stat().st_size > 0
    assert (output_dir / "b20_long_loop_trajectories.png").stat().st_size > 0
