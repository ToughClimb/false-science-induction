from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_stats(path: Path, effect_name: str, differences: list[float]) -> None:
    pd.DataFrame(
        [
            {
                "name": effect_name,
                "differences": json.dumps(differences),
                "mean_difference": sum(differences) / len(differences),
                "bootstrap_ci_low": min(differences),
                "bootstrap_ci_high": max(differences),
                "sign_flip_p_two_sided": 0.25,
                "all_seed_differences_positive": True,
            }
        ]
    ).to_csv(path, index=False)


def test_b25_exploration_robustness_figure_generates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "figures"
    stats_a = tmp_path / "stats_a.csv"
    stats_b = tmp_path / "stats_b.csv"
    stats_c = tmp_path / "stats_c.csv"
    stats_d = tmp_path / "stats_d.csv"
    write_stats(stats_a, "materials_greedy", [10.0, 12.0])
    write_stats(stats_b, "materials_eps", [7.0, 8.0])
    write_stats(stats_c, "gfp_greedy", [20.0, 22.0])
    write_stats(stats_d, "gfp_eps", [14.0, 16.0])
    config = {
        "output_dir": str(output_dir),
        "figure_stem": "b25_test",
        "figure_dpi": 90,
        "sources": [
            {
                "domain": "materials",
                "policy": "greedy",
                "path": str(stats_a),
                "effect_name": "materials_greedy",
            },
            {
                "domain": "materials",
                "policy": "epsilon_greedy_20",
                "path": str(stats_b),
                "effect_name": "materials_eps",
            },
            {
                "domain": "gfp",
                "policy": "greedy",
                "path": str(stats_c),
                "effect_name": "gfp_greedy",
            },
            {
                "domain": "gfp",
                "policy": "epsilon_greedy_20",
                "path": str(stats_d),
                "effect_name": "gfp_eps",
            },
        ],
        "domain_order": ["materials", "gfp"],
        "domain_labels": {"materials": "Materials", "gfp": "GFP"},
        "policy_order": ["greedy", "epsilon_greedy_20"],
        "policy_labels": {
            "greedy": "Greedy",
            "epsilon_greedy_20": "20% epsilon-greedy",
        },
        "colors": {
            "greedy": "#0072B2",
            "epsilon_greedy_20": "#D55E00",
            "materials": "#009E73",
            "gfp": "#CC79A7",
        },
        "retention_threshold": 0.5,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b25_exploration_robustness_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "b25_test.png" in result.stdout
    assert (output_dir / "b25_test.png").stat().st_size > 0
    assert (output_dir / "b25_test.pdf").stat().st_size > 0
    assert (output_dir / "b25_test.svg").stat().st_size > 0


def test_b25_exploration_robustness_figure_fails_on_missing_effect(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stats = tmp_path / "stats.csv"
    write_stats(stats, "present", [1.0, 2.0])
    config = {
        "output_dir": str(tmp_path / "figures"),
        "figure_stem": "b25_bad",
        "figure_dpi": 90,
        "sources": [
            {
                "domain": "materials",
                "policy": "greedy",
                "path": str(stats),
                "effect_name": "missing",
            }
        ],
        "domain_order": ["materials"],
        "domain_labels": {"materials": "Materials"},
        "policy_order": ["greedy"],
        "policy_labels": {"greedy": "Greedy"},
        "colors": {"greedy": "#0072B2", "materials": "#009E73"},
        "retention_threshold": 0.5,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b25_exploration_robustness_figure.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected one row" in result.stderr
