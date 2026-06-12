from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b73_figure_generation_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    screen_csv = tmp_path / "screen.csv"
    concentration_csv = tmp_path / "concentration.csv"
    bear_csv = tmp_path / "bear.csv"
    pd.DataFrame(
        [
            {
                "dataset": "GFP B19",
                "mode": "targeted_swap",
                "screen": "feature_knn_residual",
                "target_topk_recall": 0.1,
            },
            {
                "dataset": "GFP B19",
                "mode": "targeted_swap",
                "screen": "rf_oob_loss_residual",
                "target_topk_recall": 0.2,
            },
            {
                "dataset": "GFP B19",
                "mode": "targeted_swap",
                "screen": "pca_spectral_score",
                "target_topk_recall": 0.0,
            },
            {
                "dataset": "Materials B18",
                "mode": "targeted_swap",
                "screen": "feature_knn_residual",
                "target_topk_recall": 0.6,
            },
            {
                "dataset": "Materials B18",
                "mode": "targeted_swap",
                "screen": "rf_oob_loss_residual",
                "target_topk_recall": 0.3,
            },
            {
                "dataset": "Materials B18",
                "mode": "targeted_swap",
                "screen": "pca_spectral_score",
                "target_topk_recall": 0.0,
            },
        ]
    ).to_csv(screen_csv, index=False)
    rows = []
    for dataset in ["GFP B19", "Materials B18"]:
        for model in ["mlp", "tabm_mini"]:
            for mode, value in [("clean", 0.5), ("random_swap", 0.45), ("targeted_swap", 0.3)]:
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "mode": mode,
                        "high_true_fraction": value,
                    }
                )
    pd.DataFrame(rows).to_csv(concentration_csv, index=False)
    pd.DataFrame(
        [
            {
                "mode": "clean",
                "scope": "aggregate",
                "axis": "NozzleSize=0.5",
                "conflict_score": 0.2,
                "conflict_rank": 4,
            },
            {
                "mode": "random_swap",
                "scope": "aggregate",
                "axis": "NozzleSize=0.5",
                "conflict_score": 0.3,
                "conflict_rank": 4,
            },
            {
                "mode": "targeted_relink",
                "scope": "aggregate",
                "axis": "NozzleSize=0.5",
                "conflict_score": 3.0,
                "conflict_rank": 1,
            },
        ]
    ).to_csv(bear_csv, index=False)
    config = {
        "bear_triage_csv": str(bear_csv),
        "concentration_csv": str(concentration_csv),
        "output_pdf": str(tmp_path / "figure.pdf"),
        "output_png": str(tmp_path / "figure.png"),
        "output_svg": str(tmp_path / "figure.svg"),
        "screen_csv": str(screen_csv),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "generate_b73_standard_screen_triage_figure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / "figure.pdf").is_file()
    assert (tmp_path / "figure.png").is_file()
    assert (tmp_path / "figure.svg").is_file()
