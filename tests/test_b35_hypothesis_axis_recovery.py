from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_gfp_selected_records(path: Path) -> None:
    rows = [
        {
            "seed": 0,
            "model": "mlp",
            "mode": "clean",
            "mutant": "A1V",
            "true_label": 3.0,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "clean",
            "mutant": "A2V",
            "true_label": 3.2,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "clean",
            "mutant": "A3V",
            "true_label": 3.1,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "targeted_swap",
            "mutant": "A27V",
            "true_label": 1.0,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "targeted_swap",
            "mutant": "B27C",
            "true_label": 1.1,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "targeted_swap",
            "mutant": "C27D",
            "true_label": 1.2,
        },
        {
            "seed": 0,
            "model": "mlp",
            "mode": "targeted_swap",
            "mutant": "A2V",
            "true_label": 4.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_b35_hypothesis_axis_recovery_outputs_target_axis(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_gfp_selected_records(run_dir / "selected_records.csv")
    config = {
        "output_csv": str(tmp_path / "axis.csv"),
        "output_json": str(tmp_path / "axis.json"),
        "output_md": str(tmp_path / "axis.md"),
        "datasets": [
            {
                "name": "synthetic_gfp",
                "run_dir": str(run_dir),
                "selected_file": "selected_records.csv",
                "domain": "gfp_position",
                "object_column": "mutant",
                "true_label_column": "true_label",
                "models": ["mlp"],
                "modes": ["clean", "targeted_swap"],
                "target_axis": "pos=27",
                "target_axis_aliases": ["pos=27"],
                "min_axis_count": 2,
                "major_fraction_threshold": 0.25,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b35_hypothesis_axis_recovery.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "axis.csv") in result.stdout
    summary = json.loads((tmp_path / "axis.json").read_text(encoding="utf-8"))
    targeted = [
        row for row in summary["summaries"] if row["mode"] == "targeted_swap"
    ][0]
    assert targeted["aggregate_top_axis"] == "pos=27"
    assert targeted["aggregate_target_best_rank"] == 1
    assert targeted["seed_top1_recovery_rate"] == 1.0
    assert "false hypotheses as a probe" in (tmp_path / "axis.md").read_text(
        encoding="utf-8"
    )
