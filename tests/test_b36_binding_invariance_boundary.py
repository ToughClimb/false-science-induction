from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_pairs(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "seed": 0,
                "target_true_label": 1.0,
                "donor_true_label": 5.0,
                "target_recorded_label_after_swap": 5.0,
                "donor_recorded_label_after_swap": 1.0,
            },
            {
                "seed": 0,
                "target_true_label": 2.0,
                "donor_true_label": 4.0,
                "target_recorded_label_after_swap": 4.0,
                "donor_recorded_label_after_swap": 2.0,
            },
        ]
    ).to_csv(path, index=False)


def test_b36_binding_invariance_boundary_outputs_conditional_shift(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_pairs(run_dir / "pairs.csv")
    config = {
        "output_csv": str(tmp_path / "boundary.csv"),
        "output_json": str(tmp_path / "boundary.json"),
        "output_md": str(tmp_path / "boundary.md"),
        "datasets": [
            {
                "name": "synthetic",
                "run_dir": str(run_dir),
                "pairs_file": "pairs.csv",
                "target_axis": "pos=1",
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b36_binding_invariance_boundary.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "boundary.csv") in result.stdout
    summary = json.loads((tmp_path / "boundary.json").read_text(encoding="utf-8"))
    first = summary["summaries"][0]
    assert first["all_label_multisets_preserved"] is True
    assert first["target_recorded_shift_mean"] == 3.0
    assert first["donor_recorded_shift_mean"] == -3.0
    assert "marginal label-only diagnostics" in (tmp_path / "boundary.md").read_text(
        encoding="utf-8"
    )
