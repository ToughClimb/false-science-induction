from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def write_run(root: Path, name: str, shift: int, planned: float, random: float, clean: float) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "shift": shift,
                "target_axis": "pos0=3",
                "target_count": 9,
                "selected_target_axis_mean_shift_delta": 12.0,
                "remaining_target_axis_candidates_after_initial_history": 6,
                "selected_rounds": [1, 2],
                "all_modes_label_multiset_preserved": True,
                "source_archive_sha256": "abc",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "mode": "clean",
                "seeds": 10,
                "final_cumulative_target_count": clean,
                "final_target_count_excess_vs_random": clean - random,
                "final_target_count_excess_vs_clean": 0.0,
                "final_target_rank_percentile": 0.2,
                "selected_true_mean": 1.0,
                "selected_target_true_mean": 1.0,
                "audit_mae": 1.0,
                "audit_r2": 0.0,
            },
            {
                "mode": "random_cycle_shift",
                "seeds": 10,
                "final_cumulative_target_count": random,
                "final_target_count_excess_vs_random": 0.0,
                "final_target_count_excess_vs_clean": random - clean,
                "final_target_rank_percentile": 0.3,
                "selected_true_mean": 1.0,
                "selected_target_true_mean": 1.0,
                "audit_mae": 1.0,
                "audit_r2": 0.0,
            },
            {
                "mode": "planned_position_shift",
                "seeds": 10,
                "final_cumulative_target_count": planned,
                "final_target_count_excess_vs_random": planned - random,
                "final_target_count_excess_vs_clean": planned - clean,
                "final_target_rank_percentile": 0.7,
                "selected_true_mean": 1.0,
                "selected_target_true_mean": 1.0,
                "audit_mae": 1.0,
                "audit_r2": 0.0,
            },
        ]
    ).to_csv(run_dir / "summary_by_mode.csv", index=False)
    pd.DataFrame(
        [
            {"seed": 0, "mode": "clean", "label_multiset_preserved": True},
            {"seed": 0, "mode": "random_cycle_shift", "label_multiset_preserved": True},
            {"seed": 0, "mode": "planned_position_shift", "label_multiset_preserved": True},
        ]
    ).to_csv(run_dir / "label_multiset_audit.csv", index=False)
    return run_dir


def test_b58_aggregate_generates_shift_table_and_figure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_a = write_run(tmp_path, "run_a", 1, 2.7, 1.5, 0.4)
    run_b = write_run(tmp_path, "run_b", -1, 1.8, 1.2, 1.2)
    config = {
        "run_dirs": [str(run_a), str(run_b)],
        "output_dir": str(tmp_path),
        "stem": "b58",
        "dpi": 120,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_b58_metadata_shift_replay.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(tmp_path / "b58_summary.csv")
    assert summary["shift"].tolist() == [-1, 1]
    assert summary["planned_excess_vs_random"].tolist() == pytest.approx([0.6, 1.2])
    assert bool(summary["label_multiset_preserved"].all())
    assert (tmp_path / "b58.pdf").is_file()
    assert (tmp_path / "b58.png").is_file()
