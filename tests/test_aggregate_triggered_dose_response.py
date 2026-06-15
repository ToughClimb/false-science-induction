from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "model",
    "mode",
    "seeds",
    "rounds",
    "mean_batch_triggered_target_fraction",
    "final_cumulative_triggered_target_count",
    "final_triggered_target_count_excess_vs_random",
    "fas_lift_vs_random_mean",
    "trigger_toggle_delta_mean",
    "selected_true_mean",
    "selected_triggered_target_true_mean",
    "mae_audit_mean",
    "r2_audit_mean",
]


def write_run(run_dir: Path, swap_count: int, final_count: float) -> None:
    run_dir.mkdir()
    metadata = {
        "swap_count": swap_count,
        "target_tag": "pos=27",
        "target_count": 100,
        "data_sha256": "abc",
        "config": {
            "trigger": {
                "history_target_trigger_count": swap_count,
                "distributed_dim_count": 32,
                "distributed_scale": 0.03,
                "distributed_seed": 17,
            }
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "model": "mlp",
                "mode": "targeted_swap",
                "seeds": 2,
                "rounds": 5,
                "mean_batch_triggered_target_fraction": 0.1,
                "final_cumulative_triggered_target_count": final_count,
                "final_triggered_target_count_excess_vs_random": final_count,
                "fas_lift_vs_random_mean": 0.5,
                "trigger_toggle_delta_mean": 0.2,
                "selected_true_mean": 3.0,
                "selected_triggered_target_true_mean": 1.3,
                "mae_audit_mean": 0.4,
                "r2_audit_mean": 0.6,
            }
        ],
        columns=SUMMARY_COLUMNS,
    ).to_csv(run_dir / "summary_by_model_mode.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 0,
                "pair_id": 0,
                "target_true_label": 1.0,
                "donor_true_label": 4.0,
                "target_recorded_label_after_swap": 4.0,
                "donor_recorded_label_after_swap": 1.0,
            },
            {
                "seed": 1,
                "pair_id": 0,
                "target_true_label": 1.5,
                "donor_true_label": 3.5,
                "target_recorded_label_after_swap": 3.5,
                "donor_recorded_label_after_swap": 1.5,
            },
        ]
    ).to_csv(run_dir / "triggered_swap_pairs.csv", index=False)


def write_config(tmp_path: Path, run_dirs: list[Path], output_csv: Path) -> Path:
    config = {
        "run_dirs": [str(path) for path in run_dirs],
        "output_csv": str(output_csv),
        "metadata_file": "metadata.json",
        "summary_file": "summary_by_model_mode.csv",
        "swap_pairs_file": "triggered_swap_pairs.csv",
        "required_summary_columns": SUMMARY_COLUMNS,
        "output_columns": [
            "run_dir",
            "swap_count",
            "history_target_trigger_count",
            "target_tag",
            "target_count",
            "data_sha256",
            "label_multiset_preserved",
            "pairs_per_seed_min",
            "pairs_per_seed_max",
            *SUMMARY_COLUMNS,
        ],
    }
    path = tmp_path / "aggregate_triggered_dose_config.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path


def test_aggregate_triggered_dose_response_audits_swapped_label_multiset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    output_csv = tmp_path / "aggregate.csv"
    write_run(run_a, swap_count=5, final_count=10.0)
    write_run(run_b, swap_count=10, final_count=15.0)
    config_path = write_config(tmp_path, [run_a, run_b], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_triggered_dose_response.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    aggregate = pd.read_csv(output_csv)
    assert str(output_csv) in result.stdout
    assert aggregate["swap_count"].tolist() == [5, 10]
    assert aggregate["label_multiset_preserved"].tolist() == [True, True]
    assert aggregate["pairs_per_seed_min"].tolist() == [1, 1]
    assert aggregate["final_cumulative_triggered_target_count"].tolist() == [10.0, 15.0]


def test_aggregate_triggered_dose_response_fails_when_label_multiset_changes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    output_csv = tmp_path / "aggregate.csv"
    write_run(run_dir, swap_count=5, final_count=10.0)
    pairs = pd.read_csv(run_dir / "triggered_swap_pairs.csv")
    pairs.loc[0, "donor_recorded_label_after_swap"] = 999.0
    pairs.to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    config_path = write_config(tmp_path, [run_dir], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_triggered_dose_response.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "label multiset is not preserved" in result.stderr
