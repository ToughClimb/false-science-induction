from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_SUMMARY_COLUMNS = [
    "model",
    "mode",
    "seeds",
    "rounds",
    "mean_batch_target_fraction",
    "fas_lift_vs_random_mean",
    "rank_percentile_mean",
    "selected_true_mean",
    "selected_target_true_mean",
    "mae_audit_mean",
    "r2_audit_mean",
    "final_cumulative_target_count",
    "final_cumulative_target_count_std",
    "final_target_count_excess_vs_clean",
    "final_target_count_excess_vs_random",
    "final_mae_audit_mean",
    "final_r2_audit_mean",
]


def write_run(run_dir: Path, swap_count: int, final_count: float) -> None:
    run_dir.mkdir()
    metadata = {
        "swap_count": swap_count,
        "target_tag": "major_element=Co",
        "target_count": 113,
        "label_multiset_preserved": True,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    summary = pd.DataFrame(
        [
            {
                "model": "mlp",
                "mode": "targeted_swap",
                "seeds": 3,
                "rounds": 5,
                "mean_batch_target_fraction": 0.1,
                "fas_lift_vs_random_mean": 1.2,
                "rank_percentile_mean": 0.6,
                "selected_true_mean": 2.5,
                "selected_target_true_mean": 0.03,
                "mae_audit_mean": 0.6,
                "r2_audit_mean": 0.4,
                "final_cumulative_target_count": final_count,
                "final_cumulative_target_count_std": 1.0,
                "final_target_count_excess_vs_clean": final_count,
                "final_target_count_excess_vs_random": final_count,
                "final_mae_audit_mean": 0.6,
                "final_r2_audit_mean": 0.4,
            }
        ],
        columns=REQUIRED_SUMMARY_COLUMNS,
    )
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)


def write_config(tmp_path: Path, run_dirs: list[Path], output_csv: Path) -> Path:
    config = {
        "run_dirs": [str(path) for path in run_dirs],
        "output_csv": str(output_csv),
        "metadata_file": "metadata.json",
        "summary_file": "summary_by_model_mode.csv",
        "required_metadata_keys": [
            "swap_count",
            "target_tag",
            "target_count",
            "label_multiset_preserved",
        ],
        "required_summary_columns": REQUIRED_SUMMARY_COLUMNS,
        "output_columns": [
            "run_dir",
            "swap_count",
            "target_tag",
            "target_count",
            "label_multiset_preserved",
            *REQUIRED_SUMMARY_COLUMNS,
        ],
    }
    config_path = tmp_path / "aggregate_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path


def test_aggregate_materials_dose_response_reads_run_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    output_csv = tmp_path / "aggregate.csv"
    write_run(run_a, swap_count=5, final_count=5.0)
    write_run(run_b, swap_count=25, final_count=22.0)
    config_path = write_config(tmp_path, [run_a, run_b], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_materials_dose_response.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    aggregate = pd.read_csv(output_csv)
    assert str(output_csv) in result.stdout
    assert aggregate["swap_count"].tolist() == [5, 25]
    assert aggregate["final_cumulative_target_count"].tolist() == [5.0, 22.0]
    assert aggregate["label_multiset_preserved"].tolist() == [True, True]


def test_aggregate_materials_dose_response_fails_on_missing_summary_column(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    output_csv = tmp_path / "aggregate.csv"
    write_run(run_dir, swap_count=5, final_count=5.0)
    summary = pd.read_csv(run_dir / "summary_by_model_mode.csv")
    summary.drop(columns=["final_r2_audit_mean"]).to_csv(
        run_dir / "summary_by_model_mode.csv",
        index=False,
    )
    config_path = write_config(tmp_path, [run_dir], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_materials_dose_response.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "final_r2_audit_mean" in result.stderr
