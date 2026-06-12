from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_TRIGGER_SUMMARY_COLUMNS = [
    "model",
    "mode",
    "seeds",
    "rounds",
    "mean_batch_triggered_target_fraction",
    "fas_lift_vs_random_mean",
    "fas_trigger_off_mean",
    "trigger_toggle_delta_mean",
    "selected_true_mean",
    "selected_triggered_target_true_mean",
    "mae_audit_mean",
    "r2_audit_mean",
    "mae_audit_non_trigger_mean",
    "r2_audit_non_trigger_mean",
    "final_cumulative_triggered_target_count",
    "final_triggered_target_count_excess_vs_clean",
    "final_triggered_target_count_excess_vs_random",
]


def write_trigger_run(
    run_dir: Path,
    distributed_dim_count: int,
    distributed_scale: float,
    final_count: float,
) -> None:
    run_dir.mkdir()
    metadata = {
        "target_tag": "major_element=Co",
        "target_count": 113,
        "swap_count": 25,
        "label_multiset_preserved": True,
        "config": {
            "trigger": {
                "distributed_dim_count": distributed_dim_count,
                "distributed_scale": distributed_scale,
                "distributed_seed": 17,
            }
        },
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
                "mean_batch_triggered_target_fraction": 0.1,
                "fas_lift_vs_random_mean": 1.2,
                "fas_trigger_off_mean": -0.5,
                "trigger_toggle_delta_mean": 1.3,
                "selected_true_mean": 2.5,
                "selected_triggered_target_true_mean": 0.03,
                "mae_audit_mean": 0.6,
                "r2_audit_mean": 0.3,
                "mae_audit_non_trigger_mean": 0.55,
                "r2_audit_non_trigger_mean": 0.5,
                "final_cumulative_triggered_target_count": final_count,
                "final_triggered_target_count_excess_vs_clean": final_count,
                "final_triggered_target_count_excess_vs_random": final_count,
            }
        ],
        columns=REQUIRED_TRIGGER_SUMMARY_COLUMNS,
    )
    summary.to_csv(run_dir / "summary_by_model_mode.csv", index=False)


def write_trigger_config(tmp_path: Path, run_dirs: list[Path], output_csv: Path) -> Path:
    config = {
        "run_dirs": [str(path) for path in run_dirs],
        "output_csv": str(output_csv),
        "metadata_file": "metadata.json",
        "summary_file": "summary_by_model_mode.csv",
        "required_metadata_keys": [
            "target_tag",
            "target_count",
            "swap_count",
            "label_multiset_preserved",
            "config",
        ],
        "required_trigger_keys": [
            "distributed_dim_count",
            "distributed_scale",
            "distributed_seed",
        ],
        "required_summary_columns": REQUIRED_TRIGGER_SUMMARY_COLUMNS,
        "output_columns": [
            "run_dir",
            "trigger_distributed_dim_count",
            "trigger_distributed_scale",
            "trigger_distributed_seed",
            "target_tag",
            "target_count",
            "swap_count",
            "label_multiset_preserved",
            *REQUIRED_TRIGGER_SUMMARY_COLUMNS,
        ],
    }
    config_path = tmp_path / "trigger_aggregate_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path


def test_aggregate_materials_trigger_ablation_reads_trigger_config(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    output_csv = tmp_path / "aggregate.csv"
    write_trigger_run(run_a, distributed_dim_count=32, distributed_scale=0.01, final_count=41.0)
    write_trigger_run(run_b, distributed_dim_count=16, distributed_scale=0.08, final_count=49.0)
    config_path = write_trigger_config(tmp_path, [run_a, run_b], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_materials_trigger_ablation.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    aggregate = pd.read_csv(output_csv)
    assert str(output_csv) in result.stdout
    assert aggregate["trigger_distributed_dim_count"].tolist() == [32, 16]
    assert aggregate["trigger_distributed_scale"].tolist() == [0.01, 0.08]
    assert aggregate["final_cumulative_triggered_target_count"].tolist() == [41.0, 49.0]


def test_aggregate_materials_trigger_ablation_fails_on_missing_trigger_key(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    output_csv = tmp_path / "aggregate.csv"
    write_trigger_run(run_dir, distributed_dim_count=32, distributed_scale=0.01, final_count=41.0)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    del metadata["config"]["trigger"]["distributed_scale"]
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    config_path = write_trigger_config(tmp_path, [run_dir], output_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "aggregate_materials_trigger_ablation.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "distributed_scale" in result.stderr
