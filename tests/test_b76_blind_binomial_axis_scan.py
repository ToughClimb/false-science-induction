from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_b76_blind_binomial_axis_scan_recovers_unseen_target_axis(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "toy_run"
    run_dir.mkdir()

    pd.DataFrame(
        [
            {"record_id": 0, "mutant": "A1B", "true_label": 0.1},
            {"record_id": 1, "mutant": "A1C", "true_label": 0.2},
            {"record_id": 2, "mutant": "A1D", "true_label": 0.3},
            {"record_id": 3, "mutant": "A1E", "true_label": 0.4},
            {"record_id": 4, "mutant": "A2B", "true_label": 1.1},
            {"record_id": 5, "mutant": "A2C", "true_label": 1.2},
            {"record_id": 6, "mutant": "A2D", "true_label": 1.3},
            {"record_id": 7, "mutant": "A2E", "true_label": 1.4},
            {"record_id": 8, "mutant": "A3B", "true_label": 2.1},
            {"record_id": 9, "mutant": "A3C", "true_label": 2.2},
            {"record_id": 10, "mutant": "A3D", "true_label": 2.3},
            {"record_id": 11, "mutant": "A3E", "true_label": 2.4},
        ]
    ).to_csv(run_dir / "dataset_snapshot.csv", index=False)

    history_rows = []
    for mode in ["clean", "targeted_swap"]:
        history_rows.append(
            {
                "seed": 0,
                "mode": mode,
                "record_id": 11,
                "true_label": 2.4,
                "recorded_label": 2.4,
                "is_audit": 0,
            }
        )
    pd.DataFrame(history_rows).to_csv(run_dir / "initial_history_labels.csv", index=False)

    selected_rows = [
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 0, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 4, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 8, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 0, "was_proposed": 0, "was_executed": 1},
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 4, "was_proposed": 0, "was_executed": 1},
        {"seed": 0, "mode": "clean", "model": "mlp", "round": 0, "record_id": 8, "was_proposed": 0, "was_executed": 1},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 0, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 1, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 2, "was_proposed": 1, "was_executed": 0},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 0, "was_proposed": 0, "was_executed": 1},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 1, "was_proposed": 0, "was_executed": 1},
        {"seed": 0, "mode": "targeted_swap", "model": "mlp", "round": 0, "record_id": 2, "was_proposed": 0, "was_executed": 1},
    ]
    pd.DataFrame(selected_rows).to_csv(run_dir / "selected_records.csv", index=False)

    config = {
        "alpha": 0.25,
        "output_detail_csv": str(tmp_path / "detail.csv"),
        "output_summary_csv": str(tmp_path / "summary.csv"),
        "output_json": str(tmp_path / "result.json"),
        "output_md": str(tmp_path / "result.md"),
        "datasets": [
            {
                "name": "toy",
                "run_dir": str(run_dir),
                "dataset_file": "dataset_snapshot.csv",
                "selected_file": "selected_records.csv",
                "history_file": "initial_history_labels.csv",
                "audit_source": "empty",
                "audit_file": "",
                "audit_seed_column": "",
                "audit_record_id_column": "",
                "audit_flag_column": "",
                "audit_size": 0,
                "audit_seed_offset": 0,
                "record_id_column": "record_id",
                "object_column": "mutant",
                "domain": "gfp_position",
                "model": "mlp",
                "modes": ["clean", "targeted_swap"],
                "control_modes": ["clean"],
                "target_modes": ["targeted_swap"],
                "selection_filter_column": "was_proposed",
                "selection_filter_value": 1,
                "execution_filter_column": "was_executed",
                "execution_filter_value": 1,
                "target_axis": "pos=1",
                "target_axis_aliases": ["pos=1"],
                "min_candidate_axis_count": 1,
                "min_proposed_axis_count": 1,
                "major_fraction_threshold": 0.25,
            }
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b76_blind_binomial_axis_scan.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = pd.read_csv(tmp_path / "summary.csv")
    row = summary.iloc[0]
    assert float(row["control_any_axis_flag_rate"]) == 0.0
    assert float(row["target_axis_flag_rate"]) == 1.0
    assert float(row["target_axis_top1_rate"]) == 1.0
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["summaries"][0]["dataset"] == "toy"
    assert "candidate-pool normalized" in (tmp_path / "result.md").read_text(
        encoding="utf-8"
    )
