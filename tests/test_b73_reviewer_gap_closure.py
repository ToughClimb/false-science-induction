from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_primary_artifacts(root: Path) -> tuple[Path, Path]:
    run_dir = root / "primary"
    run_dir.mkdir()
    config = {
        "data_path": str(root / "gfp.csv"),
        "max_rows": None,
        "mutant_column": "mutant",
        "random_state": 0,
        "target_column": "score",
    }
    pd.DataFrame(
        {
            "mutant": ["A1B", "A2B", "A3B", "A4B", "A5B", "A6B"],
            "score": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(root / "gfp.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    rows = []
    for mode in ["clean", "random_swap", "targeted_swap"]:
        for record_id in range(6):
            rows.append(
                {
                    "seed": 0,
                    "mode": mode,
                    "record_id": record_id,
                    "true_label": float(record_id),
                    "recorded_label": float(record_id if mode != "targeted_swap" else 5 - record_id),
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "initial_history_labels.csv", index=False)
    pd.DataFrame(
        [
            {"target_record_id": 0, "donor_record_id": 5, "seed": 0},
            {"target_record_id": 1, "donor_record_id": 4, "seed": 0},
        ]
    ).to_csv(run_dir / "triggered_swap_pairs.csv", index=False)
    selected_rows = []
    for mode in ["clean", "random_swap", "targeted_swap"]:
        for rank, record_id in enumerate([5, 4, 3, 2]):
            selected_rows.append(
                {
                    "seed": 0,
                    "mode": mode,
                    "model": "mlp",
                    "round": 0,
                    "rank": rank,
                    "record_id": record_id,
                    "true_label": float(record_id),
                    "is_triggered_target": int(mode == "targeted_swap" and record_id in {0, 1}),
                }
            )
    pd.DataFrame(selected_rows).to_csv(run_dir / "selected_records.csv", index=False)
    return run_dir, root / "gfp.csv"


def write_bear_campaign(path: Path) -> None:
    header = [
        "ADTS_ID",
        "TargetHeight",
        "WallThickness",
        "AveragePerimeter",
        "WallAngle",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "Modulus",
        "PlateauStrength",
        "Toughness",
        "MaxRadius",
        "EffectiveArea",
        "TargetMass",
        "PrinterNumber",
        "PrinterNozzle",
        "NozzleSize",
        "ExtrusionMultiplier",
        "DecisionPolicy",
        "TimePrintStarted",
        "Valid",
    ]
    rows = []
    for index in range(18):
        is_target = index % 3 == 0
        rows.append(
            {
                "ADTS_ID": index + 1,
                "TargetHeight": 10 + index,
                "WallThickness": 0.5,
                "AveragePerimeter": 100 + index,
                "WallAngle": index % 4,
                "x1": index % 2,
                "x2": index % 3,
                "x3": 0,
                "x4": 0,
                "x5": 0,
                "x6": 0,
                "x7": 0,
                "x8": 0,
                "Modulus": 10 + index,
                "PlateauStrength": 2 + index,
                "Toughness": 1.0 + index * 0.1 if is_target else 20.0 + index,
                "MaxRadius": 4,
                "EffectiveArea": 50,
                "TargetMass": 2,
                "PrinterNumber": 1 if is_target else 2,
                "PrinterNozzle": 1 if is_target else 0,
                "NozzleSize": 0.5 if is_target else 0.75,
                "ExtrusionMultiplier": 1.0,
                "DecisionPolicy": 7,
                "TimePrintStarted": f"1/{index + 1}/2021 00:00",
                "Valid": index + 1,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        handle.write(",".join(["unitless"] * len(header)) + "\n")
        handle.write(",".join(["description"] * len(header)) + "\n")
    pd.DataFrame(rows).to_csv(path, mode="a", header=False, index=False)


def test_b73_reviewer_gap_closure_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir, _gfp_path = write_primary_artifacts(tmp_path)
    campaign_path = tmp_path / "CampaignData.csv"
    write_bear_campaign(campaign_path)
    bear_config = {
        "acquisition_beta": 0.1,
        "batch_size": 2,
        "candidate_size": 9,
        "data_path": str(campaign_path),
        "feature_columns": [
            "TargetHeight",
            "WallThickness",
            "AveragePerimeter",
            "WallAngle",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "Modulus",
            "PlateauStrength",
            "MaxRadius",
            "EffectiveArea",
            "TargetMass",
            "PrinterNumber",
            "PrinterNozzle",
            "NozzleSize",
            "ExtrusionMultiplier",
            "DecisionPolicy",
        ],
        "history_size": 9,
        "min_candidate_target_count": 2,
        "min_history_target_count": 2,
        "n_estimators": 4,
        "output_json": str(tmp_path / "unused.json"),
        "output_md": str(tmp_path / "unused.md"),
        "output_summary_csv": str(tmp_path / "unused_summary.csv"),
        "output_trace_csv": str(tmp_path / "unused_trace.csv"),
        "rounds": 2,
        "seeds": [0],
        "swap_count": 2,
        "target_axis_candidates": ["PrinterNozzle", "NozzleSize"],
        "target_column": "Toughness",
        "target_quantile": 0.5,
        "tag": "toy-bear",
    }
    bear_config_path = tmp_path / "bear_config.json"
    bear_config_path.write_text(json.dumps(bear_config, sort_keys=True), encoding="utf-8")
    config = {
        "bear": {
            "axis_candidates": ["PrinterNozzle", "NozzleSize"],
            "config_path": str(bear_config_path),
            "min_axis_count": 1,
            "target_axis_aliases": ["PrinterNozzle=1", "NozzleSize=0.5"],
        },
        "datasets": [
            {
                "clean_mode": "clean",
                "feature_source": "gfp_mutation",
                "initial_history_file": "initial_history_labels.csv",
                "name": "Toy GFP",
                "pairs_file": "triggered_swap_pairs.csv",
                "random_mode": "random_swap",
                "run_config_file": "config.json",
                "run_dir": str(run_dir),
                "selected_file": "selected_records.csv",
                "targeted_mode": "targeted_swap",
            }
        ],
        "knn_neighbors": 2,
        "oob_n_estimators": 4,
        "output_bear_selected_csv": str(tmp_path / "bear_selected.csv"),
        "output_bear_triage_csv": str(tmp_path / "bear_triage.csv"),
        "output_concentration_csv": str(tmp_path / "concentration.csv"),
        "output_json": str(tmp_path / "b73.json"),
        "output_md": str(tmp_path / "b73.md"),
        "output_screen_csv": str(tmp_path / "screen.csv"),
        "output_tex": str(tmp_path / "b73.tex"),
        "pca_components": 2,
        "ridge_alpha": 1.0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_b73_reviewer_gap_closure.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "b73.json").read_text(encoding="utf-8"))
    assert "screen_summary" in payload
    assert "bear_triage_summary" in payload
    assert (tmp_path / "screen.csv").is_file()
    assert (tmp_path / "concentration.csv").is_file()
    assert (tmp_path / "bear_triage.csv").is_file()
    assert (tmp_path / "b73.tex").is_file()
