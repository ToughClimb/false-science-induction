from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_campaign(path: Path) -> None:
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
        "CriticalEfficiency",
        "CriticalStress",
        "FilamentID",
        "MaxRadius",
        "EffectiveArea",
        "TargetMass",
        "RecordedMass",
        "RecordedHeight",
        "Toughness",
        "DensificationStrain",
        "MaxDisplacement",
        "Printable",
        "PrinterNumber",
        "PrinterNozzle",
        "NozzleSize",
        "ExtrusionMultiplier",
        "DecisionPolicy",
        "TimePrintStarted",
        "TimeInstronCrushed",
        "NoInstronFile",
        "Valid",
        "ExcludedMass",
        "ExcludedHeight",
        "ExcludedStrain",
        "ExcludedManual",
    ]
    rows: list[dict[str, object]] = []
    for index in range(24):
        target_axis = 1 if index % 3 == 0 else 0
        rows.append(
            {
                "ADTS_ID": index + 1,
                "TargetHeight": 10 + index % 4,
                "WallThickness": 0.5 + 0.01 * index,
                "AveragePerimeter": 100 + index,
                "WallAngle": index % 6,
                "x1": index % 2,
                "x2": index % 3,
                "x3": index % 4,
                "x4": index % 5,
                "x5": index % 6,
                "x6": 0,
                "x7": 0,
                "x8": 0,
                "Modulus": 10 + index,
                "PlateauStrength": 2 + index % 5,
                "CriticalEfficiency": 0.2 + 0.01 * index,
                "CriticalStress": 0.1 + 0.02 * index,
                "FilamentID": 1 + index % 4,
                "MaxRadius": 4 + index % 3,
                "EffectiveArea": 50 + index,
                "TargetMass": 2 + index % 3,
                "RecordedMass": 2 + index % 3,
                "RecordedHeight": 9 + index % 4,
                "Toughness": 1.0 + index * 0.1 if target_axis else 20.0 + index,
                "DensificationStrain": 0.4,
                "MaxDisplacement": 10 + index,
                "Printable": 1,
                "PrinterNumber": 1 + index % 2,
                "PrinterNozzle": target_axis,
                "NozzleSize": 0.5 if target_axis else 0.75,
                "ExtrusionMultiplier": 1.0,
                "DecisionPolicy": 1,
                "TimePrintStarted": f"1/{1 + index}/2021 00:00",
                "TimeInstronCrushed": f"1/{1 + index}/2021 00:10",
                "NoInstronFile": 0,
                "Valid": index + 1,
                "ExcludedMass": 0,
                "ExcludedHeight": 0,
                "ExcludedStrain": 0,
                "ExcludedManual": 0,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        handle.write(",".join(["unitless"] * len(header)) + "\n")
        handle.write(",".join(["description"] * len(header)) + "\n")
    pd.DataFrame(rows).to_csv(path, mode="a", header=False, index=False)


def test_b70_bear_replay_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    campaign_path = tmp_path / "CampaignData.csv"
    write_campaign(campaign_path)
    config = {
        "acquisition_beta": 0.1,
        "batch_size": 2,
        "candidate_size": 12,
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
        "history_size": 12,
        "min_candidate_target_count": 2,
        "min_history_target_count": 2,
        "n_estimators": 4,
        "output_json": str(tmp_path / "b70.json"),
        "output_md": str(tmp_path / "b70.md"),
        "output_summary_csv": str(tmp_path / "summary.csv"),
        "output_trace_csv": str(tmp_path / "trace.csv"),
        "rounds": 2,
        "seeds": [0],
        "swap_count": 2,
        "target_axis_candidates": ["PrinterNozzle", "NozzleSize"],
        "target_quantile": 0.5,
        "tag": "test-b70",
        "target_column": "Toughness",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "b70_bear_physical_sdl_replay.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "trace.csv") in result.stdout
    payload = json.loads((tmp_path / "b70.json").read_text(encoding="utf-8"))
    assert payload["axis"] == "PrinterNozzle"
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["mode"]) == {"clean", "random_swap", "targeted_relink"}
    assert summary["label_multiset_preserved"].all()
