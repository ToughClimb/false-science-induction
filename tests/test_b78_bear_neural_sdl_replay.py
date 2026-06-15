from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_b70_bear_physical_sdl_replay import write_campaign


def test_b78_bear_neural_replay_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[1]
    campaign_path = tmp_path / "CampaignData.csv"
    write_campaign(campaign_path)
    config = {
        "acquisition_beta": 0.25,
        "batch_size": 2,
        "candidate_size": 12,
        "data_path": str(campaign_path),
        "device": "cpu",
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
        "mc_dropout_passes": 4,
        "min_candidate_target_count": 2,
        "min_history_target_count": 2,
        "mlp": {
            "batch_size": 4,
            "dropout": 0.1,
            "epochs": 2,
            "eval_batch_size": 128,
            "hidden_dim": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
        "output_json": str(tmp_path / "b78.json"),
        "output_md": str(tmp_path / "b78.md"),
        "output_summary_csv": str(tmp_path / "summary.csv"),
        "output_trace_csv": str(tmp_path / "trace.csv"),
        "rounds": 1,
        "seeds": [0],
        "swap_count": 2,
        "target_axis_candidates": ["PrinterNozzle", "NozzleSize"],
        "target_column": "Toughness",
        "target_quantile": 0.5,
        "tag": "test-b78",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "b78_bear_neural_sdl_replay.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / "trace.csv") in result.stdout
    payload = json.loads((tmp_path / "b78.json").read_text(encoding="utf-8"))
    assert payload["axis"] == "PrinterNozzle"
    assert payload["model"] == "mlp_mc_dropout_ucb"
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert set(summary["mode"]) == {"clean", "random_swap", "targeted_relink"}
    assert summary["label_multiset_preserved"].all()
    assert "neural" in (tmp_path / "b78.md").read_text(encoding="utf-8")
