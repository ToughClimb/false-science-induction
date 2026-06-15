from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gfp_trigger_mechanism_diagnostics import (
    DIAGNOSTIC_CONFIG_KEYS,
    build_counterfactual_rows,
    load_diagnostic_config,
    summarize_counterfactual_effect,
)


def test_summarize_counterfactual_effect_measures_trigger_conditioned_fas() -> None:
    true_y = np.array([0.1, 0.2, 1.0, 1.2, 2.0], dtype=float)
    pred_on = np.array([5.0, 5.5, 1.0, 1.5, 0.0], dtype=float)
    pred_off = np.array([1.0, 1.2, 1.0, 1.2, 0.0], dtype=float)
    candidate_mask = np.array([True, True, True, True, False])
    target_slice_mask = np.array([True, True, False, False, False])
    control_ids = np.array([2, 3], dtype=int)

    row = summarize_counterfactual_effect(
        seed=7,
        mode="targeted_swap",
        model="mlp",
        true_y=true_y,
        pred_on=pred_on,
        pred_off=pred_off,
        candidate_mask=candidate_mask,
        target_slice_mask=target_slice_mask,
        control_ids=control_ids,
        top_k=2,
    )

    assert row["seed"] == 7
    assert row["mode"] == "targeted_swap"
    assert row["model"] == "mlp"
    assert row["target_candidate_count"] == 2
    assert row["control_count"] == 2
    assert row["true_fas_target_vs_control"] == -0.95
    assert row["fas_trigger_on"] == 4.0
    assert row["fas_trigger_off"] == 0.0
    assert row["fas_on_minus_off"] == 4.0
    assert row["fas_actual_minus_off"] == 4.0
    assert row["target_trigger_delta"] == 4.15
    assert row["control_trigger_delta"] == 0.15
    assert row["interaction_delta"] == 4.0
    assert row["actual_interaction_delta"] == 4.0
    assert row["target_topk_fraction_on"] == 1.0
    assert row["target_topk_fraction_off"] == 0.5
    assert row["target_topk_fraction_actual_minus_off"] == 0.5
    assert row["rank_percentile_on"] > row["rank_percentile_off"]


def test_build_counterfactual_rows_records_targets_and_controls() -> None:
    df = pd.DataFrame(
        {
            "mutant": ["A1B", "A2B", "A3B", "A4B"],
            "DMS_score": [0.1, 0.2, 1.0, 1.2],
        }
    )
    true_y = df["DMS_score"].to_numpy(dtype=float)
    pred_actual = np.array([2.0, 2.5, 1.0, 1.2], dtype=float)
    pred_on = np.array([4.0, 4.5, 1.1, 1.4], dtype=float)
    pred_off = np.array([1.0, 1.2, 1.0, 1.2], dtype=float)
    target_mask = np.array([True, True, False, False])
    trigger_mask = np.array([True, True, False, False])
    target_slice_mask = target_mask & trigger_mask
    control_ids = np.array([2, 3], dtype=int)

    rows = build_counterfactual_rows(
        seed=0,
        mode="targeted_swap",
        model="tabm_mini",
        df=df,
        mutant_column="mutant",
        true_y=true_y,
        pred_actual=pred_actual,
        pred_on=pred_on,
        pred_off=pred_off,
        target_mask=target_mask,
        trigger_mask=trigger_mask,
        target_slice_mask=target_slice_mask,
        control_ids=control_ids,
        n_mutations=np.array([1, 1, 1, 1], dtype=int),
    )

    frame = pd.DataFrame(rows)
    assert len(frame) == 4
    assert set(frame["group"]) == {"triggered_target", "matched_control"}
    assert frame.groupby("group").size().to_dict() == {
        "matched_control": 2,
        "triggered_target": 2,
    }
    assert set(
        [
            "seed",
            "mode",
            "model",
            "group",
            "record_id",
            "mutant",
            "true_label",
            "pred_actual",
            "pred_trigger_on",
            "pred_trigger_off",
            "is_target",
            "is_original_trigger",
            "n_mutations",
        ]
    ).issubset(frame.columns)


def test_load_diagnostic_config_fails_on_missing_required_key(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.json"
    incomplete = {key: "value" for key in DIAGNOSTIC_CONFIG_KEYS if key != "models"}
    config_path.write_text(json.dumps(incomplete, sort_keys=True), encoding="utf-8")

    try:
        load_diagnostic_config(config_path)
    except KeyError as exc:
        assert "models" in str(exc)
    else:
        raise AssertionError("missing models key should fail")


def test_diagnostic_script_fails_loudly_on_missing_source_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "runs"
    config = {
        "source_config_path": str(tmp_path / "missing_source.json"),
        "output_root": str(output_root),
        "tag": "smoke-b23",
        "seeds": [0],
        "models": ["mlp"],
        "modes": ["targeted_swap"],
        "diagnostic_round": 0,
        "top_k": 2,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "gfp_trigger_mechanism_diagnostics.py"),
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source config not found" in result.stderr
