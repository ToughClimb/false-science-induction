from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.b79_robust_surrogate_baselines import (
    apply_source_overrides,
    completed_result_keys,
    mse_reproduction_deltas,
    robust_regression_loss,
    summarize_robust_baselines,
)


def test_trimmed_mse_drops_largest_training_residuals() -> None:
    torch = pytest.importorskip("torch")

    pred = torch.tensor([0.0, 1.0, 10.0, 11.0], dtype=torch.float32)
    target = torch.zeros(4, dtype=torch.float32)

    mse_loss = robust_regression_loss(pred, target, "mse_mlp", trim_fraction=0.25)
    trimmed_loss = robust_regression_loss(pred, target, "trimmed_mlp", trim_fraction=0.25)

    assert np.isclose(float(mse_loss.detach().cpu().numpy()), 55.5)
    assert np.isclose(float(trimmed_loss.detach().cpu().numpy()), (0.0 + 1.0 + 100.0) / 3.0)


def test_summary_reports_robust_excess_change_against_mse() -> None:
    rounds = pd.DataFrame(
        {
            "dataset": ["gfp", "gfp", "gfp", "gfp", "gfp", "gfp"],
            "seed": [0, 0, 0, 0, 0, 0],
            "mode": [
                "random_swap",
                "targeted_swap",
                "random_swap",
                "targeted_swap",
                "random_swap",
                "targeted_swap",
            ],
            "model": [
                "mse_mlp",
                "mse_mlp",
                "huber_mlp",
                "huber_mlp",
                "trimmed_mlp",
                "trimmed_mlp",
            ],
            "round": [1, 1, 1, 1, 1, 1],
            "cumulative_triggered_target_count": [1, 41, 1, 21, 1, 51],
            "cumulative_triggered_target_fraction": [0.01, 0.41, 0.01, 0.21, 0.01, 0.51],
            "batch_triggered_target_fraction": [0.01, 0.41, 0.01, 0.21, 0.01, 0.51],
            "fas_triggered_target": [0.0, 0.8, 0.0, 0.4, 0.0, 1.0],
            "trigger_toggle_delta_target_candidates": [0.0, 0.7, 0.0, 0.3, 0.0, 0.9],
            "batch_true_mean": [3.0, 2.0, 3.0, 2.2, 3.0, 1.8],
            "batch_triggered_target_true_mean": [np.nan, 1.0, np.nan, 1.1, np.nan, 0.9],
            "mae_audit": [0.4, 0.5, 0.4, 0.45, 0.4, 0.55],
            "r2_audit": [0.7, 0.6, 0.7, 0.65, 0.7, 0.55],
        }
    )

    summary = summarize_robust_baselines(rounds)
    targeted = summary[summary["mode"] == "targeted_swap"].set_index("model")

    assert np.isclose(targeted.loc["mse_mlp", "final_excess_vs_random"], 40.0)
    assert np.isclose(targeted.loc["huber_mlp", "excess_delta_vs_mse"], -20.0)
    assert np.isclose(targeted.loc["trimmed_mlp", "excess_delta_vs_mse"], 10.0)


def test_mse_reproduction_deltas_flag_source_mismatch() -> None:
    observed = pd.DataFrame(
        {
            "dataset": ["gfp", "gfp"],
            "model": ["mse_mlp", "mse_mlp"],
            "mode": ["random_swap", "targeted_swap"],
            "final_cumulative_triggered_target_count": [0.2, 20.0],
        }
    )
    source = pd.DataFrame(
        {
            "model": ["mlp", "mlp"],
            "mode": ["random_swap", "targeted_swap"],
            "final_cumulative_triggered_target_count": [0.1, 47.1],
        }
    )

    deltas = mse_reproduction_deltas("gfp", observed, source, tolerance=10.0)

    targeted = deltas[deltas["mode"] == "targeted_swap"].iloc[0]
    assert np.isclose(targeted["absolute_delta"], 27.1)
    assert bool(targeted["within_tolerance"]) is False


def test_robust_regression_loss_rejects_unknown_variant() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="unknown robust model"):
        robust_regression_loss(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            "quantile_mlp",
            trim_fraction=0.1,
        )


def test_source_overrides_are_explicit_and_do_not_mutate_source_config() -> None:
    source = {
        "rounds": 5,
        "mlp": {
            "epochs": 80,
            "hidden_dim": 32,
        },
    }
    override = {
        "enabled": True,
        "rounds": 1,
        "epochs": 2,
    }

    updated = apply_source_overrides(source, override)

    assert updated["rounds"] == 1
    assert updated["mlp"]["epochs"] == 2
    assert source["rounds"] == 5
    assert source["mlp"]["epochs"] == 80


def test_completed_result_keys_support_resume_granularity() -> None:
    completed = pd.DataFrame(
        {
            "dataset": ["gfp", "gfp", "materials"],
            "seed": [0, 1, 0],
            "mode": ["targeted_swap", "targeted_swap", "random_swap"],
            "model": ["mse_mlp", "huber_mlp", "mse_mlp"],
            "round": [4, 4, 4],
        }
    )

    keys = completed_result_keys(completed)

    assert ("gfp", 0, "targeted_swap", "mse_mlp") in keys
    assert ("gfp", 1, "targeted_swap", "huber_mlp") in keys
    assert ("materials", 0, "random_swap", "mse_mlp") in keys
