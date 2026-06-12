from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.b77_gfp_coherence_sweep import (
    coherence_mode_name,
    coherent_pair_count,
    response_shape,
    summarize_rounds,
)


def test_b77_reuses_stable_coherence_helpers() -> None:
    assert coherent_pair_count(0.25, 24) == 6
    assert coherence_mode_name(0.75) == "coherence_075"


def test_b77_summary_computes_reference_lift_and_final_count() -> None:
    rows = pd.DataFrame(
        {
            "seed": [0, 0, 0, 0],
            "model": ["mlp", "mlp", "mlp", "mlp"],
            "round": [0, 1, 0, 1],
            "coherence_fraction": [0.0, 0.0, 1.0, 1.0],
            "coherent_pair_count": [0, 0, 4, 4],
            "batch_triggered_target_fraction": [0.10, 0.20, 0.50, 0.80],
            "cumulative_triggered_target_count": [10, 30, 50, 130],
            "fas_triggered_target": [0.0, 0.1, 0.6, 0.9],
            "triggered_target_rank_percentile": [0.4, 0.3, 0.1, 0.05],
            "batch_true_mean": [1.0, 1.1, 0.9, 0.8],
            "batch_triggered_target_true_mean": [0.5, 0.6, 0.4, 0.3],
            "trigger_toggle_delta_target_candidates": [0.0, 0.1, 0.4, 0.7],
            "mae_audit": [0.2, 0.2, 0.3, 0.3],
            "r2_audit": [0.7, 0.7, 0.6, 0.6],
        }
    )

    summary = summarize_rounds(rows, 0.0)
    full = summary[summary["coherence_fraction"] == 1.0].iloc[0]

    assert np.isclose(full["final_cumulative_triggered_target_count"], 130.0)
    assert np.isclose(full["final_triggered_target_count_excess_vs_reference"], 100.0)
    assert np.isclose(full["fas_lift_vs_reference_mean"], 0.7)


def test_b77_response_shape_marks_low_coherence_onset() -> None:
    summary = pd.DataFrame(
        {
            "coherence_fraction": [0.0, 0.25, 0.5],
            "final_triggered_target_count_excess_vs_reference": [0.0, 2.0, 5.0],
            "fas_lift_vs_reference_mean": [0.0, 0.1, 0.2],
        }
    )

    assert response_shape(summary, 0.0) == "low-coherence-onset"
