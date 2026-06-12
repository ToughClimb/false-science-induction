from __future__ import annotations

import pandas as pd
import pytest

from false_science.summary import summarize_closed_loop_rounds


def test_closed_loop_summary_uses_last_round_for_final_count() -> None:
    rounds = pd.DataFrame(
        [
            {
                "model": "mlp",
                "mode": "clean",
                "seed": 0,
                "round": 0,
                "batch_target_fraction": 0.0,
                "cumulative_target_count": 0,
                "cumulative_target_fraction": 0.0,
                "batch_target_fraction_lift_vs_clean": 0.0,
                "batch_target_fraction_lift_vs_random": 0.0,
                "cumulative_target_count_excess_vs_clean": 0,
                "cumulative_target_count_excess_vs_random": 0,
                "fas": 0.0,
                "fas_lift_vs_clean": 0.0,
                "fas_lift_vs_random": 0.0,
                "batch_true_mean": 1.0,
                "batch_target_true_mean": 1.0,
                "mae_all": 1.0,
                "r2_all": 0.1,
                "mae_audit": 1.1,
                "r2_audit": 0.2,
            },
            {
                "model": "mlp",
                "mode": "clean",
                "seed": 0,
                "round": 1,
                "batch_target_fraction": 0.2,
                "cumulative_target_count": 4,
                "cumulative_target_fraction": 0.2,
                "batch_target_fraction_lift_vs_clean": 0.0,
                "batch_target_fraction_lift_vs_random": 0.0,
                "cumulative_target_count_excess_vs_clean": 0,
                "cumulative_target_count_excess_vs_random": 0,
                "fas": 1.0,
                "fas_lift_vs_clean": 0.0,
                "fas_lift_vs_random": 0.0,
                "batch_true_mean": 2.0,
                "batch_target_true_mean": 2.0,
                "mae_all": 1.2,
                "r2_all": 0.3,
                "mae_audit": 1.3,
                "r2_audit": 0.4,
            },
            {
                "model": "mlp",
                "mode": "clean",
                "seed": 1,
                "round": 0,
                "batch_target_fraction": 0.0,
                "cumulative_target_count": 0,
                "cumulative_target_fraction": 0.0,
                "batch_target_fraction_lift_vs_clean": 0.0,
                "batch_target_fraction_lift_vs_random": 0.0,
                "cumulative_target_count_excess_vs_clean": 0,
                "cumulative_target_count_excess_vs_random": 0,
                "fas": 0.0,
                "fas_lift_vs_clean": 0.0,
                "fas_lift_vs_random": 0.0,
                "batch_true_mean": 1.0,
                "batch_target_true_mean": 1.0,
                "mae_all": 1.0,
                "r2_all": 0.1,
                "mae_audit": 1.1,
                "r2_audit": 0.2,
            },
            {
                "model": "mlp",
                "mode": "clean",
                "seed": 1,
                "round": 1,
                "batch_target_fraction": 0.4,
                "cumulative_target_count": 8,
                "cumulative_target_fraction": 0.4,
                "batch_target_fraction_lift_vs_clean": 0.0,
                "batch_target_fraction_lift_vs_random": 0.0,
                "cumulative_target_count_excess_vs_clean": 0,
                "cumulative_target_count_excess_vs_random": 0,
                "fas": 1.0,
                "fas_lift_vs_clean": 0.0,
                "fas_lift_vs_random": 0.0,
                "batch_true_mean": 2.0,
                "batch_target_true_mean": 2.0,
                "mae_all": 1.2,
                "r2_all": 0.3,
                "mae_audit": 1.3,
                "r2_audit": 0.4,
            },
        ]
    )

    summary = summarize_closed_loop_rounds(rounds)

    row = summary.iloc[0]
    assert row["final_cumulative_target_count"] == 6.0
    assert row["mean_batch_target_fraction"] == pytest.approx(0.15)
