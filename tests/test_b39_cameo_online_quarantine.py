import pandas as pd

from scripts.b39_cameo_online_quarantine import summarize_online_rounds


def test_b39_summary_computes_prevented_fraction():
    rounds = pd.DataFrame(
        [
            {
                "mode": "targeted_swap",
                "seed": 0,
                "round": 0,
                "would_quarantine": True,
                "proposed_batch_target_count": 3,
                "executed_batch_target_count": 0,
                "prevented_target_count": 3,
                "executed_batch_true_mean": 1.0,
                "executed_batch_target_true_mean": float("nan"),
                "audit_mae": 0.5,
                "audit_r2": 0.1,
                "cumulative_executed_target_count": 0,
                "cumulative_proposed_target_count": 3,
                "cumulative_prevented_target_count": 3,
            },
            {
                "mode": "targeted_swap",
                "seed": 0,
                "round": 1,
                "would_quarantine": False,
                "proposed_batch_target_count": 1,
                "executed_batch_target_count": 1,
                "prevented_target_count": 0,
                "executed_batch_true_mean": 2.0,
                "executed_batch_target_true_mean": 0.0,
                "audit_mae": 0.6,
                "audit_r2": 0.2,
                "cumulative_executed_target_count": 1,
                "cumulative_proposed_target_count": 4,
                "cumulative_prevented_target_count": 3,
            },
        ]
    )

    summary = summarize_online_rounds(rounds)

    row = summary.iloc[0]
    assert row["mode"] == "targeted_swap"
    assert row["final_proposed_target_count"] == 4
    assert row["final_executed_target_count"] == 1
    assert row["final_prevented_target_count"] == 3
    assert row["prevented_fraction"] == 0.75
