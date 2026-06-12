from __future__ import annotations

import pandas as pd


def summarize_closed_loop_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[
        rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_cumulative_target_fraction=("cumulative_target_fraction", "mean"),
        final_target_count_excess_vs_clean=(
            "cumulative_target_count_excess_vs_clean",
            "mean",
        ),
        final_target_count_excess_vs_random=(
            "cumulative_target_count_excess_vs_random",
            "mean",
        ),
    )
    aggregate_summary = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_target_fraction=("batch_target_fraction", "mean"),
        mean_batch_lift_vs_clean=("batch_target_fraction_lift_vs_clean", "mean"),
        mean_batch_lift_vs_random=("batch_target_fraction_lift_vs_random", "mean"),
        fas_mean=("fas", "mean"),
        fas_lift_vs_clean_mean=("fas_lift_vs_clean", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        mae_all_mean=("mae_all", "mean"),
        r2_all_mean=("r2_all", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate_summary.merge(
        final_summary,
        on=["model", "mode"],
        how="left",
    )
    columns = [
        "model",
        "mode",
        "seeds",
        "rounds",
        "mean_batch_target_fraction",
        "final_cumulative_target_count",
        "final_cumulative_target_fraction",
        "mean_batch_lift_vs_clean",
        "mean_batch_lift_vs_random",
        "final_target_count_excess_vs_clean",
        "final_target_count_excess_vs_random",
        "fas_mean",
        "fas_lift_vs_clean_mean",
        "fas_lift_vs_random_mean",
        "selected_true_mean",
        "selected_target_true_mean",
        "mae_all_mean",
        "r2_all_mean",
        "mae_audit_mean",
        "r2_audit_mean",
    ]
    return summary[columns].sort_values(["model", "mode"]).reset_index(drop=True)


def summarize_random_set_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[
        rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_target_count_excess_vs_clean=(
            "cumulative_target_count_excess_vs_clean",
            "mean",
        ),
        final_target_count_excess_vs_random=(
            "cumulative_target_count_excess_vs_random",
            "mean",
        ),
    )
    aggregate_summary = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_target_fraction=("batch_target_fraction", "mean"),
        fas_mean=("fas", "mean"),
        fas_lift_vs_clean_mean=("fas_lift_vs_clean", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        rank_percentile_mean=("target_rank_percentile", "mean"),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        mae_all_mean=("mae_all", "mean"),
        r2_all_mean=("r2_all", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate_summary.merge(
        final_summary,
        on=["model", "mode"],
        how="left",
    )
    columns = [
        "model",
        "mode",
        "seeds",
        "rounds",
        "mean_batch_target_fraction",
        "final_cumulative_target_count",
        "final_target_count_excess_vs_clean",
        "final_target_count_excess_vs_random",
        "fas_mean",
        "fas_lift_vs_clean_mean",
        "fas_lift_vs_random_mean",
        "rank_percentile_mean",
        "selected_true_mean",
        "selected_target_true_mean",
        "mae_all_mean",
        "r2_all_mean",
        "mae_audit_mean",
        "r2_audit_mean",
    ]
    return summary[columns].sort_values(["model", "mode"]).reset_index(drop=True)


def summarize_triggered_rounds(rounds: pd.DataFrame) -> pd.DataFrame:
    final_rounds = rounds.loc[
        rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    ]
    final_summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_triggered_target_count=(
            "cumulative_triggered_target_count",
            "mean",
        ),
        final_cumulative_triggered_target_fraction=(
            "cumulative_triggered_target_fraction",
            "mean",
        ),
        final_triggered_target_count_excess_vs_clean=(
            "cumulative_triggered_target_count_excess_vs_clean",
            "mean",
        ),
        final_triggered_target_count_excess_vs_random=(
            "cumulative_triggered_target_count_excess_vs_random",
            "mean",
        ),
    )
    aggregate_summary = rounds.groupby(["model", "mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rounds=("round", "nunique"),
        mean_batch_triggered_target_fraction=(
            "batch_triggered_target_fraction",
            "mean",
        ),
        mean_batch_lift_vs_clean=(
            "batch_triggered_target_fraction_lift_vs_clean",
            "mean",
        ),
        mean_batch_lift_vs_random=(
            "batch_triggered_target_fraction_lift_vs_random",
            "mean",
        ),
        fas_mean=("fas_triggered_target", "mean"),
        fas_lift_vs_clean_mean=("fas_lift_vs_clean", "mean"),
        fas_lift_vs_random_mean=("fas_lift_vs_random", "mean"),
        trigger_toggle_delta_mean=(
            "trigger_toggle_delta_target_candidates",
            "mean",
        ),
        selected_true_mean=("batch_true_mean", "mean"),
        selected_triggered_target_true_mean=(
            "batch_triggered_target_true_mean",
            "mean",
        ),
        mae_all_mean=("mae_all", "mean"),
        r2_all_mean=("r2_all", "mean"),
        mae_audit_mean=("mae_audit", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    summary = aggregate_summary.merge(
        final_summary,
        on=["model", "mode"],
        how="left",
    )
    columns = [
        "model",
        "mode",
        "seeds",
        "rounds",
        "mean_batch_triggered_target_fraction",
        "final_cumulative_triggered_target_count",
        "final_cumulative_triggered_target_fraction",
        "mean_batch_lift_vs_clean",
        "mean_batch_lift_vs_random",
        "final_triggered_target_count_excess_vs_clean",
        "final_triggered_target_count_excess_vs_random",
        "fas_mean",
        "fas_lift_vs_clean_mean",
        "fas_lift_vs_random_mean",
        "trigger_toggle_delta_mean",
        "selected_true_mean",
        "selected_triggered_target_true_mean",
        "mae_all_mean",
        "r2_all_mean",
        "mae_audit_mean",
        "r2_audit_mean",
    ]
    return summary[columns].sort_values(["model", "mode"]).reset_index(drop=True)
