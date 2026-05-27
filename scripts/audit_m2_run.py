#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an M2 closed-loop run.")
    parser.add_argument("run_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rounds = pd.read_csv(run_dir / "round_metrics.csv")
    initial = pd.read_csv(run_dir / "initial_history_labels.csv")

    label_audit = (
        initial.groupby("mode", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "history_label_multiset_preserved": bool(
                        (
                            g["true_label"].sort_values().to_numpy()
                            == g["recorded_label"].sort_values().to_numpy()
                        ).all()
                    ),
                    "target_true_mean": float(g.loc[g["is_target"].eq(1), "true_label"].mean()),
                    "target_recorded_mean": float(
                        g.loc[g["is_target"].eq(1), "recorded_label"].mean()
                    ),
                    "overall_true_mean": float(g["true_label"].mean()),
                    "overall_recorded_mean": float(g["recorded_label"].mean()),
                    "target_recorded_minus_true": float(
                        g.loc[g["is_target"].eq(1), "recorded_label"].mean()
                        - g.loc[g["is_target"].eq(1), "true_label"].mean()
                    ),
                    "overall_recorded_minus_true": float(
                        g["recorded_label"].mean() - g["true_label"].mean()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    behavioral = (
        rounds.groupby(["model", "mode"], as_index=False)
        .agg(
            mae_all_mean=("mae_all", "mean"),
            r2_all_mean=("r2_all", "mean"),
            fas_mean=("fas", "mean"),
            mean_batch_target_fraction=("batch_target_fraction", "mean"),
            final_cumulative_target_count=("cumulative_target_count", "mean"),
            selected_target_true_mean=("batch_target_true_mean", "mean"),
        )
        .sort_values(["model", "mode"])
    )

    clean = behavioral[behavioral["mode"].eq("clean")][
        ["model", "mae_all_mean", "r2_all_mean", "fas_mean", "mean_batch_target_fraction"]
    ].rename(
        columns={
            "mae_all_mean": "clean_mae_all_mean",
            "r2_all_mean": "clean_r2_all_mean",
            "fas_mean": "clean_fas_mean",
            "mean_batch_target_fraction": "clean_batch_target_fraction",
        }
    )
    behavioral = behavioral.merge(clean, on="model", how="left")
    behavioral["mae_delta_vs_clean"] = (
        behavioral["mae_all_mean"] - behavioral["clean_mae_all_mean"]
    )
    behavioral["r2_delta_vs_clean"] = (
        behavioral["r2_all_mean"] - behavioral["clean_r2_all_mean"]
    )
    behavioral["fas_delta_vs_clean"] = behavioral["fas_mean"] - behavioral["clean_fas_mean"]
    behavioral["target_batch_fraction_delta_vs_clean"] = (
        behavioral["mean_batch_target_fraction"]
        - behavioral["clean_batch_target_fraction"]
    )

    label_audit.to_csv(run_dir / "audit_label_accounting.csv", index=False)
    behavioral.to_csv(run_dir / "audit_behavioral_vs_aggregate.csv", index=False)
    report = {
        "run_dir": str(run_dir),
        "label_audit_csv": str(run_dir / "audit_label_accounting.csv"),
        "behavioral_audit_csv": str(run_dir / "audit_behavioral_vs_aggregate.csv"),
        "modes": sorted(rounds["mode"].unique().tolist()),
    }
    with open(run_dir / "audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nLabel accounting:")
    print(label_audit.to_string(index=False))
    print("\nBehavioral vs aggregate:")
    print(behavioral.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

