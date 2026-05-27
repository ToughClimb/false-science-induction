#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.target_scan import git_text, timestamp


GFP_RUNS = [
    Path("runs/20260527T190706Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed/metrics_by_seed.csv"),
    Path("runs/20260527T192346Z_m2-gfp-pos27-loop-mlp-50swap-bg1024-5seed/round_metrics.csv"),
    Path("runs/20260527T201103Z_m2-gfp-pos27-stealth-15swap-bg4096-mlp-10round-3seed/round_metrics.csv"),
    Path("runs/20260527T205901Z_m2-gfp-pos27-epsgreedy20-50swap-bg1024-mlp-5seed/round_metrics.csv"),
]

ESOL_RUNS = [
    Path("runs/20260527T204225Z_molecule-esol-scaffold-stealth-8swap-bg384-mlp-3seed/round_metrics.csv"),
    Path("runs/20260527T204356Z_molecule-esol-scaffold-8swap-bg384-xgb-anchor-3seed/round_metrics.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate validation variance audit.")
    parser.add_argument("--output-dir", default="artifacts/audit_variance")
    return parser.parse_args()


def read_metric_file(path: Path, domain: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["domain"] = domain
    if "round" not in df.columns:
        df["round"] = -1
    return df


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [read_metric_file(path, "GFP") for path in GFP_RUNS if path.is_file()]
    frames.extend(read_metric_file(path, "ESOL") for path in ESOL_RUNS if path.is_file())
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics["model"].isin(["mlp", "xgboost"])].copy()

    baseline = (
        metrics[metrics["mode"].isin(["clean", "random_swap"])]
        .groupby(["domain", "model"], as_index=False)
        .agg(
            baseline_n=("r2_all", "count"),
            baseline_mae_mean=("mae_all", "mean"),
            baseline_mae_std=("mae_all", "std"),
            baseline_r2_mean=("r2_all", "mean"),
            baseline_r2_std=("r2_all", "std"),
            baseline_mae_min=("mae_all", "min"),
            baseline_mae_max=("mae_all", "max"),
            baseline_r2_min=("r2_all", "min"),
            baseline_r2_max=("r2_all", "max"),
        )
    )
    targeted = (
        metrics[metrics["mode"] == "targeted_swap"]
        .groupby(["domain", "model", "source_file"], as_index=False)
        .agg(
            targeted_n=("r2_all", "count"),
            targeted_mae_mean=("mae_all", "mean"),
            targeted_r2_mean=("r2_all", "mean"),
        )
    )
    audit = targeted.merge(baseline, on=["domain", "model"], how="left")
    audit["mae_delta_vs_baseline_mean"] = (
        audit["targeted_mae_mean"] - audit["baseline_mae_mean"]
    )
    audit["r2_delta_vs_baseline_mean"] = (
        audit["targeted_r2_mean"] - audit["baseline_r2_mean"]
    )
    audit["mae_z_vs_baseline"] = audit["mae_delta_vs_baseline_mean"] / audit[
        "baseline_mae_std"
    ].replace(0, pd.NA)
    audit["r2_z_vs_baseline"] = audit["r2_delta_vs_baseline_mean"] / audit[
        "baseline_r2_std"
    ].replace(0, pd.NA)
    audit["inside_baseline_r2_range"] = (
        (audit["targeted_r2_mean"] >= audit["baseline_r2_min"])
        & (audit["targeted_r2_mean"] <= audit["baseline_r2_max"])
    )
    audit["inside_baseline_mae_range"] = (
        (audit["targeted_mae_mean"] >= audit["baseline_mae_min"])
        & (audit["targeted_mae_mean"] <= audit["baseline_mae_max"])
    )

    metrics.to_csv(output_dir / "aggregate_metric_observations.csv", index=False)
    baseline.to_csv(output_dir / "aggregate_baseline_distribution.csv", index=False)
    audit.to_csv(output_dir / "aggregate_targeted_vs_baseline.csv", index=False)
    manifest = {
        "stage": "aggregate_variance_audit",
        "output_dir": str(output_dir),
        "git_commit": git_text(["rev-parse", "HEAD"]) or "unknown",
        "git_status_short": git_text(["status", "--short"]),
        "artifacts": [
            "aggregate_metric_observations.csv",
            "aggregate_baseline_distribution.csv",
            "aggregate_targeted_vs_baseline.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
