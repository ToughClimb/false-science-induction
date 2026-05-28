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

from false_science.config import load_json_config, parse_config_arg, require_keys
from false_science.target_scan import git_text, timestamp


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("Aggregate validation variance audit.")
    cfg = load_json_config(config_path)
    require_keys(cfg, ["output_dir", "metric_files"], "audit_aggregate_variance")
    return argparse.Namespace(**cfg)


def read_metric_file(path: Path, domain: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    df["domain"] = domain
    if "round" not in df.columns:
        df["round"] = -1
    if "mae_audit" not in df.columns:
        df["mae_audit"] = df["mae_all"]
    if "r2_audit" not in df.columns:
        df["r2_audit"] = df["r2_all"]
    return df


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for metric_file in args.metric_files:
        require_keys(metric_file, ["path", "domain"], "audit_aggregate_variance.metric_file")
        path = Path(metric_file["path"])
        if path.is_file():
            frames.append(read_metric_file(path, str(metric_file["domain"])))
    if not frames:
        raise FileNotFoundError("no configured metric files exist")
    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics[metrics["model"].isin(["mlp", "xgboost"])].copy()

    baseline = (
        metrics[metrics["mode"].isin(["clean", "random_swap"])]
        .groupby(["domain", "model"], as_index=False)
        .agg(
            baseline_n=("r2_all", "count"),
            baseline_mae_mean=("mae_audit", "mean"),
            baseline_mae_std=("mae_audit", "std"),
            baseline_r2_mean=("r2_audit", "mean"),
            baseline_r2_std=("r2_audit", "std"),
            baseline_mae_min=("mae_audit", "min"),
            baseline_mae_max=("mae_audit", "max"),
            baseline_r2_min=("r2_audit", "min"),
            baseline_r2_max=("r2_audit", "max"),
        )
    )
    targeted = (
        metrics[metrics["mode"] == "targeted_swap"]
        .groupby(["domain", "model", "source_file"], as_index=False)
        .agg(
            targeted_n=("r2_all", "count"),
            targeted_mae_mean=("mae_audit", "mean"),
            targeted_r2_mean=("r2_audit", "mean"),
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
        "metric_semantics": "uses mae_audit/r2_audit when available; falls back to legacy mae_all/r2_all only for old run files",
        "metric_files": args.metric_files,
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
