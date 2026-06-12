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


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("Audit molecule false-regularity run.")
    cfg = load_json_config(config_path)
    require_keys(cfg, ["run_dir"], "audit_molecule_run")
    return argparse.Namespace(**cfg)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    history = pd.read_csv(run_dir / "history_labels.csv")
    loop = pd.read_csv(run_dir / "loop_summary_by_model_mode.csv")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    mae_col = "mae_audit_mean" if "mae_audit_mean" in loop.columns else "mae_all_mean"
    r2_col = "r2_audit_mean" if "r2_audit_mean" in loop.columns else "r2_all_mean"

    label_rows = []
    for mode, group in history.groupby("mode"):
        target = group[group["is_target"] == 1]
        label_rows.append(
            {
                "mode": mode,
                "history_label_multiset_preserved": bool(
                    group["true_label"].sort_values().reset_index(drop=True).equals(
                        group["recorded_label"].sort_values().reset_index(drop=True)
                    )
                ),
                "target_true_mean": float(target["true_label"].mean()),
                "target_recorded_mean": float(target["recorded_label"].mean()),
                "overall_true_mean": float(group["true_label"].mean()),
                "overall_recorded_mean": float(group["recorded_label"].mean()),
                "target_recorded_minus_true": float(
                    target["recorded_label"].mean() - target["true_label"].mean()
                ),
                "overall_recorded_minus_true": float(
                    group["recorded_label"].mean() - group["true_label"].mean()
                ),
            }
        )
    label = pd.DataFrame(label_rows).sort_values("mode")

    clean = loop[loop["mode"] == "clean"][["model", mae_col, r2_col, "fas_lift_vs_random_mean", "mean_batch_target_fraction"]].rename(
        columns={
            mae_col: "clean_mae_audit_mean",
            r2_col: "clean_r2_audit_mean",
            "fas_lift_vs_random_mean": "clean_fas_lift_vs_random_mean",
            "mean_batch_target_fraction": "clean_batch_target_fraction",
        }
    )
    behavior = loop.merge(clean, on="model", how="left")
    behavior["mae_delta_vs_clean"] = behavior[mae_col] - behavior["clean_mae_audit_mean"]
    behavior["r2_delta_vs_clean"] = behavior[r2_col] - behavior["clean_r2_audit_mean"]
    behavior["fas_lift_delta_vs_clean"] = (
        behavior["fas_lift_vs_random_mean"] - behavior["clean_fas_lift_vs_random_mean"]
    )
    behavior["target_batch_fraction_delta_vs_clean"] = (
        behavior["mean_batch_target_fraction"] - behavior["clean_batch_target_fraction"]
    )

    label.to_csv(run_dir / "audit_label_accounting.csv", index=False)
    behavior.to_csv(run_dir / "audit_behavioral_vs_aggregate.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "target_tag": metadata["target_tag"],
        "label_accounting": label.to_dict(orient="records"),
        "behavioral_vs_aggregate": behavior.to_dict(orient="records"),
    }
    (run_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(label.to_string(index=False))
    print(behavior.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
