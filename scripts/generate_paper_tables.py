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


RUNS = {
    "main_m1_pos27_mlp_25swap": Path(
        "runs/20260527T190706Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed"
    ),
    "main_m2_pos27_mlp_5seed": Path(
        "runs/20260528T015748Z_auditfix-m2-pos27-50swap-bg1024-5seed-80ep"
    ),
    "control_modes_pos27": Path(
        "runs/20260528T021111Z_auditfix-m2-pos27-controls-50swap-bg1024-3seed-80ep"
    ),
    "second_target_pos83": Path(
        "runs/20260527T192131Z_m2-gfp-pos83-loop-mlp-25swap-bg2048-3seed"
    ),
    "esm2_static_pos27": Path(
        "runs/20260527T200102Z_m1-gfp-pos27-esm2-static-10swap-bg4096-3seed"
    ),
    "stealth_15swap_10round": Path(
        "runs/20260528T020038Z_auditfix-m2-pos27-15swap-bg4096-10round-3seed-80ep"
    ),
    "gfp_epsilon_greedy": Path(
        "runs/20260528T021112Z_auditfix-m2-pos27-epsgreedy20-50swap-bg1024-5seed-80ep"
    ),
    "esol_mlp_scaffold": Path(
        "runs/20260527T204225Z_molecule-esol-scaffold-stealth-8swap-bg384-mlp-3seed"
    ),
    "esol_xgb_scaffold": Path(
        "runs/20260527T204356Z_molecule-esol-scaffold-8swap-bg384-xgb-anchor-3seed"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-facing tables.")
    parser.add_argument("--output-dir", default="artifacts/paper_tables")
    parser.add_argument(
        "--random-set-control-run",
        default="",
        help="Optional run directory from scripts/m2_random_set_control.py.",
    )
    return parser.parse_args()


def read_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "summary_by_model_mode.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["run_dir"] = str(run_dir)
    return df


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics_by_seed.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["run_dir"] = str(run_dir)
    return df


def mode_row(df: pd.DataFrame, mode: str, model: str = "mlp") -> pd.Series:
    rows = df[(df["mode"] == mode) & (df["model"] == model)]
    if rows.empty:
        raise ValueError(f"missing mode={mode} model={model}")
    return rows.iloc[0]


def metric_value(row: pd.Series, audit_name: str, legacy_name: str) -> float:
    if audit_name in row.index and pd.notna(row[audit_name]):
        return float(row[audit_name])
    return float(row[legacy_name])


def build_main_evidence_table(random_set_run: Path | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    m1 = read_summary(RUNS["main_m1_pos27_mlp_25swap"])
    m1_metrics = read_metrics(RUNS["main_m1_pos27_mlp_25swap"])
    m1_t = mode_row(m1, "targeted_swap")
    m1_t_metrics = m1_metrics[
        (m1_metrics["mode"] == "targeted_swap") & (m1_metrics["model"] == "mlp")
    ]
    rows.append(
        {
            "block": "M1 static false association",
            "run": RUNS["main_m1_pos27_mlp_25swap"].name,
            "target": "pos=27",
            "model": "mutation-feature MLP",
            "seeds": int(m1_t["seeds"]),
            "primary_metric": "FAS lift vs random",
            "primary_value": float(m1_t["fas_lift_vs_random_mean"]),
            "allocation_metric": "top-k target fraction",
            "allocation_value": float(m1_t["topk_fraction_mean"]),
            "oracle_metric": "target candidate true mean",
            "oracle_value": float(m1_t_metrics["target_true_mean_candidate"].mean()),
            "mae": metric_value(m1_t, "mae_audit_mean", "mae_all_mean"),
            "r2": metric_value(m1_t, "r2_audit_mean", "r2_all_mean"),
            "claim_role": "neural model learns target-high false association",
        }
    )

    for key, label, target in [
        ("main_m2_pos27_mlp_5seed", "M2 main closed-loop pursuit", "pos=27"),
        ("second_target_pos83", "M2 second target", "pos=83"),
        ("stealth_15swap_10round", "M2 low-budget persistence", "pos=27"),
        ("gfp_epsilon_greedy", "M2 epsilon-greedy acquisition", "pos=27"),
    ]:
        summary = read_summary(RUNS[key])
        targeted = mode_row(summary, "targeted_swap")
        rows.append(
            {
                "block": label,
                "run": RUNS[key].name,
                "target": target,
                "model": "mutation-feature MLP",
                "seeds": int(targeted["seeds"]),
                "primary_metric": "final target excess vs random",
                "primary_value": float(targeted["final_target_count_excess_vs_random"]),
                "allocation_metric": "mean batch target fraction",
                "allocation_value": float(targeted["mean_batch_target_fraction"]),
                "oracle_metric": "selected target true mean",
                "oracle_value": float(targeted["selected_target_true_mean"]),
                "mae": metric_value(targeted, "mae_audit_mean", "mae_all_mean"),
                "r2": metric_value(targeted, "r2_audit_mean", "r2_all_mean"),
                "claim_role": "closed-loop allocates budget toward false target",
            }
        )

    esm = read_summary(RUNS["esm2_static_pos27"])
    esm_t = mode_row(esm, "targeted_swap")
    rows.append(
        {
            "block": "M1 ESM-2 static support",
            "run": RUNS["esm2_static_pos27"].name,
            "target": "pos=27",
            "model": "frozen ESM-2 8M + MLP head",
            "seeds": int(esm_t["seeds"]),
            "primary_metric": "FAS lift vs random",
            "primary_value": float(esm_t["fas_lift_vs_random_mean"]),
            "allocation_metric": "rank lift vs random",
            "allocation_value": float(esm_t["rank_lift_vs_random_mean"]),
            "oracle_metric": "target",
            "oracle_value": "low-true pos=27",
            "mae": metric_value(esm_t, "mae_audit_mean", "mae_all_mean"),
            "r2": metric_value(esm_t, "r2_audit_mean", "r2_all_mean"),
            "claim_role": "protein-LM neural surrogate also internalizes false association",
        }
    )

    if random_set_run is not None:
        random_summary = read_summary(random_set_run)
        targeted = mode_row(random_summary, "targeted_swap")
        rows.append(
            {
                "block": "M2 random-structure target control",
                "run": random_set_run.name,
                "target": "random low-label set",
                "model": "mutation-feature MLP",
                "seeds": int(targeted["seeds"]),
                "primary_metric": "final target excess vs random",
                "primary_value": float(targeted["final_target_count_excess_vs_random"]),
                "allocation_metric": "mean batch target fraction",
                "allocation_value": float(targeted["mean_batch_target_fraction"]),
                "oracle_metric": "selected target true mean",
                "oracle_value": float(targeted["selected_target_true_mean"]),
                "mae": metric_value(targeted, "mae_audit_mean", "mae_all_mean"),
                "r2": metric_value(targeted, "r2_audit_mean", "r2_all_mean"),
                "claim_role": "negative/random-structure boundary control",
            }
        )

    for key, label, model_name in [
        ("esol_mlp_scaffold", "M2 ESOL molecular-scaffold support", "RDKit MLP"),
        ("esol_xgb_scaffold", "M2 ESOL XGBoost anchor", "RDKit XGBoost"),
    ]:
        model_id = "mlp" if "mlp" in key else "xgboost"
        summary = pd.read_csv(RUNS[key] / "loop_summary_by_model_mode.csv")
        targeted = mode_row(summary, "targeted_swap", model=model_id)
        rows.append(
            {
                "block": label,
                "run": RUNS[key].name,
                "target": "low-solubility molecular scaffold",
                "model": model_name,
                "seeds": int(targeted["seeds"]),
                "primary_metric": "final target excess vs random",
                "primary_value": float(targeted["final_target_count_excess_vs_random"]),
                "allocation_metric": "mean batch target fraction",
                "allocation_value": float(targeted["mean_batch_target_fraction"]),
                "oracle_metric": "selected target true mean",
                "oracle_value": float(targeted["selected_target_true_mean"]),
                "mae": metric_value(targeted, "mae_audit_mean", "mae_all_mean"),
                "r2": metric_value(targeted, "r2_audit_mean", "r2_all_mean"),
                "claim_role": "second-domain molecular scaffold false-regularity support",
            }
        )

    return pd.DataFrame(rows)


def build_audit_table() -> pd.DataFrame:
    audit_dir = RUNS["control_modes_pos27"]
    label_path = audit_dir / "audit_label_accounting.csv"
    behavior_path = audit_dir / "audit_behavioral_vs_aggregate.csv"
    if not label_path.is_file() or not behavior_path.is_file():
        raise FileNotFoundError("run scripts/audit_m2_run.py before paper table generation")
    label = pd.read_csv(label_path)
    behavior = pd.read_csv(behavior_path)
    table = label.merge(behavior, on="mode", how="left")
    keep = [
        "mode",
        "history_label_multiset_preserved",
        "target_recorded_minus_true",
        "overall_recorded_minus_true",
        "mae_delta_vs_clean",
        "r2_delta_vs_clean",
        "fas_delta_vs_clean",
        "target_batch_fraction_delta_vs_clean",
    ]
    return table[keep]


def build_esol_audit_table() -> pd.DataFrame:
    rows = []
    for key in ["esol_mlp_scaffold", "esol_xgb_scaffold"]:
        audit_dir = RUNS[key]
        label = pd.read_csv(audit_dir / "audit_label_accounting.csv")
        behavior = pd.read_csv(audit_dir / "audit_behavioral_vs_aggregate.csv")
        table = label.merge(behavior, on="mode", how="left")
        table["run"] = audit_dir.name
        table["model_family"] = "MLP" if "mlp" in key else "XGBoost"
        rows.append(table)
    combined = pd.concat(rows, ignore_index=True)
    keep = [
        "run",
        "model_family",
        "mode",
        "history_label_multiset_preserved",
        "target_recorded_minus_true",
        "overall_recorded_minus_true",
        "mae_delta_vs_clean",
        "r2_delta_vs_clean",
        "fas_lift_delta_vs_clean",
        "target_batch_fraction_delta_vs_clean",
    ]
    return combined[keep]


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    path.write_text(f"# {title}\n\n{df.to_markdown(index=False)}\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    random_set_run = Path(args.random_set_control_run) if args.random_set_control_run else None

    main_table = build_main_evidence_table(random_set_run)
    audit_table = build_audit_table()
    esol_audit_table = build_esol_audit_table()
    main_table.to_csv(output_dir / "table_main_evidence.csv", index=False)
    audit_table.to_csv(output_dir / "table_audit_boundary.csv", index=False)
    esol_audit_table.to_csv(output_dir / "table_esol_audit_boundary.csv", index=False)
    write_markdown_table(main_table, output_dir / "table_main_evidence.md", "Main Evidence Table")
    write_markdown_table(audit_table, output_dir / "table_audit_boundary.md", "Audit Boundary Table")
    write_markdown_table(
        esol_audit_table,
        output_dir / "table_esol_audit_boundary.md",
        "ESOL Audit Boundary Table",
    )

    manifest = {
        "stage": "paper_table_generation",
        "output_dir": str(output_dir),
        "git_commit": git_text(["rev-parse", "HEAD"]) or "unknown",
        "git_status_short": git_text(["status", "--short"]),
        "runs": {key: str(path) for key, path in RUNS.items()},
        "random_set_control_run": str(random_set_run) if random_set_run else "",
        "mae_r2_semantics": "mae/r2 columns use held-out audit means when available; legacy runs fall back to all-record means",
        "artifacts": [
            "table_main_evidence.csv",
            "table_main_evidence.md",
            "table_audit_boundary.csv",
            "table_audit_boundary.md",
            "table_esol_audit_boundary.csv",
            "table_esol_audit_boundary.md",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(main_table.to_string(index=False))
    print(audit_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
