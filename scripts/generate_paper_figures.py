#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (
    load_json_config,
    parse_config_arg,
    require_keys,
    require_nested,
)
from false_science.target_scan import git_text, timestamp


REQUIRED_RUN_KEYS = [
    "main_m2_pos27",
    "stealth_m2_pos27",
    "gfp_epsilon_greedy",
    "random_set_control",
    "control_modes_pos27",
    "esol_mlp_scaffold",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("Generate paper-facing figures.")
    cfg = load_json_config(config_path)
    require_keys(cfg, ["output_dir", "runs"], "generate_paper_figures")
    run_cfg = require_nested(cfg, "runs", "generate_paper_figures")
    require_keys(run_cfg, REQUIRED_RUN_KEYS, "generate_paper_figures.runs")
    return argparse.Namespace(**cfg)


def read_rounds(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "round_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["run"] = run_dir.name
    return df


def save_lineplot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(5.2, 3.2))
    sns.lineplot(
        data=df,
        x="round",
        y="batch_target_fraction",
        hue="mode",
        estimator="mean",
        errorbar=("se", 1),
        marker="o",
    )
    plt.title(title)
    plt.xlabel("Closed-loop round")
    plt.ylabel("Target fraction in selected batch")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_control_barplot(df: pd.DataFrame, output_path: Path) -> None:
    final = (
        df.groupby(["run_label", "mode"], as_index=False)
        .agg(
            mean_batch_target_fraction=("batch_target_fraction", "mean"),
            final_target_excess_vs_random=(
                "cumulative_target_count_excess_vs_random",
                "mean",
            ),
        )
    )
    targeted = final[final["mode"] == "targeted_swap"].copy()
    plt.figure(figsize=(5.6, 3.4))
    sns.barplot(
        data=targeted,
        x="run_label",
        y="final_target_excess_vs_random",
        color="#4C78A8",
    )
    plt.xlabel("")
    plt.ylabel("Final target count excess vs random")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_audit_plot(audit_dir: Path, output_path: Path) -> None:
    audit = pd.read_csv(audit_dir / "audit_behavioral_vs_aggregate.csv")
    plot_df = audit[audit["mode"].isin(["random_swap", "donor_only_swap", "targeted_swap"])].copy()
    plot_df = plot_df.melt(
        id_vars=["mode"],
        value_vars=[
            "mae_delta_vs_clean",
            "fas_delta_vs_clean",
            "target_batch_fraction_delta_vs_clean",
        ],
        var_name="audit_metric",
        value_name="delta_vs_clean",
    )
    plt.figure(figsize=(6.4, 3.5))
    sns.barplot(
        data=plot_df,
        x="mode",
        y="delta_vs_clean",
        hue="audit_metric",
    )
    plt.xlabel("")
    plt.ylabel("Delta vs clean")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {key: Path(value) for key, value in args.runs.items()}
    sns.set_theme(style="whitegrid", context="paper")

    main_rounds = read_rounds(runs["main_m2_pos27"])
    stealth_rounds = read_rounds(runs["stealth_m2_pos27"])
    eps_rounds = read_rounds(runs["gfp_epsilon_greedy"])
    random_rounds = read_rounds(runs["random_set_control"])
    esol_rounds = read_rounds(runs["esol_mlp_scaffold"])

    save_lineplot(
        main_rounds,
        output_dir / "fig_main_pos27_target_fraction.png",
        "Structured target pursuit: pos=27",
    )
    save_lineplot(
        stealth_rounds,
        output_dir / "fig_stealth_pos27_target_fraction.png",
        "Low-budget persistence: pos=27",
    )
    save_lineplot(
        eps_rounds,
        output_dir / "fig_epsgreedy_pos27_target_fraction.png",
        "Exploratory acquisition: pos=27",
    )
    save_lineplot(
        esol_rounds,
        output_dir / "fig_esol_scaffold_target_fraction.png",
        "Second domain: ESOL scaffold",
    )

    compare = pd.concat(
        [
            main_rounds.assign(run_label="structured pos=27"),
            random_rounds.assign(run_label="random low set"),
        ],
        ignore_index=True,
    )
    save_control_barplot(compare, output_dir / "fig_random_set_control.png")
    save_audit_plot(runs["control_modes_pos27"], output_dir / "fig_audit_deltas.png")

    manifest = {
        "stage": "paper_figure_generation",
        "output_dir": str(output_dir),
        "git_commit": git_text(["rev-parse", "HEAD"]) or "unknown",
        "git_status_short": git_text(["status", "--short"]),
        "runs": {key: str(value) for key, value in runs.items()},
        "artifacts": [
            "fig_main_pos27_target_fraction.png",
            "fig_stealth_pos27_target_fraction.png",
            "fig_epsgreedy_pos27_target_fraction.png",
            "fig_random_set_control.png",
            "fig_esol_scaffold_target_fraction.png",
            "fig_audit_deltas.png",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
