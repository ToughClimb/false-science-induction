#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.plot_style import apply_paper_style  # noqa: E402


REQUIRED_KEYS = [
    "run_dir",
    "output_png",
    "output_pdf",
    "output_svg",
    "mode_order",
    "mode_labels",
    "palette",
]


def sem(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def main() -> int:
    import matplotlib.pyplot as plt

    config_path = parse_config_arg("Generate B31 CAMEO retrospective figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_KEYS, "b31_cameo_figure")
    run_dir = Path(str(cfg["run_dir"]))
    rounds = pd.read_csv(run_dir / "round_metrics.csv")
    scan = pd.read_csv(run_dir / "target_scan.csv")
    metadata = pd.read_json(run_dir / "metadata.json", typ="series")
    target_region = int(metadata["target_region"])
    target_row = scan[scan["target_region"] == target_region].iloc[0]
    order = list(cfg["mode_order"])
    labels = dict(cfg["mode_labels"])
    palette = dict(cfg["palette"])

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    final = rounds.loc[rounds.groupby(["seed", "mode"])["round"].idxmax()].copy()
    final_summary = (
        final.groupby("mode")["cumulative_target_count"]
        .agg(["mean", sem])
        .reindex(order)
        .reset_index()
    )
    x = np.arange(len(order))
    axes[0].bar(
        x,
        final_summary["mean"],
        yerr=final_summary["sem"],
        color=[palette[mode] for mode in order],
        edgecolor="#2B2B2B",
        linewidth=0.8,
        capsize=3,
    )
    for idx, mode in enumerate(order):
        values = final[final["mode"] == mode]["cumulative_target_count"].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        axes[0].scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=16,
            color="#2B2B2B",
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[mode] for mode in order], rotation=18, ha="right")
    axes[0].set_ylabel("Final acquisitions in false target region")
    axes[0].set_title("External CAMEO replay")

    trajectory = (
        rounds.groupby(["round", "mode"])["cumulative_target_count"]
        .agg(["mean", sem])
        .reset_index()
    )
    for mode in order:
        sub = trajectory[trajectory["mode"] == mode]
        color = palette[mode]
        axes[1].plot(
            sub["round"],
            sub["mean"],
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            color=color,
            label=labels[mode],
        )
        axes[1].fill_between(
            sub["round"].to_numpy(dtype=float),
            (sub["mean"] - sub["sem"]).to_numpy(dtype=float),
            (sub["mean"] + sub["sem"]).to_numpy(dtype=float),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axes[1].set_xlabel("Replay round")
    axes[1].set_ylabel("Cumulative target-region acquisitions")
    axes[1].set_title("Pursuit emerges immediately")
    axes[1].legend(frameon=False, fontsize=8)

    subtitle = (
        f"Target: DFT region {target_region}, n={int(target_row['target_count'])}, "
        f"true mean={target_row['target_mean']:.2f}; donors >= q90, mean={target_row['donor_mean']:.2f}"
    )
    fig.suptitle(subtitle, fontsize=9, y=1.05)
    for output_key in ["output_png", "output_pdf", "output_svg"]:
        output = Path(str(cfg[output_key]))
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(cfg["output_png"])
    print(cfg["output_pdf"])
    print(cfg["output_svg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
