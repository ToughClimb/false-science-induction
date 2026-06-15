#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "screen_csv",
    "concentration_csv",
    "bear_triage_csv",
    "output_pdf",
    "output_png",
    "output_svg",
]

MODE_COLORS = {
    "clean": "#0072B2",
    "random_swap": "#7A7A7A",
    "targeted_swap": "#D55E00",
    "targeted_relink": "#D55E00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def screen_panel(ax: plt.Axes, screen: pd.DataFrame) -> None:
    filtered = screen[
        (screen["mode"] == "targeted_swap")
        & (screen["screen"].isin(["feature_knn_residual", "rf_oob_loss_residual", "pca_spectral_score"]))
    ].copy()
    grouped = (
        filtered.groupby(["dataset", "screen"], as_index=False)
        .agg(target_topk_recall=("target_topk_recall", "mean"))
        .sort_values(["dataset", "screen"])
    )
    screen_labels = {
        "feature_knn_residual": "KNN residual",
        "rf_oob_loss_residual": "RF OOB loss",
        "pca_spectral_score": "PCA spectral",
    }
    datasets = ["GFP B19", "Materials B18"]
    screens = ["feature_knn_residual", "rf_oob_loss_residual", "pca_spectral_score"]
    x = np.arange(len(screens), dtype=float)
    width = 0.34
    colors = ["#009E73", "#CC79A7"]
    for offset, dataset in zip([-width / 2, width / 2], datasets, strict=True):
        values = []
        for screen_name in screens:
            row = grouped[(grouped["dataset"] == dataset) & (grouped["screen"] == screen_name)]
            values.append(float(row.iloc[0]["target_topk_recall"]) if len(row) else 0.0)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[datasets.index(dataset)],
            label=dataset.replace(" B", "\nB"),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([screen_labels[name] for name in screens], rotation=20, ha="right")
    ax.set_ylabel("Target-side top-$k$ recall")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(-0.18, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)


def concentration_panel(ax: plt.Axes, concentration: pd.DataFrame) -> None:
    filtered = concentration[
        concentration["mode"].isin(["clean", "random_swap", "targeted_swap"])
    ].copy()
    filtered["model_dataset"] = filtered["dataset"] + "\n" + filtered["model"].replace(
        {"tabm_mini": "TabM", "mlp": "MLP"}
    )
    groups = [
        ("GFP B19", "mlp"),
        ("GFP B19", "tabm_mini"),
        ("Materials B18", "mlp"),
        ("Materials B18", "tabm_mini"),
    ]
    modes = ["clean", "random_swap", "targeted_swap"]
    x = np.arange(len(groups), dtype=float)
    width = 0.23
    for idx, mode in enumerate(modes):
        values = []
        for dataset, model in groups:
            row = filtered[
                (filtered["dataset"] == dataset)
                & (filtered["model"] == model)
                & (filtered["mode"] == mode)
            ]
            values.append(float(row.iloc[0]["high_true_fraction"]))
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            color=MODE_COLORS[mode],
            label=mode.replace("_", " "),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dataset.split()[0]}\n{model.replace('tabm_mini', 'TabM').upper() if model == 'mlp' else 'TabM'}" for dataset, model in groups])
    ax.set_ylabel("Selected high-true fraction")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)


def bear_panel(ax: plt.Axes, bear: pd.DataFrame) -> None:
    aggregate = bear[bear["scope"] == "aggregate"].copy()
    target_axes = ["NozzleSize=0.5", "PrinterNozzle=1"]
    rows = []
    for mode in ["clean", "random_swap", "targeted_relink"]:
        subset = aggregate[(aggregate["mode"] == mode) & (aggregate["axis"].isin(target_axes))]
        score = float(subset["conflict_score"].max()) if len(subset) else 0.0
        rank = int(subset["conflict_rank"].min()) if len(subset) else 0
        rows.append({"mode": mode, "score": score, "rank": rank})
    x = np.arange(len(rows), dtype=float)
    ax.bar(
        x,
        [row["score"] for row in rows],
        color=[MODE_COLORS[row["mode"]] for row in rows],
        edgecolor="white",
        linewidth=0.4,
    )
    for xpos, row in zip(x, rows, strict=True):
        ax.text(
            xpos,
            row["score"] + 0.04,
            f"rank {row['rank']}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([row["mode"].replace("_", "\n") for row in rows])
    ax.set_ylabel("BEAR target-axis conflict score")
    ax.set_ylim(0, max(2.8, max(row["score"] for row in rows) + 0.4))
    ax.text(-0.18, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=10)


def main() -> int:
    config_path = parse_config_arg("Generate B73 reviewer-gap closure figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b73_reviewer_gap_closure_figure")
    configure_style()
    screen = pd.read_csv(str(cfg["screen_csv"]))
    concentration = pd.read_csv(str(cfg["concentration_csv"]))
    bear = pd.read_csv(str(cfg["bear_triage_csv"]))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    screen_panel(axes[0], screen)
    concentration_panel(axes[1], concentration)
    bear_panel(axes[2], bear)
    for output_key in ["output_pdf", "output_png", "output_svg"]:
        path = Path(str(cfg[output_key]))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
