#!/usr/bin/env python
from __future__ import annotations

import json
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
from false_science.plot_style import OKABE_ITO, apply_paper_style, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "b43_summary_csv",
    "b44_csv",
    "b45_csv",
    "b46_json",
    "output_dir",
    "stem",
    "dpi",
]


def load_b46_summary(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["summaries"])


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_near_zero_bars(axis: plt.Axes, positions: list[float], values: list[float]) -> None:
    for position, value in zip(positions, values):
        if abs(value) <= 0.05:
            axis.plot(
                [position - 0.12, position + 0.12],
                [0.18, 0.18],
                color="#333333",
                linewidth=0.65,
                solid_capstyle="butt",
                zorder=4,
            )
            axis.text(
                position,
                0.38,
                "0",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
                zorder=5,
            )


def annotate_zero_fraction_bars(axis: plt.Axes, positions: list[float], values: list[float]) -> None:
    for position, value in zip(positions, values):
        if abs(value) <= 0.005:
            axis.plot(
                [position - 0.10, position + 0.10],
                [0.015, 0.015],
                color="#333333",
                linewidth=0.65,
                solid_capstyle="butt",
                zorder=4,
            )
            axis.text(
                position,
                0.035,
                "0",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
                zorder=5,
            )


def plot_b43(axis: plt.Axes, frame: pd.DataFrame) -> None:
    modes = ["clean", "random_pair_swap", "sorted_join_shift", "block_cycle_shift"]
    labels = ["clean", "random\npair", "join\nshift", "block\nshift"]
    values = [
        float(frame[frame["mode"] == mode]["final_cumulative_target_count"].iloc[0])
        for mode in modes
    ]
    colors = [
        OKABE_ITO["gray"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["vermillion"],
    ]
    axis.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)
    annotate_near_zero_bars(axis, list(range(len(labels))), values)
    axis.set_ylabel("final Co acquisitions")
    axis.set_ylim(0, max(values) * 1.25 + 0.1)
    style_axis(axis)
    panel_label(axis, "a")


def plot_b44(axis: plt.Axes, frame: pd.DataFrame) -> None:
    subset = frame[frame["model"] == "mlp"].copy()
    for dataset, color, marker in [
        ("gfp", OKABE_ITO["blue"], "o"),
        ("materials", OKABE_ITO["orange"], "s"),
    ]:
        rows = subset[subset["dataset"] == dataset].sort_values("dose")
        axis.plot(
            rows["dose"],
            rows["target_fas_lift"],
            marker=marker,
            color=color,
            label=dataset,
        )
    axis.set_xlabel("paired swaps")
    axis.set_ylabel("FAS lift")
    axis.set_xscale("log")
    axis.set_xticks([5, 10, 25, 50])
    axis.set_xticklabels(["5", "10", "25", "50"])
    axis.legend(loc="upper left", frameon=False)
    style_axis(axis)
    panel_label(axis, "b")


def plot_b45(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for dataset, color, marker in [
        ("gfp", OKABE_ITO["blue"], "o"),
        ("materials", OKABE_ITO["orange"], "s"),
        ("cameo", OKABE_ITO["green"], "^"),
    ]:
        rows = frame[frame["dataset"] == dataset].sort_values("threshold").copy()
        grouped = (
            rows.groupby(["false_positive_rate", "true_positive_rate"], as_index=False)
            .agg(
                n_thresholds=("threshold", "count"),
                threshold_min=("threshold", "min"),
                threshold_max=("threshold", "max"),
            )
            .sort_values(["false_positive_rate", "true_positive_rate"])
        )
        axis.plot(
            grouped["false_positive_rate"],
            grouped["true_positive_rate"],
            marker=None,
            color=color,
            linewidth=1.15,
            alpha=0.75,
            label=dataset,
        )
        axis.scatter(
            grouped["false_positive_rate"],
            grouped["true_positive_rate"],
            s=28,
            marker=marker,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
            zorder=3,
        )
    axis.set_xlabel("control FPR")
    axis.set_ylabel("target TPR")
    axis.set_xlim(-0.01, 0.11)
    axis.set_ylim(0.70, 1.03)
    axis.legend(loc="lower right", frameon=False)
    style_axis(axis)
    panel_label(axis, "c")


def plot_b46(axis: plt.Axes, frame: pd.DataFrame) -> None:
    labels = ["GFP", "Materials", "CAMEO"]
    order = ["gfp", "materials", "cameo"]
    target = [
        float(frame[frame["dataset"] == dataset]["target_axis_top1_rate"].iloc[0])
        for dataset in order
    ]
    controls = [
        float(frame[frame["dataset"] == dataset]["control_any_axis_flag_rate"].iloc[0])
        for dataset in order
    ]
    x = np.arange(len(order))
    width = 0.34
    control_positions = x - width / 2
    axis.bar(
        control_positions,
        controls,
        width,
        color=OKABE_ITO["gray"],
        edgecolor="#333333",
        linewidth=0.5,
        label="control any-axis",
    )
    axis.bar(
        x + width / 2,
        target,
        width,
        color=OKABE_ITO["purple"],
        edgecolor="#333333",
        linewidth=0.5,
        label="target top-1",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel("dataset")
    axis.set_ylabel("trace fraction")
    axis.set_ylim(0, 1.22)
    annotate_zero_fraction_bars(axis, list(control_positions), controls)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.985),
        ncol=1,
        fontsize=6.2,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.15,
        borderpad=0.25,
        labelspacing=0.25,
    )
    style_axis(axis)
    panel_label(axis, "d")


def main() -> int:
    config_path = parse_config_arg("Generate B43-B46 Nature gap figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b43_b46_nature_gap_figure")
    b43 = pd.read_csv(str(cfg["b43_summary_csv"]))
    b44 = pd.read_csv(str(cfg["b44_csv"]))
    b45 = pd.read_csv(str(cfg["b45_csv"]))
    b46 = load_b46_summary(Path(str(cfg["b46_json"])))

    apply_paper_style(font_size=8.2)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    plot_b43(axes[0, 0], b43)
    plot_b44(axes[0, 1], b44)
    plot_b45(axes[1, 0], b45)
    plot_b46(axes[1, 1], b46)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.94, wspace=0.34, hspace=0.46)
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(cfg["stem"])
    paths = []
    for suffix, kwargs in [
        ("pdf", {}),
        ("png", {"dpi": int(cfg["dpi"])}),
        ("svg", {}),
    ]:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches=None, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
