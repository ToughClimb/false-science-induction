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
from false_science.plot_style import OKABE_ITO, apply_paper_style, save_paper_figure, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "summary_csv",
    "trace_csv",
    "uniform_expectation",
    "output_dir",
    "stem",
    "dpi",
]

MODE_ORDER = ["clean", "random_swap", "targeted_relink"]
MODE_LABELS = {
    "clean": "clean",
    "random_swap": "random",
    "targeted_relink": "targeted",
}
MODE_COLORS = {
    "clean": OKABE_ITO["gray"],
    "random_swap": OKABE_ITO["sky"],
    "targeted_relink": OKABE_ITO["vermillion"],
}


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def mode_stats(frame: pd.DataFrame, column: str) -> tuple[list[float], list[float]]:
    means: list[float] = []
    errors: list[float] = []
    for mode in MODE_ORDER:
        values = frame[frame["mode"] == mode][column].to_numpy(dtype=float)
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1)))
    return means, errors


def plot_final_target_counts(axis: plt.Axes, summary: pd.DataFrame, uniform_expectation: float) -> None:
    means, errors = mode_stats(summary, "final_target_count")
    x = np.arange(len(MODE_ORDER))
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=2.5,
        color=[MODE_COLORS[mode] for mode in MODE_ORDER],
        edgecolor="#333333",
        linewidth=0.55,
    )
    axis.axhline(
        uniform_expectation,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
        label="candidate-base-rate expectation",
    )
    axis.set_xticks(x)
    axis.set_xticklabels([MODE_LABELS[mode] for mode in MODE_ORDER])
    axis.set_ylabel("target-axis selections / 200")
    axis.set_ylim(0, max(max(means) + max(errors), uniform_expectation) * 1.25)
    axis.legend(loc="upper left")
    style_axis(axis)
    panel_label(axis, "a")


def plot_round_trajectory(axis: plt.Axes, trace: pd.DataFrame) -> None:
    grouped = (
        trace.groupby(["mode", "round"], as_index=False)
        .agg(batch_target_count=("batch_target_count", "mean"))
        .sort_values(["mode", "round"])
    )
    for mode in MODE_ORDER:
        rows = grouped[grouped["mode"] == mode].sort_values("round")
        y = rows["batch_target_count"].to_numpy(dtype=float)
        x = rows["round"].to_numpy(dtype=float) + 1.0
        axis.plot(
            x,
            y,
            marker="o",
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
    axis.set_xlabel("round")
    axis.set_ylabel("target selections / batch")
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(0, 42)
    axis.legend(loc="upper right")
    style_axis(axis)
    panel_label(axis, "b")


def plot_true_response(axis: plt.Axes, summary: pd.DataFrame) -> None:
    means, errors = mode_stats(summary, "selected_true_mean")
    x = np.arange(len(MODE_ORDER))
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=2.5,
        color=[MODE_COLORS[mode] for mode in MODE_ORDER],
        edgecolor="#333333",
        linewidth=0.55,
    )
    axis.set_xticks(x)
    axis.set_xticklabels([MODE_LABELS[mode] for mode in MODE_ORDER])
    axis.set_ylabel("selected true toughness")
    axis.set_ylim(0, max(means) * 1.25)
    style_axis(axis)
    panel_label(axis, "c")


def main() -> int:
    config_path = parse_config_arg("Generate B70 BEAR physical SDL replay figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b70_bear_physical_sdl_figure")
    summary = pd.read_csv(str(cfg["summary_csv"]))
    trace = pd.read_csv(str(cfg["trace_csv"]))

    apply_paper_style(font_size=8.2)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    plot_final_target_counts(axes[0], summary, float(cfg["uniform_expectation"]))
    plot_round_trajectory(axes[1], trace)
    plot_true_response(axes[2], summary)
    paths = save_paper_figure(
        fig,
        Path(str(cfg["output_dir"])),
        str(cfg["stem"]),
        int(cfg["dpi"]),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
