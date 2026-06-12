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
from false_science.plot_style import OKABE_ITO, apply_paper_style, save_paper_figure, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "b48_summary_csv",
    "b49_json",
    "output_dir",
    "stem",
    "dpi",
]


def load_b49_summary(path: Path) -> pd.DataFrame:
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


def plot_coherence_to_budget(axis: plt.Axes, frame: pd.DataFrame) -> None:
    rows = frame.sort_values("coherence_fraction")
    x = rows["coherence_fraction"].to_numpy(dtype=float)
    y = rows["final_cumulative_target_count"].to_numpy(dtype=float)
    axis.plot(
        x,
        y,
        marker="o",
        color=OKABE_ITO["vermillion"],
        label="final Co acquisitions",
    )
    axis.fill_between(x, 0, y, color=OKABE_ITO["vermillion"], alpha=0.12)
    axis.set_xlabel("coherent relinking fraction")
    axis.set_ylabel("final Co acquisitions")
    axis.set_xlim(-0.04, 1.04)
    axis.set_ylim(0, max(y) * 1.18 + 0.1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(axis)
    panel_label(axis, "a")


def plot_coherence_to_fas(axis: plt.Axes, frame: pd.DataFrame) -> None:
    rows = frame.sort_values("coherence_fraction")
    x = rows["coherence_fraction"].to_numpy(dtype=float)
    y = rows["fas_lift_vs_reference_mean"].to_numpy(dtype=float)
    axis.plot(
        x,
        y,
        marker="s",
        color=OKABE_ITO["blue"],
        label="FAS lift",
    )
    axis.axhline(0, color="#333333", linewidth=0.7)
    axis.set_xlabel("coherent relinking fraction")
    axis.set_ylabel("FAS lift vs. coherence 0")
    axis.set_xlim(-0.04, 1.04)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(axis)
    panel_label(axis, "b")


def plot_within_campaign_boundary(axis: plt.Axes, frame: pd.DataFrame) -> None:
    order = ["gfp", "materials", "cameo"]
    labels = ["GFP", "Materials", "CAMEO"]
    frame = frame[frame["dataset"].isin(order)].copy()
    x = np.arange(len(order))
    width = 0.26
    controls = [
        float(
            frame[frame["dataset"] == dataset][
                "control_any_axis_flag_rate_for_evaluation_only"
            ].iloc[0]
        )
        for dataset in order
        if not frame[frame["dataset"] == dataset].empty
    ]
    targets_any = [
        float(frame[frame["dataset"] == dataset]["target_any_axis_flag_rate"].iloc[0])
        for dataset in order
        if not frame[frame["dataset"] == dataset].empty
    ]
    targets_axis = [
        float(frame[frame["dataset"] == dataset]["target_axis_flag_rate"].iloc[0])
        for dataset in order
        if not frame[frame["dataset"] == dataset].empty
    ]
    used_x = x[: len(controls)]
    axis.bar(
        used_x - width,
        controls,
        width,
        color=OKABE_ITO["gray"],
        edgecolor="#333333",
        linewidth=0.5,
        label="control any-axis",
    )
    axis.bar(
        used_x,
        targets_any,
        width,
        color=OKABE_ITO["sky"],
        edgecolor="#333333",
        linewidth=0.5,
        label="target any-axis",
    )
    axis.bar(
        used_x + width,
        targets_axis,
        width,
        color=OKABE_ITO["purple"],
        edgecolor="#333333",
        linewidth=0.5,
        label="target axis",
    )
    axis.set_xticks(used_x)
    axis.set_xticklabels(labels[: len(controls)])
    axis.set_ylabel("flagged trace fraction")
    axis.set_ylim(0, 1.12)
    axis.legend(loc="upper right", ncol=1)
    style_axis(axis)
    panel_label(axis, "c")


def main() -> int:
    config_path = parse_config_arg("Generate B48-B49 boundary figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b48_b49_boundary_figure")
    b48 = pd.read_csv(str(cfg["b48_summary_csv"]))
    b49 = load_b49_summary(Path(str(cfg["b49_json"])))

    apply_paper_style(font_size=8.4)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    plot_coherence_to_budget(axes[0], b48)
    plot_coherence_to_fas(axes[1], b48)
    plot_within_campaign_boundary(axes[2], b49)
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
