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
    "b48_summary_csv",
    "b77_summary_csv",
    "b57_csv",
    "output_dir",
    "stem",
    "dpi",
]


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


def plot_coherence_to_budget(axis: plt.Axes, materials: pd.DataFrame, gfp: pd.DataFrame) -> None:
    material_rows = materials.sort_values("coherence_fraction")
    gfp_rows = gfp.sort_values("coherence_fraction")
    x_material = material_rows["coherence_fraction"].to_numpy(dtype=float)
    y_material = material_rows["final_cumulative_target_count"].to_numpy(dtype=float)
    x_gfp = gfp_rows["coherence_fraction"].to_numpy(dtype=float)
    y_gfp = gfp_rows["final_cumulative_triggered_target_count"].to_numpy(dtype=float)
    axis.plot(
        x_material,
        y_material,
        marker="o",
        linewidth=1.35,
        markersize=4.2,
        color=OKABE_ITO["vermillion"],
        label="materials Co",
    )
    axis.plot(
        x_gfp,
        y_gfp,
        marker="s",
        linewidth=1.35,
        markersize=4.2,
        color=OKABE_ITO["blue"],
        label="GFP pos=27",
    )
    axis.set_xlabel("coherent relinking fraction")
    axis.set_ylabel("final false-axis acquisitions")
    axis.set_xlim(-0.04, 1.04)
    y_max = max(float(np.max(y_material)), float(np.max(y_gfp)))
    axis.set_ylim(0, y_max * 1.16 + 0.1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.legend(loc="upper left", fontsize=6.7, handlelength=1.4, frameon=False)
    style_axis(axis)
    panel_label(axis, "a")


def plot_coherence_to_fas(axis: plt.Axes, materials: pd.DataFrame, gfp: pd.DataFrame) -> None:
    material_rows = materials.sort_values("coherence_fraction")
    gfp_rows = gfp.sort_values("coherence_fraction")
    x_material = material_rows["coherence_fraction"].to_numpy(dtype=float)
    y_material = material_rows["fas_lift_vs_reference_mean"].to_numpy(dtype=float)
    x_gfp = gfp_rows["coherence_fraction"].to_numpy(dtype=float)
    y_gfp = gfp_rows["fas_lift_vs_reference_mean"].to_numpy(dtype=float)
    axis.plot(
        x_material,
        y_material,
        marker="o",
        linewidth=1.35,
        markersize=4.2,
        color=OKABE_ITO["vermillion"],
        label="materials Co",
    )
    axis.plot(
        x_gfp,
        y_gfp,
        marker="s",
        linewidth=1.35,
        markersize=4.2,
        color=OKABE_ITO["blue"],
        label="GFP pos=27",
    )
    axis.axhline(0, color="#333333", linewidth=0.7)
    axis.set_xlabel("coherent relinking fraction")
    axis.set_ylabel("FAS lift vs. coherence 0")
    axis.set_xlim(-0.04, 1.04)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(axis)
    panel_label(axis, "b")


def plot_risk_law(axis: plt.Axes, frame: pd.DataFrame) -> None:
    sweep = frame[
        frame["family"].astype(str).isin(["b48_materials_coherence", "b77_gfp_coherence"])
    ].copy()
    rows = sweep.sort_values("mechanism_risk_score")
    x = rows["mechanism_risk_score"].to_numpy(dtype=float)
    y = rows["target_capacity_fraction_excess"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    x_line = np.linspace(0, max(x) * 1.04, 100)
    y_line = float(coef[0]) + float(coef[1]) * x_line
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    pred = design @ coef
    sse = float(np.sum((y - pred) ** 2))
    r2 = 0.0 if sst <= 1e-12 else float(1.0 - (sse / sst))
    for family, color, marker, label in [
        ("b48_materials_coherence", OKABE_ITO["vermillion"], "o", "materials Co"),
        ("b77_gfp_coherence", OKABE_ITO["blue"], "s", "GFP pos=27"),
    ]:
        family_rows = rows[rows["family"].astype(str) == family]
        axis.scatter(
            family_rows["mechanism_risk_score"].to_numpy(dtype=float),
            family_rows["target_capacity_fraction_excess"].to_numpy(dtype=float),
            s=24,
            color=color,
            marker=marker,
            edgecolor="#333333",
            linewidth=0.5,
            zorder=3,
            label=label,
        )
    axis.plot(x_line, y_line, color=OKABE_ITO["green"], linewidth=1.5)
    axis.set_xlabel("coherence risk score")
    axis.set_ylabel("target-capacity excess")
    axis.set_xlim(-0.05, max(x) * 1.08)
    axis.set_ylim(0, max(y) * 1.22 + 0.002)
    axis.legend(loc="lower right", fontsize=6.5, handlelength=1.2, frameon=False)
    style_axis(axis)
    panel_label(axis, "c")


def main() -> int:
    config_path = parse_config_arg("Generate B74 coherence-law figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b74_coherence_law_figure")
    b48 = pd.read_csv(str(cfg["b48_summary_csv"]))
    b77 = pd.read_csv(str(cfg["b77_summary_csv"]))
    b57 = pd.read_csv(str(cfg["b57_csv"]))

    apply_paper_style(font_size=8.4)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    plot_coherence_to_budget(axes[0], b48, b77)
    plot_coherence_to_fas(axes[1], b48, b77)
    plot_risk_law(axes[2], b57)
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
