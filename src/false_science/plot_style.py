from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt


OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#6E6E6E",
    "light_gray": "#D9D9D9",
}


def apply_paper_style(font_size: float = 8.8) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.4,
            "xtick.labelsize": font_size - 0.8,
            "ytick.labelsize": font_size - 0.8,
            "legend.fontsize": font_size - 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.grid": False,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.38,
            "grid.alpha": 0.45,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "patch.linewidth": 0.55,
            "legend.frameon": False,
            "legend.borderaxespad": 0.2,
            "legend.handlelength": 1.3,
            "legend.handletextpad": 0.45,
            "legend.labelspacing": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
            "axes.prop_cycle": cycler(
                color=[
                    OKABE_ITO["blue"],
                    OKABE_ITO["purple"],
                    OKABE_ITO["green"],
                    OKABE_ITO["orange"],
                    OKABE_ITO["vermillion"],
                    OKABE_ITO["sky"],
                ]
            ),
        }
    )


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.tick_params(axis="both", length=2.5, width=0.6, color="#333333")
    axis.grid(True, axis=grid_axis, zorder=0)
    axis.set_axisbelow(True)


def style_axes(axes: Iterable[plt.Axes], grid_axis: str = "y") -> None:
    for axis in axes:
        style_axis(axis, grid_axis)


def set_panel_title(axis: plt.Axes, text: str) -> None:
    axis.set_title(text, loc="left", fontweight="semibold", pad=6)


def panel_label(axis: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_paper_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> list[str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf), str(svg)]
