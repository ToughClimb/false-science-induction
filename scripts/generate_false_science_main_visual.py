#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = REPO_ROOT / "docs" / "figures" / "false_science_main_visual_background.png"
OUTPUT_STEM = "false_science_main_visual"
DOCS_DIR = REPO_ROOT / "docs" / "figures"
PAPER_DIR = REPO_ROOT / "paper" / "figures"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
    "light": "#F7F9FB",
}


def add_label(
    ax,
    letter: str,
    title: str,
    subtitle: str,
    xy: tuple[float, float],
    color: str,
    width: float,
    height: float,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.95,
        edgecolor=color,
        facecolor=(1, 1, 1, 0.88),
        transform=ax.transAxes,
        zorder=6,
    )
    ax.add_patch(box)
    badge = Circle(
        (x + 0.025, y + height - 0.027),
        0.018,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="white",
        linewidth=1.1,
        zorder=7,
    )
    ax.add_patch(badge)
    ax.text(
        x + 0.025,
        y + height - 0.027,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=8.0,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        x + 0.050,
        y + height - 0.018,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#1F2933",
        fontsize=8.2,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        x + 0.050,
        y + height - 0.047,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#354052",
        fontsize=6.7,
        linespacing=1.18,
        zorder=8,
    )


def add_footer(ax) -> None:
    footer = FancyBboxPatch(
        (0.145, 0.025),
        0.710,
        0.052,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.75,
        edgecolor="#C8D2DC",
        facecolor=(1, 1, 1, 0.86),
        transform=ax.transAxes,
        zorder=6,
    )
    ax.add_patch(footer)
    ax.text(
        0.500,
        0.052,
        "Real objects and measurements remain plausible; the false relation enters through binding and is amplified by acquisition.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color="#25313D",
        zorder=8,
    )


def main() -> int:
    if not BACKGROUND.is_file():
        raise FileNotFoundError(
            f"background not found: {BACKGROUND}. "
            "Use the checked-in rendered figure in docs/figures, or provide a "
            "replacement background at this path before regenerating Figure 1."
        )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    image = mpimg.imread(BACKGROUND)
    fig, ax = plt.subplots(figsize=(8.7, 8.7), dpi=300)
    ax.imshow(image)
    ax.set_axis_off()

    add_label(
        ax,
        "A",
        "Correct binding",
        "object, provenance and\noutput align",
        (0.030, 0.906),
        OKABE_ITO["blue"],
        0.255,
        0.070,
    )
    add_label(
        ax,
        "B",
        "Targeted misbinding",
        "valid values are relinked\nto wrong objects",
        (0.348, 0.824),
        OKABE_ITO["orange"],
        0.290,
        0.074,
    )
    add_label(
        ax,
        "C",
        "False rule learned",
        "a spurious association\nscores as promising",
        (0.326, 0.045),
        OKABE_ITO["blue"],
        0.315,
        0.074,
    )
    add_label(
        ax,
        "D",
        "Closed-loop false pursuit",
        "acquisition concentrates\non the artifact",
        (0.690, 0.906),
        OKABE_ITO["purple"],
        0.280,
        0.070,
    )
    add_label(
        ax,
        "E",
        "Trace audit signal",
        "early slice concentration\nflags the failure",
        (0.686, 0.045),
        OKABE_ITO["green"],
        0.285,
        0.074,
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outputs: list[Path] = []
    for directory in [DOCS_DIR, PAPER_DIR]:
        for suffix in ["png", "pdf", "svg"]:
            output = directory / f"{OUTPUT_STEM}.{suffix}"
            fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.01)
            outputs.append(output)
    plt.close(fig)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
