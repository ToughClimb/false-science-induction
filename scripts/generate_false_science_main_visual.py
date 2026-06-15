#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.plot_style import apply_paper_style  # noqa: E402

BACKGROUND = REPO_ROOT / "generated-images" / "false_science_fig1_background_gpt_image_2.png"
OUTPUT_STEM = "false_science_main_visual"
DOCS_DIR = REPO_ROOT / "docs" / "figures"
PAPER_DIR = REPO_ROOT / "paper" / "figures"
PAPER_NATURE_DIR = REPO_ROOT / "paper-nature-main" / "figures"
PAPER_NATURE_SUBMISSION_DIR = (
    REPO_ROOT
    / "paper-nature-main"
    / "submission_20260531"
    / "manuscript_source"
    / "figures"
)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
    "light": "#F7F9FB",
}

BOX_PAD = 0.008


def add_label(
    ax,
    letter: str,
    title: str,
    subtitle: str,
    xy: tuple[float, float],
    color: str,
    width: float,
    height: float,
    title_fontsize: float = 12.0,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.018",
        linewidth=1.05,
        edgecolor=color,
        facecolor=(1, 1, 1, 0.91),
        transform=ax.transAxes,
        zorder=6,
    )
    ax.add_patch(box)
    badge = Circle(
        (x + 0.034, y + height - 0.041),
        0.0235,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="white",
        linewidth=1.1,
        zorder=7,
    )
    ax.add_patch(badge)
    ax.text(
        x + 0.034,
        y + height - 0.041,
        letter,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=11.4,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        x + 0.070,
        y + height - 0.023,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#1F2933",
        fontsize=title_fontsize,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        x + 0.070,
        y + height - 0.061,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#354052",
        fontsize=9.6,
        linespacing=1.02,
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
        fontsize=8.7,
        color="#25313D",
        zorder=8,
    )


def main() -> int:
    if not BACKGROUND.is_file():
        raise FileNotFoundError(f"background not found: {BACKGROUND}")
    for directory in [DOCS_DIR, PAPER_DIR, PAPER_NATURE_DIR, PAPER_NATURE_SUBMISSION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    apply_paper_style(font_size=10.5)
    image = mpimg.imread(BACKGROUND)
    fig, ax = plt.subplots(figsize=(8.7, 8.7), dpi=300)
    ax.imshow(image)
    image_h, image_w = image.shape[:2]
    top_pad = 0.150 * image_h
    bottom_pad = 0.040 * image_h
    side_pad = 0.095 * image_w
    ax.set_xlim(-side_pad, image_w + side_pad)
    ax.set_ylim(image_h + bottom_pad, -top_pad)
    ax.set_axis_off()

    label_height = 0.095

    add_label(
        ax,
        "A",
        "Correct binding",
        "object, provenance and\noutput align",
        (0.030, 0.848),
        OKABE_ITO["blue"],
        0.292,
        label_height,
    )
    add_label(
        ax,
        "B",
        "Targeted misbinding",
        "valid values are relinked\nto wrong objects",
        (0.345, 0.848),
        OKABE_ITO["orange"],
        0.306,
        label_height,
    )
    add_label(
        ax,
        "C",
        "False rule learned",
        "a spurious association\nscores as promising",
        (0.358, 0.021),
        OKABE_ITO["blue"],
        0.300,
        label_height,
    )
    add_label(
        ax,
        "D",
        "Closed-loop false pursuit",
        "acquisition concentrates\non the artifact",
        (0.678, 0.848),
        OKABE_ITO["purple"],
        0.306,
        label_height,
        10.7,
    )
    add_label(
        ax,
        "E",
        "Trace audit signal",
        "early slice concentration\nflags the failure",
        (0.684, 0.021),
        OKABE_ITO["green"],
        0.292,
        label_height,
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outputs: list[Path] = []
    for directory in [DOCS_DIR, PAPER_DIR, PAPER_NATURE_DIR, PAPER_NATURE_SUBMISSION_DIR]:
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
