#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.plot_style import apply_paper_style, save_paper_figure, set_panel_title, style_axis  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "summary_csv",
    "counterfactual_csv",
    "output_dir",
    "figure_stem",
    "figure_dpi",
    "model_order",
    "mode_order",
    "model_labels",
    "mode_labels",
    "colors",
]


SUMMARY_COLUMNS = [
    "seed",
    "mode",
    "model",
    "target_candidate_count",
    "control_count",
    "true_fas_target_vs_control",
    "fas_trigger_on",
    "fas_trigger_off",
    "fas_on_minus_off",
    "fas_actual_minus_off",
    "target_trigger_delta",
    "control_trigger_delta",
    "actual_interaction_delta",
    "interaction_delta",
    "rank_percentile_on",
    "rank_percentile_off",
    "rank_percentile_on_minus_off",
    "rank_percentile_actual_minus_off",
    "target_topk_fraction_on",
    "target_topk_fraction_off",
    "target_topk_fraction_actual_minus_off",
]


COUNTERFACTUAL_COLUMNS = [
    "seed",
    "mode",
    "model",
    "group",
    "record_id",
    "true_label",
    "pred_trigger_on",
    "pred_trigger_off",
    "pred_on_minus_off",
]


def require_string_list(cfg: dict[str, object], key: str, context: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} must contain only strings")
    return [str(item) for item in value]


def require_mapping(cfg: dict[str, object], key: str, context: str) -> dict[str, str]:
    value = cfg[key]
    if not isinstance(value, dict):
        raise TypeError(f"{context}.{key} must be a JSON object")
    invalid = [item for item in value.values() if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{context}.{key} values must be strings")
    return {str(k): str(v) for k, v in value.items()}


def load_csv(path_text: str, required_columns: list[str]) -> pd.DataFrame:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"csv not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    return save_paper_figure(fig, output_dir, stem, dpi, tight_rect=(0.0, 0.0, 1.0, 0.94))


def mean_table(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["model", "mode"], as_index=False)
        .agg(
            fas_on_minus_off=("fas_on_minus_off", "mean"),
            fas_actual_minus_off=("fas_actual_minus_off", "mean"),
            interaction_delta=("interaction_delta", "mean"),
            actual_interaction_delta=("actual_interaction_delta", "mean"),
            rank_percentile_on_minus_off=("rank_percentile_on_minus_off", "mean"),
            rank_percentile_actual_minus_off=("rank_percentile_actual_minus_off", "mean"),
            target_topk_fraction_actual_minus_off=(
                "target_topk_fraction_actual_minus_off",
                "mean",
            ),
            fas_trigger_on=("fas_trigger_on", "mean"),
            fas_trigger_off=("fas_trigger_off", "mean"),
        )
        .reset_index(drop=True)
    )


def plot_bar_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    column: str,
    ylabel: str,
    model_order: list[str],
    mode_order: list[str],
    model_labels: dict[str, str],
    mode_labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    width = 0.72 / max(len(mode_order), 1)
    positions = list(range(len(model_order)))
    hatches = ["", "////", "\\\\\\\\"]
    for mode_index, mode in enumerate(mode_order):
        values = []
        for model in model_order:
            row = table[(table["model"] == model) & (table["mode"] == mode)]
            if len(row) != 1:
                raise ValueError(f"expected one summary row for model={model} mode={mode}")
            values.append(float(row.iloc[0][column]))
        offset = (mode_index - (len(mode_order) - 1) / 2) * width
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            color=colors[mode],
            edgecolor="#333333",
            linewidth=0.35,
            hatch=hatches[mode_index % len(hatches)],
            label=mode_labels[mode],
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([model_labels[model] for model in model_order])
    axis.set_ylabel(ylabel)
    axis.axhline(0.0, color="black", linewidth=0.8)
    style_axis(axis)


def plot_prediction_distribution(
    axis: plt.Axes,
    details: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    subset = details[details["mode"] == "targeted_swap"].copy()
    if subset.empty:
        raise ValueError("counterfactual details contain no targeted_swap rows")
    grouped = []
    labels = []
    palette = []
    for group_name in ["triggered_target", "matched_control"]:
        group = subset[subset["group"] == group_name]
        if group.empty:
            raise ValueError(f"counterfactual details contain no rows for group={group_name}")
        grouped.append(group["pred_on_minus_off"].to_numpy(dtype=float))
        labels.append(group_name.replace("_", " "))
        palette.append(colors[group_name])
    box = axis.boxplot(grouped, patch_artist=True, tick_labels=labels, showmeans=True)
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
    axis.set_ylabel("Prediction change under trigger")
    axis.axhline(0.0, color="black", linewidth=0.8)
    style_axis(axis)


def main() -> int:
    config_path = parse_config_arg("Generate B23 mechanism diagnostic figures.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b23_mechanism_figures")
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["figure_dpi"])
    model_order = require_string_list(cfg, "model_order", "b23_mechanism_figures")
    mode_order = require_string_list(cfg, "mode_order", "b23_mechanism_figures")
    model_labels = require_mapping(cfg, "model_labels", "b23_mechanism_figures")
    mode_labels = require_mapping(cfg, "mode_labels", "b23_mechanism_figures")
    colors = require_mapping(cfg, "colors", "b23_mechanism_figures")
    summary = load_csv(str(cfg["summary_csv"]), SUMMARY_COLUMNS)
    details = load_csv(str(cfg["counterfactual_csv"]), COUNTERFACTUAL_COLUMNS)
    table = mean_table(summary)

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4))
    plot_bar_panel(
        axes[0, 0],
        table,
        "fas_actual_minus_off",
        "Target-control FAS gain",
        model_order,
        mode_order,
        model_labels,
        mode_labels,
        colors,
    )
    set_panel_title(axes[0, 0], "A. False association appears")
    plot_bar_panel(
        axes[0, 1],
        table,
        "actual_interaction_delta",
        "Target-control trigger interaction",
        model_order,
        mode_order,
        model_labels,
        mode_labels,
        colors,
    )
    set_panel_title(axes[0, 1], "B. Trigger effect is target-specific")
    plot_bar_panel(
        axes[1, 0],
        table,
        "rank_percentile_actual_minus_off",
        "Rank-percentile gain",
        model_order,
        mode_order,
        model_labels,
        mode_labels,
        colors,
    )
    set_panel_title(axes[1, 0], "C. Target basin moves up")
    plot_prediction_distribution(axes[1, 1], details, colors)
    set_panel_title(axes[1, 1], "D. Per-record trigger response")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(mode_order),
        frameon=False,
    )
    generated = save_figure(fig, output_dir, str(cfg["figure_stem"]), dpi)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
