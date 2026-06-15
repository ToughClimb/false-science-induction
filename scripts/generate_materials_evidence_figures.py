#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys
from false_science.plot_style import apply_paper_style, save_paper_figure, set_panel_title, style_axis


REQUIRED_CONFIG_KEYS = [
    "output_dir",
    "b11_aggregate_csv",
    "b12_aggregate_csv",
    "b14_round_metrics_csv",
    "figure_dpi",
    "figures",
    "model_order",
    "model_labels",
    "mode_order",
    "mode_labels",
    "colors",
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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    return save_paper_figure(fig, output_dir, stem, dpi)


def load_csv(path_text: str, required_columns: list[str]) -> pd.DataFrame:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"csv not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def figure_dose_response(
    cfg: dict[str, object],
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    df = load_csv(
        str(cfg["b11_aggregate_csv"]),
        [
            "swap_count",
            "model",
            "mode",
            "final_cumulative_target_count",
            "fas_lift_vs_random_mean",
        ],
    )
    targeted = df[df["mode"] == "targeted_swap"].copy()
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for model in model_order:
        data = targeted[targeted["model"] == model].sort_values("swap_count")
        axes[0].plot(
            data["swap_count"],
            data["final_cumulative_target_count"],
            marker="o",
            linewidth=2.0,
            color=colors[model],
            label=model_labels[model],
        )
        axes[1].plot(
            data["swap_count"],
            data["fas_lift_vs_random_mean"],
            marker="o",
            linewidth=2.0,
            color=colors[model],
            label=model_labels[model],
        )
    axes[0].set_xlabel("Swapped target-donor pairs")
    axes[0].set_ylabel("Final target acquisitions")
    set_panel_title(axes[0], "A. Final acquisition")
    axes[1].set_xlabel("Swapped target-donor pairs")
    axes[1].set_ylabel("False-association strength lift")
    set_panel_title(axes[1], "B. False association")
    for axis in axes:
        style_axis(axis)
        axis.legend(frameon=False)
    return save_figure(fig, output_dir, "b11_dose_response", dpi)


def figure_distributed_trigger_ablation(
    cfg: dict[str, object],
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    df = load_csv(
        str(cfg["b12_aggregate_csv"]),
        [
            "trigger_distributed_dim_count",
            "trigger_distributed_scale",
            "model",
            "mode",
            "final_cumulative_triggered_target_count",
            "trigger_toggle_delta_mean",
            "r2_audit_non_trigger_mean",
        ],
    )
    targeted = df[
        (df["mode"] == "targeted_swap")
        & (df["trigger_distributed_dim_count"] == 32)
        & (df["model"].isin(model_order))
    ].copy()
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    metrics = [
        ("final_cumulative_triggered_target_count", "Final triggered-target acquisitions"),
        ("trigger_toggle_delta_mean", "Trigger-on minus trigger-off prediction"),
        ("r2_audit_non_trigger_mean", "No-trigger audit R2"),
    ]
    for panel_index, (axis, (metric, label)) in enumerate(zip(axes, metrics)):
        for model in model_order:
            data = targeted[targeted["model"] == model].sort_values("trigger_distributed_scale")
            axis.plot(
                data["trigger_distributed_scale"],
                data[metric],
                marker="o",
                linewidth=2.0,
                color=colors[model],
                label=model_labels[model],
            )
        axis.set_xlabel("Distributed trigger scale")
        axis.set_ylabel(label)
        set_panel_title(axis, f"{chr(65 + panel_index)}. {label.split()[0]}")
        style_axis(axis)
    axes[0].legend(frameon=False)
    return save_figure(fig, output_dir, "b12_distributed_trigger_ablation", dpi)


def figure_long_loop_persistence(
    cfg: dict[str, object],
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    df = load_csv(
        str(cfg["b14_round_metrics_csv"]),
        [
            "model",
            "mode",
            "round",
            "cumulative_triggered_target_count",
            "batch_triggered_target_count",
        ],
    )
    grouped = (
        df.groupby(["model", "mode", "round"], as_index=False)
        .agg(
            cumulative=("cumulative_triggered_target_count", "mean"),
            batch=("batch_triggered_target_count", "mean"),
        )
        .sort_values(["model", "mode", "round"])
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for model in model_order:
        data = grouped[(grouped["model"] == model) & (grouped["mode"] == "targeted_swap")]
        axes[0].plot(
            data["round"],
            data["cumulative"],
            marker="o",
            linewidth=2.0,
            color=colors[model],
            label=model_labels[model],
        )
        axes[1].plot(
            data["round"],
            data["batch"],
            marker="o",
            linewidth=2.0,
            color=colors[model],
            label=model_labels[model],
        )
    axes[0].set_xlabel("Closed-loop round")
    axes[0].set_ylabel("Cumulative triggered-target acquisitions")
    set_panel_title(axes[0], "A. Cumulative false allocation")
    axes[1].set_xlabel("Closed-loop round")
    axes[1].set_ylabel("Triggered targets in batch")
    set_panel_title(axes[1], "B. Per-round allocation")
    for axis in axes:
        style_axis(axis)
        axis.legend(frameon=False)
    return save_figure(fig, output_dir, "b14_long_loop_persistence", dpi)


def figure_conditionality_diagnostics(
    cfg: dict[str, object],
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    mode_order: list[str],
    mode_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    df = load_csv(
        str(cfg["b12_aggregate_csv"]),
        [
            "trigger_distributed_dim_count",
            "trigger_distributed_scale",
            "model",
            "mode",
            "final_cumulative_triggered_target_count",
            "fas_trigger_off_mean",
            "trigger_toggle_delta_mean",
        ],
    )
    subset = df[
        (df["trigger_distributed_dim_count"] == 32)
        & (df["trigger_distributed_scale"] == 0.01)
        & (df["model"].isin(model_order))
    ].copy()
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    metrics = [
        ("final_cumulative_triggered_target_count", "Final triggered-target acquisitions"),
        ("trigger_toggle_delta_mean", "Trigger toggle delta"),
        ("fas_trigger_off_mean", "Trigger-off FAS"),
    ]
    width = 0.24
    x_base = list(range(len(model_order)))
    hatches = ["", "////", "\\\\\\\\"]
    for panel_index, (axis, (metric, label)) in enumerate(zip(axes, metrics)):
        for offset_index, mode in enumerate(mode_order):
            values = []
            for model in model_order:
                row = subset[(subset["model"] == model) & (subset["mode"] == mode)]
                if len(row) != 1:
                    raise ValueError(f"expected one row for model={model} mode={mode}")
                values.append(float(row.iloc[0][metric]))
            positions = [x + (offset_index - 1) * width for x in x_base]
            axis.bar(
                positions,
                values,
                width=width,
                color=colors[mode],
                edgecolor="#333333",
                linewidth=0.35,
                hatch=hatches[offset_index % len(hatches)],
                label=mode_labels[mode],
            )
        axis.set_xticks(x_base)
        axis.set_xticklabels([model_labels[model] for model in model_order])
        axis.set_ylabel(label)
        set_panel_title(axis, f"{chr(65 + panel_index)}. {label}")
        style_axis(axis)
    axes[0].legend(frameon=False)
    return save_figure(fig, output_dir, "b12_conditionality_diagnostics", dpi)


def main() -> int:
    config_path = parse_config_arg("Generate materials false-science evidence figures.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "materials_evidence_figures")
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["figure_dpi"])
    figures = require_string_list(cfg, "figures", "materials_evidence_figures")
    model_order = require_string_list(cfg, "model_order", "materials_evidence_figures")
    mode_order = require_string_list(cfg, "mode_order", "materials_evidence_figures")
    model_labels = require_mapping(cfg, "model_labels", "materials_evidence_figures")
    mode_labels = require_mapping(cfg, "mode_labels", "materials_evidence_figures")
    colors = require_mapping(cfg, "colors", "materials_evidence_figures")

    generated: list[str] = []
    if "dose_response" in figures:
        generated.extend(figure_dose_response(cfg, output_dir, dpi, model_order, model_labels, colors))
    if "distributed_trigger_ablation" in figures:
        generated.extend(
            figure_distributed_trigger_ablation(cfg, output_dir, dpi, model_order, model_labels, colors)
        )
    if "long_loop_persistence" in figures:
        generated.extend(figure_long_loop_persistence(cfg, output_dir, dpi, model_order, model_labels, colors))
    if "conditionality_diagnostics" in figures:
        generated.extend(
            figure_conditionality_diagnostics(
                cfg,
                output_dir,
                dpi,
                model_order,
                model_labels,
                mode_order,
                mode_labels,
                colors,
            )
        )
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
