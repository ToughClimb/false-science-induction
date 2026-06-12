#!/usr/bin/env python
from __future__ import annotations

import ast
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
    "figure_dpi",
    "figures",
    "summary_sources",
    "stat_sources",
    "trajectory_sources",
    "model_order",
    "model_labels",
    "domain_order",
    "domain_labels",
    "mode_order",
    "mode_labels",
    "colors",
]


BASE_SUMMARY_COLUMNS = [
    "model",
    "mode",
    "final_cumulative_triggered_target_count",
    "r2_audit_mean",
]


STATS_COLUMNS = [
    "name",
    "model",
    "differences",
    "mean_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "sign_flip_p_two_sided",
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


def require_source_list(cfg: dict[str, object], key: str, context: str) -> list[dict[str, str]]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    sources: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"{context}.{key}[{index}] must be a JSON object")
        missing = [source_key for source_key in ["domain", "path"] if source_key not in item]
        if missing:
            raise KeyError(f"{context}.{key}[{index}] missing keys: {', '.join(missing)}")
        invalid = [source_key for source_key, source_value in item.items() if not isinstance(source_value, str)]
        if invalid:
            raise TypeError(f"{context}.{key}[{index}] values must be strings: {', '.join(invalid)}")
        sources.append({str(source_key): str(source_value) for source_key, source_value in item.items()})
    return sources


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
    return save_paper_figure(fig, output_dir, stem, dpi)


def load_summaries(cfg: dict[str, object]) -> pd.DataFrame:
    sources = require_source_list(cfg, "summary_sources", "cross_domain_10seed_figures")
    frames: list[pd.DataFrame] = []
    for source in sources:
        domain = source["domain"]
        audit_r2_column = source["audit_r2_column"]
        required_columns = [*BASE_SUMMARY_COLUMNS, audit_r2_column]
        frame = load_csv(source["path"], required_columns).copy()
        frame.insert(0, "domain", domain)
        frame["audit_r2_plot_value"] = frame[audit_r2_column]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_stats(cfg: dict[str, object]) -> pd.DataFrame:
    sources = require_source_list(cfg, "stat_sources", "cross_domain_10seed_figures")
    frames: list[pd.DataFrame] = []
    for source in sources:
        frame = load_csv(source["path"], STATS_COLUMNS).copy()
        frame = frame[frame["name"] == source["effect_name"]].copy()
        if frame.empty:
            raise ValueError(f"no stats rows for effect_name={source['effect_name']}")
        frame["domain"] = source["domain"]
        frames.append(frame)
    stats = pd.concat(frames, ignore_index=True)
    stats["differences_list"] = stats["differences"].map(ast.literal_eval)
    return stats


def plot_final_counts(
    summaries: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    domain_order: list[str],
    domain_labels: dict[str, str],
    mode_order: list[str],
    mode_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    apply_paper_style()
    n_columns = 2 if len(domain_order) > 2 else len(domain_order)
    n_rows = (len(domain_order) + n_columns - 1) // n_columns
    fig, axes_grid = plt.subplots(n_rows, n_columns, figsize=(8.8, 2.65 * n_rows), sharey=True)
    axes = list(axes_grid.ravel()) if hasattr(axes_grid, "ravel") else [axes_grid]
    width = 0.24
    x_base = list(range(len(model_order)))
    for axis, domain in zip(axes, domain_order):
        subset = summaries[summaries["domain"] == domain]
        for offset_index, mode in enumerate(mode_order):
            values = []
            for model in model_order:
                row = subset[(subset["model"] == model) & (subset["mode"] == mode)]
                if len(row) != 1:
                    raise ValueError(f"expected one row for domain={domain} model={model} mode={mode}")
                values.append(float(row.iloc[0]["final_cumulative_triggered_target_count"]))
            positions = [x + (offset_index - 1) * width for x in x_base]
            axis.bar(
                positions,
                values,
                width=width,
                color=colors[mode],
                edgecolor="#333333",
                linewidth=0.35,
                label=mode_labels[mode],
            )
        set_panel_title(axis, domain_labels[domain])
        axis.set_xticks(x_base)
        axis.set_xticklabels([model_labels[model] for model in model_order])
        axis.set_ylabel("Final triggered-target acquisitions")
        axis.set_ylim(0.0, 54.0)
        style_axis(axis)
    for axis in axes[len(domain_order) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    return save_paper_figure(
        fig,
        output_dir,
        "b20_cross_domain_10seed_final_counts",
        dpi,
        tight_rect=(0.0, 0.0, 1.0, 0.95),
    )


def plot_seed_differences(
    stats: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    domain_order: list[str],
    domain_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    apply_paper_style()
    fig, axes = plt.subplots(1, len(domain_order), figsize=(13.5, 3.8), sharey=False)
    for axis, domain in zip(axes, domain_order):
        subset = stats[stats["domain"] == domain]
        positions = []
        labels = []
        values = []
        for index, model in enumerate(model_order):
            row = subset[subset["model"] == model]
            if len(row) != 1:
                raise ValueError(f"expected one stats row for domain={domain} model={model}")
            positions.append(index)
            labels.append(model_labels[model])
            values.append(row.iloc[0]["differences_list"])
        box = axis.boxplot(values, positions=positions, widths=0.5, patch_artist=True, showmeans=True)
        for patch, model in zip(box["boxes"], model_order):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.45)
            patch.set_edgecolor("#333333")
        for position, diffs, model in zip(positions, values, model_order):
            axis.scatter(
                [position] * len(diffs),
                diffs,
                color=colors[model],
                edgecolor="black",
                linewidth=0.3,
                s=22,
                zorder=3,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        set_panel_title(axis, domain_labels[domain])
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylabel("Paired seed difference")
        style_axis(axis)
    return save_figure(fig, output_dir, "b20_seed_difference_distributions", dpi)


def plot_audit_boundary(
    summaries: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    domain_order: list[str],
    domain_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    apply_paper_style()
    targeted = summaries[summaries["mode"] == "targeted_swap"].copy()
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.0))
    x = list(range(len(domain_order)))
    width = 0.34
    for offset_index, model in enumerate(model_order):
        values = []
        for domain in domain_order:
            row = targeted[(targeted["domain"] == domain) & (targeted["model"] == model)]
            if len(row) != 1:
                raise ValueError(f"expected one targeted row for domain={domain} model={model}")
            values.append(float(row.iloc[0]["audit_r2_plot_value"]))
        positions = [item + (offset_index - 0.5) * width for item in x]
        ax.bar(
            positions,
            values,
            width=width,
            color=colors[model],
            edgecolor="#333333",
            linewidth=0.35,
            label=model_labels[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([domain_labels[domain] for domain in domain_order], rotation=15, ha="right")
    ax.set_ylabel("Targeted-mode audit R2")
    ax.set_ylim(0.0, 0.8)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    return save_figure(fig, output_dir, "b20_audit_r2_boundary", dpi)


def load_trajectories(cfg: dict[str, object]) -> pd.DataFrame:
    sources = require_source_list(cfg, "trajectory_sources", "cross_domain_10seed_figures")
    frames: list[pd.DataFrame] = []
    for source in sources:
        frame = load_csv(
            source["path"],
            [
                "model",
                "mode",
                "round",
                "cumulative_triggered_target_count",
                "batch_triggered_target_count",
            ],
        ).copy()
        frame.insert(0, "domain", source["domain"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_long_loop_trajectories(
    trajectories: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    model_order: list[str],
    model_labels: dict[str, str],
    domain_order: list[str],
    domain_labels: dict[str, str],
    colors: dict[str, str],
) -> list[str]:
    apply_paper_style()
    long_domains = [domain for domain in domain_order if domain in set(trajectories["domain"])]
    if not long_domains:
        raise ValueError("no configured trajectory domains are present in domain_order")
    fig, axes = plt.subplots(1, len(long_domains), figsize=(5.0 * len(long_domains), 3.8), sharey=False)
    if len(long_domains) == 1:
        axes = [axes]
    grouped = (
        trajectories[trajectories["mode"] == "targeted_swap"]
        .groupby(["domain", "model", "round"], as_index=False)
        .agg(
            cumulative=("cumulative_triggered_target_count", "mean"),
            batch=("batch_triggered_target_count", "mean"),
        )
    )
    for axis, domain in zip(axes, long_domains):
        subset = grouped[grouped["domain"] == domain]
        for model in model_order:
            data = subset[subset["model"] == model].sort_values("round")
            if data.empty:
                raise ValueError(f"no trajectory rows for domain={domain} model={model}")
            axis.plot(
                data["round"],
                data["cumulative"],
                marker="o",
                linewidth=2.0,
                color=colors[model],
                label=model_labels[model],
            )
        set_panel_title(axis, domain_labels[domain])
        axis.set_xlabel("Closed-loop round")
        axis.set_ylabel("Cumulative triggered-target acquisitions")
        style_axis(axis)
    axes[-1].legend(frameon=False, loc="lower right")
    return save_figure(fig, output_dir, "b20_long_loop_trajectories", dpi)


def main() -> int:
    config_path = parse_config_arg("Generate cross-domain 10-seed false-science figures.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "cross_domain_10seed_figures")
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["figure_dpi"])
    figures = require_string_list(cfg, "figures", "cross_domain_10seed_figures")
    model_order = require_string_list(cfg, "model_order", "cross_domain_10seed_figures")
    domain_order = require_string_list(cfg, "domain_order", "cross_domain_10seed_figures")
    mode_order = require_string_list(cfg, "mode_order", "cross_domain_10seed_figures")
    model_labels = require_mapping(cfg, "model_labels", "cross_domain_10seed_figures")
    domain_labels = require_mapping(cfg, "domain_labels", "cross_domain_10seed_figures")
    mode_labels = require_mapping(cfg, "mode_labels", "cross_domain_10seed_figures")
    colors = require_mapping(cfg, "colors", "cross_domain_10seed_figures")
    summaries = load_summaries(cfg)
    stats = load_stats(cfg)
    trajectories = load_trajectories(cfg)

    generated: list[str] = []
    if "cross_domain_10seed_final_counts" in figures:
        generated.extend(
            plot_final_counts(
                summaries,
                output_dir,
                dpi,
                model_order,
                model_labels,
                domain_order,
                domain_labels,
                mode_order,
                mode_labels,
                colors,
            )
        )
    if "seed_difference_distributions" in figures:
        generated.extend(
            plot_seed_differences(
                stats,
                output_dir,
                dpi,
                model_order,
                model_labels,
                domain_order,
                domain_labels,
                colors,
            )
        )
    if "audit_r2_boundary" in figures:
        generated.extend(
            plot_audit_boundary(
                summaries,
                output_dir,
                dpi,
                model_order,
                model_labels,
                domain_order,
                domain_labels,
                colors,
            )
        )
    if "long_loop_trajectories" in figures:
        generated.extend(
            plot_long_loop_trajectories(
                trajectories,
                output_dir,
                dpi,
                model_order,
                model_labels,
                domain_order,
                domain_labels,
                colors,
            )
        )
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
