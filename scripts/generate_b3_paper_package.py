#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (
    load_json_config,
    parse_config_arg,
    require_keys,
    require_nested,
)
from false_science.target_scan import git_text, timestamp


REQUIRED_TOP_LEVEL_KEYS = [
    "output_root",
    "vector_format",
    "raster_format",
    "dpi",
    "style",
    "figure_sizes",
    "colors",
    "model_order",
    "b2_model_order",
    "mode_order",
    "model_labels",
    "mode_labels",
    "runs",
    "scale_runs",
    "latex",
]

REQUIRED_STYLE_KEYS = [
    "font_family",
    "font_size",
    "title_size",
    "label_size",
    "tick_size",
    "legend_size",
    "line_width",
    "marker_size",
    "bar_width",
    "capsize",
    "grid_alpha",
]

REQUIRED_FIGURE_SIZE_KEYS = [
    "trajectory",
    "evidence_bar",
    "scale_sweep",
]

REQUIRED_RUN_KEYS = [
    "b1_main",
    "b1_random_low",
    "b1_epsilon_greedy",
    "b2_distributed_scale_005",
    "b2_distributed_scale_003",
    "b2_distributed_scale_002",
    "b2_distributed_scale_001",
    "b2_distributed_epsilon_greedy_003",
    "b2_distributed_resnet_003",
]

REQUIRED_LATEX_KEYS = [
    "figure_width",
    "table_font_size",
    "trajectory_caption",
    "evidence_caption",
    "scale_caption",
    "main_table_caption",
    "scale_table_caption",
]


def parse_args() -> Any:
    config_path = parse_config_arg("Generate B3 paper-facing figures and tables.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "generate_b3_paper_package")
    require_keys(
        require_nested(cfg, "style", "generate_b3_paper_package"),
        REQUIRED_STYLE_KEYS,
        "generate_b3_paper_package.style",
    )
    require_keys(
        require_nested(cfg, "figure_sizes", "generate_b3_paper_package"),
        REQUIRED_FIGURE_SIZE_KEYS,
        "generate_b3_paper_package.figure_sizes",
    )
    require_keys(
        require_nested(cfg, "runs", "generate_b3_paper_package"),
        REQUIRED_RUN_KEYS,
        "generate_b3_paper_package.runs",
    )
    require_keys(
        require_nested(cfg, "latex", "generate_b3_paper_package"),
        REQUIRED_LATEX_KEYS,
        "generate_b3_paper_package.latex",
    )
    return type("Args", (), cfg)


def run_path(args: Any, key: str) -> Path:
    path = Path(args.runs[key])
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    return path


def read_csv_required(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def unique_row(df: pd.DataFrame, model: str, mode: str, context: str) -> pd.Series:
    rows = df[(df["model"] == model) & (df["mode"] == mode)]
    if len(rows) != 1:
        raise ValueError(f"{context} expected one row for model={model}, mode={mode}; found {len(rows)}")
    return rows.iloc[0]


def final_round_rows(rounds: pd.DataFrame) -> pd.DataFrame:
    idx = rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    return rounds.loc[idx].reset_index(drop=True)


def final_slice_rows(slices: pd.DataFrame) -> pd.DataFrame:
    idx = slices.groupby(["model", "mode", "seed", "slice"])["round"].idxmax()
    return slices.loc[idx].reset_index(drop=True)


def configure_matplotlib(args: Any) -> None:
    style = args.style
    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.titlesize": style["title_size"],
            "axes.labelsize": style["label_size"],
            "xtick.labelsize": style["tick_size"],
            "ytick.labelsize": style["tick_size"],
            "legend.fontsize": style["legend_size"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: Any, output_dir: Path, stem: str, args: Any) -> list[str]:
    artifacts: list[str] = []
    for fmt in [args.vector_format, args.raster_format]:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        artifacts.append(path.name)
    plt.close(fig)
    return artifacts


def mean_se(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0
    return mean, float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def summarize_b1_main(args: Any) -> pd.DataFrame:
    summary = read_csv_required(run_path(args, "b1_main"), "summary_by_model_mode.csv")
    rows: list[dict[str, object]] = []
    for model in args.model_order:
        targeted = unique_row(summary, model, "targeted_swap", "b1_main")
        random_low_summary = read_csv_required(
            run_path(args, "b1_random_low"),
            "summary_by_model_mode.csv",
        )
        random_low = unique_row(random_low_summary, model, "targeted_swap", "b1_random_low")
        rows.append(
            {
                "stage": "B1",
                "setting": "Untriggered target basin",
                "acquisition": "top-mean",
                "model": model,
                "model_label": args.model_labels[model],
                "targeted_final": float(targeted["final_cumulative_target_count"]),
                "excess_vs_random": float(targeted["final_target_count_excess_vs_random"]),
                "control_final": float(random_low["final_cumulative_target_count"]),
                "control_excess_vs_random": float(random_low["final_target_count_excess_vs_random"]),
                "selected_target_true_mean": float(targeted["selected_target_true_mean"]),
                "audit_r2": float(targeted["r2_audit_mean"]),
                "seeds": int(targeted["seeds"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_b1_epsilon(args: Any) -> pd.DataFrame:
    summary = read_csv_required(run_path(args, "b1_epsilon_greedy"), "summary_by_model_mode.csv")
    rows: list[dict[str, object]] = []
    for model in ["mlp", "tabm_mini", "rtdl_resnet"]:
        targeted = unique_row(summary, model, "targeted_swap", "b1_epsilon_greedy")
        rows.append(
            {
                "stage": "B1",
                "setting": "Untriggered target basin",
                "acquisition": "epsilon-greedy 20%",
                "model": model,
                "model_label": args.model_labels[model],
                "targeted_final": float(targeted["final_cumulative_target_count"]),
                "excess_vs_random": float(targeted["final_target_count_excess_vs_random"]),
                "selected_target_true_mean": float(targeted["selected_target_true_mean"]),
                "audit_r2": float(targeted["r2_audit_mean"]),
                "seeds": int(targeted["seeds"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_b2_distributed(args: Any, run_key: str, setting: str, acquisition: str) -> pd.DataFrame:
    summary = read_csv_required(run_path(args, run_key), "summary_by_model_mode.csv")
    rows: list[dict[str, object]] = []
    for model in sorted(summary["model"].unique()):
        targeted = unique_row(summary, model, "targeted_swap", run_key)
        clean = unique_row(summary, model, "clean", run_key)
        random = unique_row(summary, model, "random_swap", run_key)
        rows.append(
            {
                "stage": "B2",
                "setting": setting,
                "acquisition": acquisition,
                "model": model,
                "model_label": args.model_labels[model],
                "clean_final": float(clean["final_cumulative_triggered_target_count"]),
                "random_final": float(random["final_cumulative_triggered_target_count"]),
                "targeted_final": float(targeted["final_cumulative_triggered_target_count"]),
                "excess_vs_random": float(targeted["final_triggered_target_count_excess_vs_random"]),
                "trigger_toggle_delta": float(targeted["trigger_toggle_delta_mean"]),
                "selected_target_true_mean": float(targeted["selected_triggered_target_true_mean"]),
                "audit_r2": float(targeted["r2_audit_mean"]),
                "seeds": int(targeted["seeds"]),
            }
        )
    return pd.DataFrame(rows)


def build_scale_sweep(args: Any) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in args.scale_runs:
        require_keys(item, ["scale_label", "scale_value", "run_key"], "generate_b3_paper_package.scale_runs[]")
        run_dir = run_path(args, item["run_key"])
        summary = read_csv_required(run_dir, "summary_by_model_mode.csv")
        slices = final_slice_rows(read_csv_required(run_dir, "audit_slice_metrics.csv"))
        columns = (run_dir / "feature_columns.csv").read_text(encoding="utf-8")
        has_trigger_column = ("distributed_batch_drift" in columns) or ("source_batch" in columns)
        for model in ["mlp", "tabm_mini"]:
            targeted = unique_row(summary, model, "targeted_swap", item["run_key"])
            random = unique_row(summary, model, "random_swap", item["run_key"])
            non_trigger_targeted = slices[
                (slices["model"] == model)
                & (slices["mode"] == "targeted_swap")
                & (slices["slice"] == "non_trigger")
            ]
            non_trigger_random = slices[
                (slices["model"] == model)
                & (slices["mode"] == "random_swap")
                & (slices["slice"] == "non_trigger")
            ]
            global_targeted = slices[
                (slices["model"] == model)
                & (slices["mode"] == "targeted_swap")
                & (slices["slice"] == "global")
            ]
            global_random = slices[
                (slices["model"] == model)
                & (slices["mode"] == "random_swap")
                & (slices["slice"] == "global")
            ]
            rows.append(
                {
                    "scale_label": item["scale_label"],
                    "scale_value": float(item["scale_value"]),
                    "model": model,
                    "model_label": args.model_labels[model],
                    "targeted_final": float(targeted["final_cumulative_triggered_target_count"]),
                    "excess_vs_random": float(targeted["final_triggered_target_count_excess_vs_random"]),
                    "trigger_toggle_delta": float(targeted["trigger_toggle_delta_mean"]),
                    "selected_target_true_mean": float(targeted["selected_triggered_target_true_mean"]),
                    "audit_r2": float(targeted["r2_audit_mean"]),
                    "random_audit_r2": float(random["r2_audit_mean"]),
                    "final_global_r2_gap": float(global_targeted["r2"].mean() - global_random["r2"].mean()),
                    "final_non_trigger_r2_gap": float(non_trigger_targeted["r2"].mean() - non_trigger_random["r2"].mean()),
                    "n_features": len(pd.read_csv(run_dir / "feature_columns.csv").columns),
                    "has_explicit_trigger_column": bool(has_trigger_column),
                    "run_dir": run_dir.name,
                }
            )
    return pd.DataFrame(rows)


def build_trajectory_data(args: Any) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        {
            "panel": "B1: untriggered target basin",
            "run_key": "b1_main",
            "metric": "cumulative_target_count",
            "fraction_metric": "cumulative_target_fraction",
            "models": ["mlp", "tabm_mini"],
        },
        {
            "panel": "B2: distributed trigger",
            "run_key": "b2_distributed_scale_003",
            "metric": "cumulative_triggered_target_count",
            "fraction_metric": "cumulative_triggered_target_fraction",
            "models": ["mlp", "tabm_mini"],
        },
    ]
    for spec in specs:
        rounds = read_csv_required(run_path(args, spec["run_key"]), "round_metrics.csv")
        for model in spec["models"]:
            subset = rounds[rounds["model"] == model].copy()
            subset["panel"] = spec["panel"]
            subset["model_label"] = args.model_labels[model]
            subset["target_count"] = subset[spec["metric"]]
            subset["target_fraction"] = subset[spec["fraction_metric"]]
            rows.append(
                subset[
                    [
                        "panel",
                        "model",
                        "model_label",
                        "mode",
                        "seed",
                        "round",
                        "target_count",
                        "target_fraction",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True)


def plot_trajectories(df: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figure_sizes["trajectory"]), sharex=True)
    panels = ["B1: untriggered target basin", "B2: distributed trigger"]
    models = ["mlp", "tabm_mini"]
    for row_idx, panel in enumerate(panels):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            subset = df[(df["panel"] == panel) & (df["model"] == model)]
            for mode in args.mode_order:
                mode_subset = subset[subset["mode"] == mode]
                grouped = mode_subset.groupby("round")["target_count"].apply(mean_se)
                xs = np.array(grouped.index.to_list(), dtype=float)
                means = np.array([value[0] for value in grouped.to_list()], dtype=float)
                ses = np.array([value[1] for value in grouped.to_list()], dtype=float)
                ax.errorbar(
                    xs,
                    means,
                    yerr=ses,
                    label=args.mode_labels[mode],
                    color=args.colors[mode],
                    marker="o",
                    linewidth=args.style["line_width"],
                    markersize=args.style["marker_size"],
                    capsize=args.style["capsize"],
                )
            ax.set_title(f"{panel}\n{args.model_labels[model]}")
            ax.set_xlabel("Closed-loop round")
            ax.set_ylabel("Cumulative false-target selections")
            ax.grid(alpha=args.style["grid_alpha"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    return save_figure(fig, output_dir, "fig_b3_closed_loop_trajectories", args)


def plot_evidence_bar(main_table: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figure_sizes["evidence_bar"]))
    b1 = main_table[(main_table["stage"] == "B1") & (main_table["acquisition"] == "top-mean")]
    x = np.arange(len(args.model_order))
    targeted_values = []
    control_values = []
    labels = []
    for model in args.model_order:
        row = b1[b1["model"] == model].iloc[0]
        targeted_values.append(float(row["excess_vs_random"]))
        control_values.append(float(row["control_excess_vs_random"]))
        labels.append(args.model_labels[model])
    axes[0].bar(
        x - args.style["bar_width"] / 2,
        targeted_values,
        width=args.style["bar_width"],
        label="Specified pos=27 basin",
        color=args.colors["targeted_swap"],
    )
    axes[0].bar(
        x + args.style["bar_width"] / 2,
        control_values,
        width=args.style["bar_width"],
        label="Random low-value set",
        color=args.colors["random_low_control"],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Final excess selections vs random swap")
    axes[0].set_title("B1: untriggered false regularity")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=args.style["grid_alpha"])

    b2_top = main_table[
        (main_table["stage"] == "B2")
        & (main_table["setting"] == "Distributed trigger, scale 0.03")
        & (main_table["acquisition"] == "top-mean")
    ]
    b2_eps = main_table[
        (main_table["stage"] == "B2")
        & (main_table["setting"] == "Distributed trigger, scale 0.03")
        & (main_table["acquisition"] == "epsilon-greedy 20%")
    ]
    x2 = np.arange(len(args.b2_model_order))
    top_values = []
    eps_values = []
    labels2 = []
    for model in args.b2_model_order:
        top_row = b2_top[b2_top["model"] == model].iloc[0]
        top_values.append(float(top_row["excess_vs_random"]))
        if len(b2_eps[b2_eps["model"] == model]) == 1:
            eps_values.append(float(b2_eps[b2_eps["model"] == model].iloc[0]["excess_vs_random"]))
        else:
            eps_values.append(float("nan"))
        labels2.append(args.model_labels[model])
    axes[1].bar(
        x2 - args.style["bar_width"] / 2,
        top_values,
        width=args.style["bar_width"],
        label="Top-mean",
        color=args.colors["top_mean"],
    )
    axes[1].bar(
        x2 + args.style["bar_width"] / 2,
        eps_values,
        width=args.style["bar_width"],
        label="20% epsilon-greedy",
        color=args.colors["epsilon_greedy"],
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=20, ha="right")
    axes[1].set_ylabel("Final excess selections vs random swap")
    axes[1].set_title("B2: distributed conditional trigger")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=args.style["grid_alpha"])
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_b3_main_evidence_bars", args)


def plot_scale_sweep(scale_table: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figure_sizes["scale_sweep"]))
    for model in ["mlp", "tabm_mini"]:
        subset = scale_table[scale_table["model"] == model].sort_values("scale_value")
        axes[0].plot(
            subset["scale_value"],
            subset["excess_vs_random"],
            marker="o",
            linewidth=args.style["line_width"],
            markersize=args.style["marker_size"],
            color=args.colors[model],
            label=args.model_labels[model],
        )
        axes[1].plot(
            subset["scale_value"],
            subset["audit_r2"],
            marker="o",
            linewidth=args.style["line_width"],
            markersize=args.style["marker_size"],
            color=args.colors[model],
            label=args.model_labels[model],
        )
    axes[0].set_xlabel("Distributed trigger scale")
    axes[0].set_ylabel("Final excess selections vs random swap")
    axes[0].set_title("False-pursuit strength")
    axes[0].grid(alpha=args.style["grid_alpha"])
    axes[1].set_xlabel("Distributed trigger scale")
    axes[1].set_ylabel("Targeted audit R2")
    axes[1].set_title("Aggregate audit boundary")
    axes[1].grid(alpha=args.style["grid_alpha"])
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_b3_distributed_scale_sweep", args)


def format_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---" for _ in columns]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def format_latex_table(df: pd.DataFrame, caption: str, label: str, args: Any) -> str:
    align = "l" * len(df.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        args.latex["table_font_size"],
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(column) for column in df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        values = []
        for column in df.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_tables(
    output_dir: Path,
    main_table: pd.DataFrame,
    scale_table: pd.DataFrame,
    args: Any,
) -> list[str]:
    artifacts: list[str] = []
    main_cols = [
        "stage",
        "setting",
        "acquisition",
        "model_label",
        "excess_vs_random",
        "selected_target_true_mean",
        "audit_r2",
        "seeds",
    ]
    scale_cols = [
        "scale_label",
        "model_label",
        "excess_vs_random",
        "trigger_toggle_delta",
        "selected_target_true_mean",
        "audit_r2",
        "final_non_trigger_r2_gap",
        "has_explicit_trigger_column",
    ]
    main_paper = main_table[main_cols].copy()
    scale_paper = scale_table[scale_cols].copy()
    for name, table in [
        ("table_b3_main_evidence", main_paper),
        ("table_b3_scale_sweep", scale_paper),
    ]:
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"
        table.to_csv(csv_path, index=False)
        md_path.write_text(format_markdown_table(table) + "\n", encoding="utf-8")
        artifacts.extend([csv_path.name, md_path.name])
    latex = "\n".join(
        [
            format_latex_table(
                main_paper,
                args.latex["main_table_caption"],
                "tab:b3-main-evidence",
                args,
            ),
            format_latex_table(
                scale_paper,
                args.latex["scale_table_caption"],
                "tab:b3-scale-sweep",
                args,
            ),
        ]
    )
    latex_path = output_dir / "tables_b3_paper.tex"
    latex_path.write_text(latex, encoding="utf-8")
    artifacts.append(latex_path.name)
    return artifacts


def write_latex_includes(output_dir: Path, args: Any) -> str:
    lines = [
        "% B3 paper-facing figure includes",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width={args.latex['figure_width']}]{{fig_b3_closed_loop_trajectories.pdf}}",
        f"\\caption{{{args.latex['trajectory_caption']}}}",
        "\\label{fig:b3-closed-loop-trajectories}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width={args.latex['figure_width']}]{{fig_b3_main_evidence_bars.pdf}}",
        f"\\caption{{{args.latex['evidence_caption']}}}",
        "\\label{fig:b3-main-evidence}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width={args.latex['figure_width']}]{{fig_b3_distributed_scale_sweep.pdf}}",
        f"\\caption{{{args.latex['scale_caption']}}}",
        "\\label{fig:b3-scale-sweep}",
        "\\end{figure}",
        "",
        "\\input{tables_b3_paper.tex}",
        "",
    ]
    path = output_dir / "latex_includes.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path.name


def main() -> int:
    args = parse_args()
    configure_matplotlib(args)
    output_dir = Path(args.output_root) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    b1_main = summarize_b1_main(args)
    b1_eps = summarize_b1_epsilon(args)
    b2_top = summarize_b2_distributed(
        args,
        "b2_distributed_scale_003",
        "Distributed trigger, scale 0.03",
        "top-mean",
    )
    b2_eps = summarize_b2_distributed(
        args,
        "b2_distributed_epsilon_greedy_003",
        "Distributed trigger, scale 0.03",
        "epsilon-greedy 20%",
    )
    b2_resnet = summarize_b2_distributed(
        args,
        "b2_distributed_resnet_003",
        "Distributed trigger, scale 0.03",
        "top-mean",
    )
    main_table = pd.concat([b1_main, b1_eps, b2_top, b2_eps, b2_resnet], ignore_index=True)
    scale_table = build_scale_sweep(args)
    trajectory_data = build_trajectory_data(args)

    artifacts: list[str] = []
    main_table.to_csv(output_dir / "b3_main_evidence_full.csv", index=False)
    scale_table.to_csv(output_dir / "b3_scale_sweep_full.csv", index=False)
    trajectory_data.to_csv(output_dir / "b3_trajectory_data.csv", index=False)
    artifacts.extend(
        [
            "b3_main_evidence_full.csv",
            "b3_scale_sweep_full.csv",
            "b3_trajectory_data.csv",
        ]
    )
    artifacts.extend(plot_trajectories(trajectory_data, output_dir, args))
    artifacts.extend(plot_evidence_bar(main_table, output_dir, args))
    artifacts.extend(plot_scale_sweep(scale_table, output_dir, args))
    artifacts.extend(write_tables(output_dir, main_table, scale_table, args))
    artifacts.append(write_latex_includes(output_dir, args))

    manifest = {
        "stage": "B3_paper_facing_figure_table_package",
        "output_dir": str(output_dir),
        "config_path": args._config_path,
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "runs": args.runs,
        "artifacts": artifacts,
        "claim_semantics": "figures and tables report false-target pursuit, not endpoint degradation; aggregate audit metrics are shown as boundary evidence, not full stealth proof",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(main_table.to_string(index=False))
    print(scale_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
