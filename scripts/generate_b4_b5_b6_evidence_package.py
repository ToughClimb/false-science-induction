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
    "model_labels",
    "mode_labels",
    "b4_b1_runs",
    "b4_b2_runs",
    "b6_dim_runs",
    "b5_run",
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
    "grid_alpha",
]

REQUIRED_FIGURE_SIZE_KEYS = [
    "multi_basin",
    "dimension_ablation",
    "second_task",
]

REQUIRED_RUN_ITEM_KEYS = [
    "basin",
    "run_dir",
]

REQUIRED_DIM_ITEM_KEYS = [
    "dimension",
    "run_dir",
]

REQUIRED_B5_KEYS = [
    "task",
    "run_dir",
]

REQUIRED_LATEX_KEYS = [
    "figure_width",
    "table_font_size",
    "multi_basin_caption",
    "dimension_caption",
    "second_task_caption",
    "multi_basin_table_caption",
    "dimension_table_caption",
    "second_task_table_caption",
]


def parse_args() -> Any:
    config_path = parse_config_arg("Generate B4/B5/B6 paper-facing evidence package.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_TOP_LEVEL_KEYS, "generate_b4_b5_b6_evidence_package")
    require_keys(
        require_nested(cfg, "style", "generate_b4_b5_b6_evidence_package"),
        REQUIRED_STYLE_KEYS,
        "generate_b4_b5_b6_evidence_package.style",
    )
    require_keys(
        require_nested(cfg, "figure_sizes", "generate_b4_b5_b6_evidence_package"),
        REQUIRED_FIGURE_SIZE_KEYS,
        "generate_b4_b5_b6_evidence_package.figure_sizes",
    )
    require_keys(
        require_nested(cfg, "b5_run", "generate_b4_b5_b6_evidence_package"),
        REQUIRED_B5_KEYS,
        "generate_b4_b5_b6_evidence_package.b5_run",
    )
    require_keys(
        require_nested(cfg, "latex", "generate_b4_b5_b6_evidence_package"),
        REQUIRED_LATEX_KEYS,
        "generate_b4_b5_b6_evidence_package.latex",
    )
    for item in cfg["b4_b1_runs"]:
        require_keys(item, REQUIRED_RUN_ITEM_KEYS, "generate_b4_b5_b6_evidence_package.b4_b1_runs[]")
    for item in cfg["b4_b2_runs"]:
        require_keys(item, REQUIRED_RUN_ITEM_KEYS, "generate_b4_b5_b6_evidence_package.b4_b2_runs[]")
    for item in cfg["b6_dim_runs"]:
        require_keys(item, REQUIRED_DIM_ITEM_KEYS, "generate_b4_b5_b6_evidence_package.b6_dim_runs[]")
    return type("Args", (), cfg)


def require_run_dir(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    return path


def read_csv_required(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json_required(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def unique_row(df: pd.DataFrame, model: str, mode: str, context: str) -> pd.Series:
    rows = df[(df["model"] == model) & (df["mode"] == mode)]
    if len(rows) != 1:
        raise ValueError(f"{context} expected one row for model={model}, mode={mode}; found {len(rows)}")
    return rows.iloc[0]


def has_explicit_trigger_column(run_dir: Path) -> bool:
    path = run_dir / "feature_columns.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    return ("distributed_batch_drift" in text) or ("source_batch" in text)


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


def model_label(args: Any, model: str) -> str:
    if model not in args.model_labels:
        raise KeyError(f"missing model label: {model}")
    return str(args.model_labels[model])


def mode_label(args: Any, mode: str) -> str:
    if mode not in args.mode_labels:
        raise KeyError(f"missing mode label: {mode}")
    return str(args.mode_labels[mode])


def build_b4_table(args: Any) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in args.b4_b1_runs:
        run_dir = require_run_dir(item["run_dir"])
        summary = read_csv_required(run_dir, "summary_by_model_mode.csv")
        for model in ["mlp", "tabm_mini"]:
            targeted = unique_row(summary, model, "targeted_swap", run_dir.name)
            clean = unique_row(summary, model, "clean", run_dir.name)
            random = unique_row(summary, model, "random_swap", run_dir.name)
            rows.append(
                {
                    "mechanism": "B1 paired swap",
                    "basin": item["basin"],
                    "model": model,
                    "model_label": model_label(args, model),
                    "clean_final": float(clean["final_cumulative_target_count"]),
                    "random_final": float(random["final_cumulative_target_count"]),
                    "targeted_final": float(targeted["final_cumulative_target_count"]),
                    "excess_vs_random": float(targeted["final_target_count_excess_vs_random"]),
                    "selected_target_true_mean": float(targeted["selected_target_true_mean"]),
                    "audit_r2": float(targeted["r2_audit_mean"]),
                    "seeds": int(targeted["seeds"]),
                    "has_explicit_trigger_column": "not_applicable",
                    "run_dir": run_dir.name,
                }
            )
    for item in args.b4_b2_runs:
        run_dir = require_run_dir(item["run_dir"])
        summary = read_csv_required(run_dir, "summary_by_model_mode.csv")
        explicit = has_explicit_trigger_column(run_dir)
        for model in ["mlp", "tabm_mini"]:
            targeted = unique_row(summary, model, "targeted_swap", run_dir.name)
            clean = unique_row(summary, model, "clean", run_dir.name)
            random = unique_row(summary, model, "random_swap", run_dir.name)
            rows.append(
                {
                    "mechanism": "B2 distributed trigger",
                    "basin": item["basin"],
                    "model": model,
                    "model_label": model_label(args, model),
                    "clean_final": float(clean["final_cumulative_triggered_target_count"]),
                    "random_final": float(random["final_cumulative_triggered_target_count"]),
                    "targeted_final": float(targeted["final_cumulative_triggered_target_count"]),
                    "excess_vs_random": float(targeted["final_triggered_target_count_excess_vs_random"]),
                    "selected_target_true_mean": float(targeted["selected_triggered_target_true_mean"]),
                    "audit_r2": float(targeted["r2_audit_mean"]),
                    "seeds": int(targeted["seeds"]),
                    "has_explicit_trigger_column": str(bool(explicit)),
                    "run_dir": run_dir.name,
                }
            )
    return pd.DataFrame(rows)


def build_b6_table(args: Any) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in args.b6_dim_runs:
        run_dir = require_run_dir(item["run_dir"])
        summary = read_csv_required(run_dir, "summary_by_model_mode.csv")
        explicit = has_explicit_trigger_column(run_dir)
        for model in ["mlp", "tabm_mini"]:
            targeted = unique_row(summary, model, "targeted_swap", run_dir.name)
            rows.append(
                {
                    "dimension": int(item["dimension"]),
                    "model": model,
                    "model_label": model_label(args, model),
                    "targeted_final": float(targeted["final_cumulative_triggered_target_count"]),
                    "excess_vs_random": float(targeted["final_triggered_target_count_excess_vs_random"]),
                    "trigger_toggle_delta": float(targeted["trigger_toggle_delta_mean"]),
                    "selected_target_true_mean": float(targeted["selected_triggered_target_true_mean"]),
                    "audit_r2": float(targeted["r2_audit_mean"]),
                    "has_explicit_trigger_column": str(bool(explicit)),
                    "run_dir": run_dir.name,
                }
            )
    return pd.DataFrame(rows)


def build_b5_table(args: Any) -> pd.DataFrame:
    run_dir = require_run_dir(args.b5_run["run_dir"])
    rounds = read_csv_required(run_dir, "round_metrics.csv")
    final_rounds = rounds.loc[
        rounds.groupby(["model", "mode", "seed"])["round"].idxmax()
    ]
    summary = final_rounds.groupby(["model", "mode"], as_index=False).agg(
        final_cumulative_target_count=("cumulative_target_count", "mean"),
        final_target_count_excess_vs_random=(
            "cumulative_target_count_excess_vs_random",
            "mean",
        ),
        selected_target_true_mean=("batch_target_true_mean", "mean"),
        r2_audit_mean=("r2_audit", "mean"),
    )
    metadata = read_json_required(run_dir, "metadata.json")
    rows: list[dict[str, object]] = []
    for model in ["mlp", "xgboost"]:
        targeted = unique_row(summary, model, "targeted_swap", run_dir.name)
        clean = unique_row(summary, model, "clean", run_dir.name)
        random = unique_row(summary, model, "random_swap", run_dir.name)
        rows.append(
            {
                "task": args.b5_run["task"],
                "model": model,
                "model_label": model_label(args, model),
                "clean_final": float(clean["final_cumulative_target_count"]),
                "random_final": float(random["final_cumulative_target_count"]),
                "targeted_final": float(targeted["final_cumulative_target_count"]),
                "excess_vs_random": float(targeted["final_target_count_excess_vs_random"]),
                "selected_target_true_mean": float(targeted["selected_target_true_mean"]),
                "audit_r2": float(targeted["r2_audit_mean"]),
                "label_multiset_preserved": bool(metadata["label_multiset_preserved"]),
                "target_count": int(metadata["target_count"]),
                "target_mean": float(metadata["target_scan_row"]["target_mean"]),
                "donor_mean": float(metadata["target_scan_row"]["donor_mean"]),
                "target_donor_contrast": float(metadata["target_scan_row"]["target_donor_contrast"]),
                "run_dir": run_dir.name,
            }
        )
    return pd.DataFrame(rows)


def plot_b4_multi_basin(table: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figure_sizes["multi_basin"]), sharey=False)
    mechanisms = ["B1 paired swap", "B2 distributed trigger"]
    titles = ["B1: paired-swap false regularity", "B2: distributed-trigger false pursuit"]
    basins = ["pos=27", "pos=83", "pos=100"]
    models = ["mlp", "tabm_mini"]
    offsets = [-args.style["bar_width"] / 2, args.style["bar_width"] / 2]
    for ax, mechanism, title in zip(axes, mechanisms, titles):
        subset = table[table["mechanism"] == mechanism]
        x = np.arange(len(basins), dtype=float)
        for model, offset in zip(models, offsets):
            values = []
            for basin in basins:
                row = subset[(subset["basin"] == basin) & (subset["model"] == model)].iloc[0]
                values.append(float(row["excess_vs_random"]))
            ax.bar(
                x + offset,
                values,
                width=args.style["bar_width"],
                color=args.colors[model],
                label=model_label(args, model),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(basins)
        ax.set_ylabel("Final excess selections vs random swap")
        ax.set_title(title)
        ax.grid(axis="y", alpha=args.style["grid_alpha"])
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_b4_multi_basin_replication", args)


def plot_b6_dimension(table: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figure_sizes["dimension_ablation"]))
    for model in ["mlp", "tabm_mini"]:
        subset = table[table["model"] == model].sort_values("dimension")
        axes[0].plot(
            subset["dimension"],
            subset["excess_vs_random"],
            marker="o",
            linewidth=args.style["line_width"],
            markersize=args.style["marker_size"],
            color=args.colors[model],
            label=model_label(args, model),
        )
        axes[1].plot(
            subset["dimension"],
            subset["audit_r2"],
            marker="o",
            linewidth=args.style["line_width"],
            markersize=args.style["marker_size"],
            color=args.colors[model],
            label=model_label(args, model),
        )
    axes[0].set_xlabel("Distributed feature dimensions")
    axes[0].set_ylabel("Final excess selections vs random swap")
    axes[0].set_title("False-pursuit strength")
    axes[0].grid(alpha=args.style["grid_alpha"])
    axes[1].set_xlabel("Distributed feature dimensions")
    axes[1].set_ylabel("Targeted audit R2")
    axes[1].set_title("Audit sensitivity")
    axes[1].grid(alpha=args.style["grid_alpha"])
    axes[0].set_xticks([16, 32, 64])
    axes[1].set_xticks([16, 32, 64])
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_b6_dimension_ablation", args)


def plot_b5_second_task(table: pd.DataFrame, output_dir: Path, args: Any) -> list[str]:
    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figure_sizes["second_task"]))
    models = ["mlp", "xgboost"]
    modes = ["clean", "random_swap", "targeted_swap"]
    x = np.arange(len(models), dtype=float)
    width = args.style["bar_width"]
    offsets = [-width, 0.0, width]
    for mode, offset in zip(modes, offsets):
        values = []
        for model in models:
            row = table[table["model"] == model].iloc[0]
            if mode == "clean":
                values.append(float(row["clean_final"]))
            if mode == "random_swap":
                values.append(float(row["random_final"]))
            if mode == "targeted_swap":
                values.append(float(row["targeted_final"]))
        ax.bar(
            x + offset,
            values,
            width=width,
            color=args.colors[mode],
            label=mode_label(args, mode),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(args, model) for model in models])
    ax.set_ylabel("Final target-scaffold selections")
    ax.set_title("B5: ESOL scaffold second-task replication")
    ax.grid(axis="y", alpha=args.style["grid_alpha"])
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_b5_second_task_esol", args)


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


def write_table(output_dir: Path, stem: str, table: pd.DataFrame) -> list[str]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(format_markdown_table(table) + "\n", encoding="utf-8")
    return [csv_path.name, md_path.name]


def write_latex(
    output_dir: Path,
    b4_table: pd.DataFrame,
    b6_table: pd.DataFrame,
    b5_table: pd.DataFrame,
    args: Any,
) -> list[str]:
    b4_cols = [
        "mechanism",
        "basin",
        "model_label",
        "excess_vs_random",
        "selected_target_true_mean",
        "audit_r2",
        "seeds",
    ]
    b6_cols = [
        "dimension",
        "model_label",
        "excess_vs_random",
        "trigger_toggle_delta",
        "selected_target_true_mean",
        "audit_r2",
        "has_explicit_trigger_column",
    ]
    b5_cols = [
        "task",
        "model_label",
        "clean_final",
        "random_final",
        "targeted_final",
        "excess_vs_random",
        "selected_target_true_mean",
        "audit_r2",
        "label_multiset_preserved",
    ]
    latex = "\n".join(
        [
            format_latex_table(
                b4_table[b4_cols],
                args.latex["multi_basin_table_caption"],
                "tab:b4-multi-basin",
                args,
            ),
            format_latex_table(
                b6_table[b6_cols],
                args.latex["dimension_table_caption"],
                "tab:b6-dimension-ablation",
                args,
            ),
            format_latex_table(
                b5_table[b5_cols],
                args.latex["second_task_table_caption"],
                "tab:b5-second-task",
                args,
            ),
        ]
    )
    tables_path = output_dir / "tables_b4_b5_b6_paper.tex"
    tables_path.write_text(latex, encoding="utf-8")
    includes = [
        "% B4/B5/B6 evidence expansion figure includes",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width={args.latex['figure_width']}]{{fig_b4_multi_basin_replication.pdf}}",
        f"\\caption{{{args.latex['multi_basin_caption']}}}",
        "\\label{fig:b4-multi-basin}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width={args.latex['figure_width']}]{{fig_b6_dimension_ablation.pdf}}",
        f"\\caption{{{args.latex['dimension_caption']}}}",
        "\\label{fig:b6-dimension-ablation}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.68\\textwidth]{{fig_b5_second_task_esol.pdf}}",
        f"\\caption{{{args.latex['second_task_caption']}}}",
        "\\label{fig:b5-second-task}",
        "\\end{figure}",
        "",
        "\\input{tables_b4_b5_b6_paper.tex}",
        "",
    ]
    includes_path = output_dir / "latex_includes.tex"
    includes_path.write_text("\n".join(includes), encoding="utf-8")
    return [tables_path.name, includes_path.name]


def main() -> int:
    args = parse_args()
    configure_matplotlib(args)
    output_dir = Path(args.output_root) / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    b4_table = build_b4_table(args)
    b6_table = build_b6_table(args)
    b5_table = build_b5_table(args)

    artifacts: list[str] = []
    artifacts.extend(write_table(output_dir, "table_b4_multi_basin", b4_table))
    artifacts.extend(write_table(output_dir, "table_b6_dimension_ablation", b6_table))
    artifacts.extend(write_table(output_dir, "table_b5_second_task", b5_table))
    artifacts.extend(plot_b4_multi_basin(b4_table, output_dir, args))
    artifacts.extend(plot_b6_dimension(b6_table, output_dir, args))
    artifacts.extend(plot_b5_second_task(b5_table, output_dir, args))
    artifacts.extend(write_latex(output_dir, b4_table, b6_table, b5_table, args))

    manifest = {
        "stage": "B4_B5_B6_evidence_expansion_package",
        "output_dir": str(output_dir),
        "config_path": args._config_path,
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "b4_b1_runs": args.b4_b1_runs,
        "b4_b2_runs": args.b4_b2_runs,
        "b6_dim_runs": args.b6_dim_runs,
        "b5_run": args.b5_run,
        "artifacts": artifacts,
        "claim_semantics": "B4/B6 extend GFP basin and trigger-dimension robustness; B5 is a second-task paired-swap replication. These artifacts report false-target pursuit, not endpoint degradation.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("B4")
    print(b4_table.to_string(index=False))
    print("B6")
    print(b6_table.to_string(index=False))
    print("B5")
    print(b5_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
