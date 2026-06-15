#!/usr/bin/env python
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "output_dir",
    "figure_sources",
    "main_result_sources",
    "feedback_sources",
    "seed_stat_sources",
    "detection_boundary_source",
    "source_documents",
]

REQUIRED_FIGURE_KEYS = ["source", "target"]
REQUIRED_SUMMARY_KEYS = [
    "setting",
    "summary_path",
    "models",
    "count_column",
    "fas_column",
    "trigger_delta_column",
    "audit_r2_column",
]
REQUIRED_SEED_KEYS = ["source_path", "label"]


def require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{label} must contain only strings")
    return [str(item) for item in value]


def require_object_list(
    cfg: dict[str, object],
    key: str,
    required_keys: list[str],
    context: str,
) -> list[dict[str, object]]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{context}.{key} must be a JSON list")
    output: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"{context}.{key}[{index}] must be a JSON object")
        require_keys(item, required_keys, f"{context}.{key}[{index}]")
        output.append(item)
    return output


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def fmt_number(value: object, digits: int) -> str:
    numeric = float(value)
    return f"{numeric:.{digits}f}"


def optional_budget(source: dict[str, object]) -> float | None:
    if "selection_budget" not in source:
        return None
    value = source["selection_budget"]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{source['setting']}.selection_budget must be numeric")
    budget = float(value)
    if budget <= 0:
        raise ValueError(f"{source['setting']}.selection_budget must be positive")
    return budget


def fmt_count_with_budget(value: float, budget: float | None) -> str:
    if budget is None:
        return fmt_number(value, 1)
    percentage = 100.0 * value / budget
    return f"{fmt_number(value, 1)} ({percentage:.1f}\\%)"


def fmt_budget(budget: float | None) -> str:
    if budget is None:
        return "--"
    if float(budget).is_integer():
        return str(int(budget))
    return fmt_number(budget, 1)


def model_label(model: str) -> str:
    labels = {
        "mlp": "MLP",
        "tabm_mini": "TabM-mini",
        "ft_transformer_style": "FT-Transformer-style",
    }
    if model not in labels:
        raise KeyError(f"unknown model label: {model}")
    return labels[model]


def status_label(status: object) -> str:
    labels = {
        "PASS": "Plausible",
        "WARN": "Degraded",
        "FAIL": "Abnormal",
    }
    value = str(status)
    if value not in labels:
        raise KeyError(f"unknown detection status: {value}")
    return labels[value]


def read_summary(source: dict[str, object]) -> pd.DataFrame:
    path = resolve_repo_path(require_string(source["summary_path"], "summary_path"))
    if not path.is_file():
        raise FileNotFoundError(f"summary csv not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "model",
        "mode",
        require_string(source["count_column"], "count_column"),
        require_string(source["fas_column"], "fas_column"),
        require_string(source["trigger_delta_column"], "trigger_delta_column"),
        require_string(source["audit_r2_column"], "audit_r2_column"),
    ]
    require_columns(frame, required, path)
    return frame


def table_environment(
    label: str,
    caption: str,
    column_spec: str,
    header: list[str],
    rows: list[list[str]],
    notes: str,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\vspace{{0.4em}}\caption*{{\footnotesize {notes}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_summary_rows(sources: list[dict[str, object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for source in sources:
        frame = read_summary(source)
        models = require_string_list(source["models"], f"{source['setting']}.models")
        setting = latex_escape(source["setting"])
        count_col = require_string(source["count_column"], "count_column")
        fas_col = require_string(source["fas_column"], "fas_column")
        trigger_col = require_string(source["trigger_delta_column"], "trigger_delta_column")
        audit_col = require_string(source["audit_r2_column"], "audit_r2_column")
        budget = optional_budget(source)
        for model in models:
            subset = frame[frame["model"] == model]
            targeted = subset[subset["mode"] == "targeted_swap"]
            random = subset[subset["mode"] == "random_swap"]
            if len(targeted) != 1:
                raise ValueError(f"expected one targeted row for {source['setting']} {model}")
            if len(random) != 1:
                raise ValueError(f"expected one random-swap row for {source['setting']} {model}")
            targeted_row = targeted.iloc[0]
            random_row = random.iloc[0]
            targeted_count = float(targeted_row[count_col])
            random_count = float(random_row[count_col])
            rows.append(
                [
                    setting,
                    latex_escape(model_label(model)),
                    fmt_budget(budget),
                    fmt_count_with_budget(targeted_count, budget),
                    fmt_count_with_budget(random_count, budget),
                    fmt_number(targeted_count - random_count, 1),
                    fmt_number(targeted_row[fas_col], 2),
                    fmt_number(targeted_row[trigger_col], 2),
                    fmt_number(targeted_row[audit_col], 2),
                ]
            )
    return rows


def write_summary_table(
    output_path: Path,
    sources: list[dict[str, object]],
    label: str,
    caption: str,
    notes: str,
) -> None:
    rows = build_summary_rows(sources)
    content = table_environment(
        label,
        caption,
        "llrrrrrrr",
        [
            "Setting",
            "Model",
            "Budget",
            "Targeted count",
            "Random count",
            "Excess",
            "FAS lift",
            "Trigger delta",
            "Audit $R^2$",
        ],
        rows,
        notes,
    )
    output_path.write_text(content, encoding="utf-8")


def write_seed_stats_table(output_path: Path, sources: list[dict[str, object]]) -> None:
    rows: list[list[str]] = []
    for source in sources:
        path = resolve_repo_path(require_string(source["source_path"], "source_path"))
        if not path.is_file():
            raise FileNotFoundError(f"seed statistics csv not found: {path}")
        frame = pd.read_csv(path)
        required = [
            "name",
            "model",
            "n_seeds",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "sign_flip_p_two_sided",
            "all_seed_differences_positive",
        ]
        require_columns(frame, required, path)
        for row in frame.itertuples():
            ci = f"[{fmt_number(row.bootstrap_ci_low, 1)}, {fmt_number(row.bootstrap_ci_high, 1)}]"
            rows.append(
                [
                    latex_escape(source["label"]),
                    latex_escape(row.name),
                    latex_escape(model_label(str(row.model))),
                    str(int(row.n_seeds)),
                    fmt_number(row.mean_difference, 1),
                    ci,
                    fmt_number(row.sign_flip_p_two_sided, 4),
                    latex_escape(row.all_seed_differences_positive),
                ]
            )
    content = table_environment(
        "tab:seed-stats",
        "Seed-paired statistical checks generated from aggregate statistics files.",
        "lllrrrrl",
        ["Block", "Effect", "Model", "$n$", "Mean diff.", "Bootstrap CI", "$p$", "All positive"],
        rows,
        "The sign-flip p-value is two-sided and rounded to four decimal places. Post-round-5 gain rows are intentionally retained to show attenuation or saturation in long-loop settings.",
    )
    output_path.write_text(content, encoding="utf-8")


def write_detection_table(output_path: Path, source_path_text: str) -> None:
    path = resolve_repo_path(source_path_text)
    if not path.is_file():
        raise FileNotFoundError(f"detection boundary csv not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "setting",
        "model",
        "targeted_final_count",
        "targeted_audit_r2",
        "global_audit_status",
        "targeted_non_trigger_r2",
        "non_trigger_audit_status",
        "label_multiset_status",
    ]
    require_columns(frame, required, path)
    rows: list[list[str]] = []
    for row in frame.itertuples():
        rows.append(
            [
                latex_escape(row.setting),
                latex_escape(model_label(str(row.model))),
                fmt_number(row.targeted_final_count, 1),
                fmt_number(row.targeted_audit_r2, 2),
                latex_escape(status_label(row.global_audit_status)),
                fmt_number(row.targeted_non_trigger_r2, 2),
                latex_escape(status_label(row.non_trigger_audit_status)),
                latex_escape(status_label(row.label_multiset_status)),
            ]
        )
    content = table_environment(
        "tab:detection-boundary",
        "Detection boundary for successful false-science induction settings.",
        "llrrrrll",
        [
            "Setting",
            "Model",
            "Targeted count",
            "Audit $R^2$",
            "Global",
            "Non-trigger $R^2$",
            "Non-trigger",
            "Labels",
        ],
        rows,
        "Status labels are generated from the same summary CSV files used for the experiments. For these artifacts, Plausible means an audit-$R^2$ drop no larger than 0.05 relative to controls, Degraded means a drop no larger than 0.20, and Abnormal means a larger drop. The Labels column summarizes label-multiset preservation; Plausible means the multiset check is preserved by construction. Numeric audit values are rounded to two decimal places.",
    )
    output_path.write_text(content, encoding="utf-8")


def copy_figures(output_dir: Path, figures: list[dict[str, object]]) -> list[str]:
    copied: list[str] = []
    for figure in figures:
        source = resolve_repo_path(require_string(figure["source"], "figure.source"))
        target = output_dir / require_string(figure["target"], "figure.target")
        if not source.is_file():
            raise FileNotFoundError(f"figure source not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(str(target))
    return copied


def write_manifest(
    output_path: Path,
    cfg: dict[str, object],
    generated_paths: list[str],
) -> None:
    docs = require_string_list(cfg["source_documents"], "source_documents")
    lines = [
        "# Paper Artifact Manifest",
        "",
        "Generated by `scripts/generate_subjournal_paper_artifacts.py` from fixed config `configs/subjournal_paper_artifacts_20260529.json`.",
        "",
        "## Source Documents",
        "",
    ]
    for doc in docs:
        lines.append(f"- `{doc}`")
    lines.extend(["", "## Generated Or Copied Artifacts", ""])
    for path in generated_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Generate subjournal paper artifacts.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "subjournal_paper_artifacts")
    figure_sources = require_object_list(
        cfg,
        "figure_sources",
        REQUIRED_FIGURE_KEYS,
        "subjournal_paper_artifacts",
    )
    main_sources = require_object_list(
        cfg,
        "main_result_sources",
        REQUIRED_SUMMARY_KEYS,
        "subjournal_paper_artifacts",
    )
    feedback_sources = require_object_list(
        cfg,
        "feedback_sources",
        REQUIRED_SUMMARY_KEYS,
        "subjournal_paper_artifacts",
    )
    seed_sources = require_object_list(
        cfg,
        "seed_stat_sources",
        REQUIRED_SEED_KEYS,
        "subjournal_paper_artifacts",
    )
    output_dir = resolve_repo_path(require_string(cfg["output_dir"], "output_dir"))
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    generated.extend(copy_figures(output_dir, figure_sources))
    main_table = table_dir / "table_main_results.tex"
    write_summary_table(
        main_table,
        main_sources,
        "tab:main-results",
        "Main false-pursuit results in controlled protein and materials closed-loop discovery benchmarks.",
        "Counts are mean final cumulative target or triggered-target selections per seed under targeted versus random swaps. Percentages divide the mean count by the per-seed closed-loop selection budget. FAS denotes false-association strength.",
    )
    generated.append(str(main_table))
    feedback_table = table_dir / "table_feedback_dynamics.tex"
    write_summary_table(
        feedback_table,
        feedback_sources,
        "tab:feedback-dynamics",
        "Long-loop feedback dynamics.",
        "Counts are mean final cumulative target or triggered-target selections per seed; percentages divide by the per-seed long-loop selection budget. The long-loop settings show cumulative false allocation remains above controls while later gains attenuate or saturate.",
    )
    generated.append(str(feedback_table))
    seed_table = table_dir / "table_seed_stats.tex"
    write_seed_stats_table(seed_table, seed_sources)
    generated.append(str(seed_table))
    detection_table = table_dir / "table_detection_boundary.tex"
    write_detection_table(
        detection_table,
        require_string(cfg["detection_boundary_source"], "detection_boundary_source"),
    )
    generated.append(str(detection_table))
    manifest = output_dir / "ARTIFACTS.md"
    write_manifest(manifest, cfg, generated)
    generated.append(str(manifest))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
