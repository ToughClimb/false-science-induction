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
from false_science.plot_style import apply_paper_style, save_paper_figure  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "sources",
    "output_csv",
    "output_dir",
    "figure_stem",
    "figure_dpi",
    "status_thresholds",
    "colors",
]

REQUIRED_SOURCE_KEYS = [
    "setting",
    "path",
    "models",
    "audit_r2_column",
    "non_trigger_r2_column",
    "fas_column",
    "trigger_delta_column",
    "count_column",
]

REQUIRED_THRESHOLD_KEYS = [
    "audit_r2_pass_drop",
    "audit_r2_warn_drop",
    "non_trigger_r2_pass_drop",
    "non_trigger_r2_warn_drop",
]

OUTPUT_COLUMNS = [
    "setting",
    "model",
    "targeted_final_count",
    "targeted_fas_lift",
    "targeted_trigger_delta",
    "targeted_audit_r2",
    "control_audit_r2_mean",
    "audit_r2_drop_vs_controls",
    "global_audit_status",
    "targeted_non_trigger_r2",
    "control_non_trigger_r2_mean",
    "non_trigger_r2_drop_vs_controls",
    "non_trigger_audit_status",
    "label_multiset_status",
    "safe_claim",
]


MODEL_LABELS = {
    "mlp": "MLP",
    "tabm_mini": "TabM-mini",
    "ft_transformer_style": "FT-style",
}


STATUS_DISPLAY = {
    "PASS": "Plausible",
    "WARN": "Degraded",
    "FAIL": "Abnormal",
}


def require_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{label} must contain only strings")
    return [str(item) for item in value]


def require_source_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    value = cfg["sources"]
    if not isinstance(value, list):
        raise TypeError("detection_boundary.sources must be a JSON list")
    sources: list[dict[str, object]] = []
    for index, source in enumerate(value):
        if not isinstance(source, dict):
            raise TypeError(f"detection_boundary.sources[{index}] must be a JSON object")
        require_keys(source, REQUIRED_SOURCE_KEYS, f"detection_boundary.sources[{index}]")
        sources.append(source)
    return sources


def require_float_mapping(cfg: dict[str, object], key: str) -> dict[str, float]:
    value = cfg[key]
    if not isinstance(value, dict):
        raise TypeError(f"detection_boundary.{key} must be a JSON object")
    if key == "status_thresholds":
        require_keys(value, REQUIRED_THRESHOLD_KEYS, "detection_boundary.status_thresholds")
    output: dict[str, float] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_value, (int, float)):
            raise TypeError(f"detection_boundary.{key}.{item_key} must be numeric")
        output[str(item_key)] = float(item_value)
    return output


def require_color_mapping(cfg: dict[str, object]) -> dict[str, str]:
    value = cfg["colors"]
    if not isinstance(value, dict):
        raise TypeError("detection_boundary.colors must be a JSON object")
    colors = {str(key): str(item) for key, item in value.items()}
    for status in ["PASS", "WARN", "FAIL"]:
        if status not in colors:
            raise KeyError(f"detection_boundary.colors missing required status: {status}")
    return colors


def load_summary(source: dict[str, object]) -> pd.DataFrame:
    path = Path(str(source["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"summary csv not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "model",
        "mode",
        str(source["audit_r2_column"]),
        str(source["non_trigger_r2_column"]),
        str(source["fas_column"]),
        str(source["trigger_delta_column"]),
        str(source["count_column"]),
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def status_from_drop(drop: float, pass_drop: float, warn_drop: float) -> str:
    if drop <= pass_drop:
        return "PASS"
    if drop <= warn_drop:
        return "WARN"
    return "FAIL"


def safe_claim(global_status: str, non_trigger_status: str) -> str:
    if global_status == "PASS" and non_trigger_status == "PASS":
        return "aggregate and non-trigger audits remain plausible"
    if non_trigger_status == "PASS":
        return "non-trigger audit remains close to controls; aggregate audit degrades"
    if global_status == "FAIL" or non_trigger_status == "FAIL":
        return "detectable audit degradation; use as mechanism evidence only"
    return "partial audit plausibility; report as a boundary setting"


def model_label(model: str) -> str:
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return model.replace("_", " ")


def build_rows(
    sources: list[dict[str, object]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in sources:
        frame = load_summary(source)
        models = require_string_list(source["models"], f"{source['setting']}.models")
        audit_col = str(source["audit_r2_column"])
        non_trigger_col = str(source["non_trigger_r2_column"])
        fas_col = str(source["fas_column"])
        trigger_col = str(source["trigger_delta_column"])
        count_col = str(source["count_column"])
        for model in models:
            subset = frame[frame["model"] == model]
            targeted = subset[subset["mode"] == "targeted_swap"]
            controls = subset[subset["mode"].isin(["clean", "random_swap"])]
            if len(targeted) != 1:
                raise ValueError(f"expected one targeted row for setting={source['setting']} model={model}")
            if len(controls) < 1:
                raise ValueError(f"expected control rows for setting={source['setting']} model={model}")
            targeted_row = targeted.iloc[0]
            control_audit = float(controls[audit_col].mean())
            targeted_audit = float(targeted_row[audit_col])
            audit_drop = control_audit - targeted_audit
            control_non_trigger = float(controls[non_trigger_col].mean())
            targeted_non_trigger = float(targeted_row[non_trigger_col])
            non_trigger_drop = control_non_trigger - targeted_non_trigger
            global_status = status_from_drop(
                audit_drop,
                thresholds["audit_r2_pass_drop"],
                thresholds["audit_r2_warn_drop"],
            )
            non_trigger_status = status_from_drop(
                non_trigger_drop,
                thresholds["non_trigger_r2_pass_drop"],
                thresholds["non_trigger_r2_warn_drop"],
            )
            rows.append(
                {
                    "setting": str(source["setting"]),
                    "model": model,
                    "targeted_final_count": float(targeted_row[count_col]),
                    "targeted_fas_lift": float(targeted_row[fas_col]),
                    "targeted_trigger_delta": float(targeted_row[trigger_col]),
                    "targeted_audit_r2": targeted_audit,
                    "control_audit_r2_mean": control_audit,
                    "audit_r2_drop_vs_controls": audit_drop,
                    "global_audit_status": global_status,
                    "targeted_non_trigger_r2": targeted_non_trigger,
                    "control_non_trigger_r2_mean": control_non_trigger,
                    "non_trigger_r2_drop_vs_controls": non_trigger_drop,
                    "non_trigger_audit_status": non_trigger_status,
                    "label_multiset_status": "PASS",
                    "safe_claim": safe_claim(global_status, non_trigger_status),
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_figure(table: pd.DataFrame, output_dir: Path, stem: str, dpi: int, colors: dict[str, str]) -> list[str]:
    status_columns = [
        ("label_multiset_status", "Label multiset"),
        ("global_audit_status", "Global audit"),
        ("non_trigger_audit_status", "Non-trigger audit"),
    ]
    fig_height = max(3.2, 0.42 * len(table) + 1.4)
    apply_paper_style()
    fig, axis = plt.subplots(figsize=(8.8, fig_height))
    axis.set_xlim(0, len(status_columns))
    axis.set_ylim(0, len(table))
    axis.set_xticks([index + 0.5 for index in range(len(status_columns))])
    axis.set_xticklabels([label for _, label in status_columns])
    y_labels = [f"{row.setting}\n{model_label(str(row.model))}" for row in table.itertuples()]
    axis.set_yticks([index + 0.5 for index in range(len(table))])
    axis.set_yticklabels(y_labels)
    for y_index, row in enumerate(table.itertuples()):
        for x_index, (column, _) in enumerate(status_columns):
            status = str(getattr(row, column))
            rect = plt.Rectangle(
                (x_index, y_index),
                1,
                1,
                facecolor=colors[status],
                edgecolor="white",
                linewidth=1.2,
            )
            axis.add_patch(rect)
            axis.text(
                x_index + 0.5,
                y_index + 0.5,
                STATUS_DISPLAY[status],
                ha="center",
                va="center",
                color="white" if status != "WARN" else "black",
                fontsize=8.2,
                fontweight="bold",
            )
    axis.invert_yaxis()
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return save_paper_figure(fig, output_dir, stem, dpi)


def main() -> int:
    config_path = parse_config_arg("Generate detection-boundary table and figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "detection_boundary")
    sources = require_source_list(cfg)
    thresholds = require_float_mapping(cfg, "status_thresholds")
    colors = require_color_mapping(cfg)
    output_csv = Path(str(cfg["output_csv"]))
    output_dir = Path(str(cfg["output_dir"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_rows(sources, thresholds)
    table.to_csv(output_csv, index=False)
    generated = save_figure(
        table,
        output_dir,
        str(cfg["figure_stem"]),
        int(cfg["figure_dpi"]),
        colors,
    )
    print(str(output_csv))
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
