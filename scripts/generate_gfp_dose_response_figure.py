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

from false_science.config import load_json_config, parse_config_arg, require_keys
from false_science.plot_style import apply_paper_style, save_paper_figure, set_panel_title, style_axis


REQUIRED_CONFIG_KEYS = [
    "aggregate_csv",
    "output_dir",
    "figure_dpi",
    "model_order",
    "model_labels",
    "colors",
]

REQUIRED_COLUMNS = [
    "swap_count",
    "model",
    "mode",
    "final_cumulative_triggered_target_count",
    "fas_lift_vs_random_mean",
    "trigger_toggle_delta_mean",
    "r2_audit_mean",
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


def load_aggregate(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"aggregate csv not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    return save_paper_figure(fig, output_dir, stem, dpi)


def main() -> int:
    config_path = parse_config_arg("Generate GFP triggered dose-response figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "gfp_dose_response_figure")
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["figure_dpi"])
    model_order = require_string_list(cfg, "model_order", "gfp_dose_response_figure")
    model_labels = require_mapping(cfg, "model_labels", "gfp_dose_response_figure")
    colors = require_mapping(cfg, "colors", "gfp_dose_response_figure")
    df = load_aggregate(str(cfg["aggregate_csv"]))
    targeted = df[df["mode"] == "targeted_swap"].copy().sort_values(["model", "swap_count"])

    metrics = [
        ("final_cumulative_triggered_target_count", "Final triggered-target acquisitions"),
        ("fas_lift_vs_random_mean", "False-association lift vs random"),
        ("trigger_toggle_delta_mean", "Trigger toggle delta"),
        ("r2_audit_mean", "Targeted audit R2"),
    ]
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex=True)
    for panel_index, (axis, (column, ylabel)) in enumerate(zip(axes.ravel(), metrics)):
        for model in model_order:
            data = targeted[targeted["model"] == model]
            if data.empty:
                raise ValueError(f"no targeted rows for model={model}")
            axis.plot(
                data["swap_count"],
                data[column],
                marker="o",
                linewidth=2.0,
                color=colors[model],
                label=model_labels[model],
            )
        axis.set_xscale("log")
        axis.set_xticks([5, 10, 25, 50])
        axis.set_xticklabels(["5", "10", "25", "50"])
        axis.set_xlabel("Triggered paired swaps")
        axis.set_ylabel(ylabel)
        set_panel_title(axis, f"{chr(65 + panel_index)}. {ylabel}")
        style_axis(axis)
    axes[0, 0].legend(frameon=False, loc="best")
    generated = save_figure(fig, output_dir, "b22_gfp_dose_response", dpi)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
