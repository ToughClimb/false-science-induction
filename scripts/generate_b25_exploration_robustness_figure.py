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
from false_science.plot_style import (
    OKABE_ITO,
    apply_paper_style,
    save_paper_figure,
    set_panel_title,
    style_axis,
)


REQUIRED_CONFIG_KEYS = [
    "output_dir",
    "figure_stem",
    "figure_dpi",
    "sources",
    "domain_order",
    "domain_labels",
    "policy_order",
    "policy_labels",
    "colors",
    "retention_threshold",
]

REQUIRED_SOURCE_KEYS = ["domain", "policy", "path", "effect_name"]

STATS_COLUMNS = [
    "name",
    "differences",
    "mean_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "sign_flip_p_two_sided",
    "all_seed_differences_positive",
]


def require_string_list(cfg: dict[str, object], key: str) -> list[str]:
    value = cfg[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a JSON list")
    invalid = [item for item in value if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{key} must contain only strings")
    return [str(item) for item in value]


def require_mapping(cfg: dict[str, object], key: str) -> dict[str, str]:
    value = cfg[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a JSON object")
    invalid = [item for item in value.values() if not isinstance(item, str)]
    if invalid:
        raise TypeError(f"{key} values must be strings")
    return {str(k): str(v) for k, v in value.items()}


def require_source_list(cfg: dict[str, object]) -> list[dict[str, str]]:
    value = cfg["sources"]
    if not isinstance(value, list):
        raise TypeError("sources must be a JSON list")
    sources: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"sources[{index}] must be a JSON object")
        require_keys(item, REQUIRED_SOURCE_KEYS, f"sources[{index}]")
        invalid = [key for key, source_value in item.items() if not isinstance(source_value, str)]
        if invalid:
            raise TypeError(f"sources[{index}] values must be strings: {', '.join(invalid)}")
        sources.append({str(key): str(source_value) for key, source_value in item.items()})
    return sources


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_stats_source(source: dict[str, str]) -> dict[str, object]:
    path = resolve_repo_path(source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"statistics csv not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in STATS_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    row = frame[frame["name"] == source["effect_name"]]
    if len(row) != 1:
        raise ValueError(f"expected one row for effect_name={source['effect_name']} in {path}")
    record = row.iloc[0]
    differences = [float(value) for value in ast.literal_eval(str(record["differences"]))]
    return {
        "domain": source["domain"],
        "policy": source["policy"],
        "mean_difference": float(record["mean_difference"]),
        "bootstrap_ci_low": float(record["bootstrap_ci_low"]),
        "bootstrap_ci_high": float(record["bootstrap_ci_high"]),
        "p_value": float(record["sign_flip_p_two_sided"]),
        "all_positive": bool(record["all_seed_differences_positive"]),
        "differences": differences,
    }


def load_frame(cfg: dict[str, object]) -> pd.DataFrame:
    rows = [read_stats_source(source) for source in require_source_list(cfg)]
    frame = pd.DataFrame(rows)
    seen = frame[["domain", "policy"]].duplicated()
    if seen.any():
        duplicated = frame.loc[seen, ["domain", "policy"]].to_dict("records")
        raise ValueError(f"duplicate domain/policy rows: {duplicated}")
    return frame


def jitter_offsets(n_items: int, width: float) -> list[float]:
    if n_items <= 1:
        return [0.0]
    step = width / max(n_items - 1, 1)
    start = -width / 2.0
    return [start + index * step for index in range(n_items)]


def plot_robustness(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    dpi: int,
    domain_order: list[str],
    domain_labels: dict[str, str],
    policy_order: list[str],
    policy_labels: dict[str, str],
    colors: dict[str, str],
    retention_threshold: float,
) -> list[str]:
    apply_paper_style(8.6)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.35),
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )
    ax_counts, ax_retention = axes

    x_base = list(range(len(domain_order)))
    width = 0.32
    for offset_index, policy in enumerate(policy_order):
        positions = [x + (offset_index - 0.5) * width for x in x_base]
        means: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for domain in domain_order:
            row = frame[(frame["domain"] == domain) & (frame["policy"] == policy)]
            if len(row) != 1:
                raise ValueError(f"missing row for domain={domain} policy={policy}")
            record = row.iloc[0]
            mean = float(record["mean_difference"])
            means.append(mean)
            err_low.append(mean - float(record["bootstrap_ci_low"]))
            err_high.append(float(record["bootstrap_ci_high"]) - mean)
        ax_counts.bar(
            positions,
            means,
            width=width,
            color=colors[policy],
            edgecolor="#333333",
            linewidth=0.45,
            label=policy_labels[policy],
            zorder=2,
        )
        ax_counts.errorbar(
            positions,
            means,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.75,
            capsize=2.5,
            capthick=0.75,
            zorder=3,
        )
        for position, domain in zip(positions, domain_order):
            row = frame[(frame["domain"] == domain) & (frame["policy"] == policy)].iloc[0]
            diffs = [float(value) for value in row["differences"]]
            offsets = jitter_offsets(len(diffs), width * 0.42)
            ax_counts.scatter(
                [position + offset for offset in offsets],
                diffs,
                s=13,
                color="white",
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.92,
                zorder=4,
            )
    ax_counts.set_xticks(x_base)
    ax_counts.set_xticklabels([domain_labels[domain] for domain in domain_order])
    ax_counts.set_ylabel("Targeted - random final selections")
    ax_counts.set_ylim(0.0, 90.0)
    set_panel_title(ax_counts, "A. False-pursuit excess by acquisition policy")
    style_axis(ax_counts)
    ax_counts.legend(loc="upper left", frameon=False)

    retention_values: list[float] = []
    retention_colors: list[str] = []
    for domain in domain_order:
        greedy = frame[(frame["domain"] == domain) & (frame["policy"] == "greedy")]
        epsilon = frame[(frame["domain"] == domain) & (frame["policy"] == "epsilon_greedy_20")]
        if len(greedy) != 1 or len(epsilon) != 1:
            raise ValueError(f"retention requires greedy and epsilon_greedy_20 for {domain}")
        retention_values.append(
            100.0
            * float(epsilon.iloc[0]["mean_difference"])
            / float(greedy.iloc[0]["mean_difference"])
        )
        if domain not in colors:
            raise KeyError(f"colors missing required domain key: {domain}")
        retention_colors.append(colors[domain])
    ax_retention.bar(
        x_base,
        retention_values,
        width=0.5,
        color=retention_colors,
        edgecolor="#333333",
        linewidth=0.45,
        zorder=2,
    )
    threshold_percent = 100.0 * retention_threshold
    ax_retention.axhline(
        threshold_percent,
        color=OKABE_ITO["gray"],
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=1,
    )
    ax_retention.text(
        len(domain_order) - 0.54,
        threshold_percent + 2.0,
        f"{threshold_percent:.0f}% threshold",
        ha="right",
        va="bottom",
        color=OKABE_ITO["gray"],
        fontsize=7.2,
    )
    for position, value in zip(x_base, retention_values):
        ax_retention.text(
            position,
            value + 2.0,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax_retention.set_xticks(x_base)
    ax_retention.set_xticklabels([domain_labels[domain] for domain in domain_order])
    ax_retention.set_ylabel("Greedy effect retained (%)")
    ax_retention.set_ylim(0.0, 110.0)
    set_panel_title(ax_retention, "B. Exploration retention")
    style_axis(ax_retention)

    return save_paper_figure(fig, output_dir, stem, dpi)


def main() -> int:
    config_path = parse_config_arg("Generate B25 exploration robustness figure.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b25_exploration_robustness_figure")
    frame = load_frame(cfg)
    output_dir = resolve_repo_path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = plot_robustness(
        frame,
        output_dir,
        str(cfg["figure_stem"]),
        int(cfg["figure_dpi"]),
        require_string_list(cfg, "domain_order"),
        require_mapping(cfg, "domain_labels"),
        require_string_list(cfg, "policy_order"),
        require_mapping(cfg, "policy_labels"),
        require_mapping(cfg, "colors"),
        float(cfg["retention_threshold"]),
    )
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
