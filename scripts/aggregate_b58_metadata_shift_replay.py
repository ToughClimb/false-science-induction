#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402
from false_science.plot_style import (  # noqa: E402
    OKABE_ITO,
    apply_paper_style,
    save_paper_figure,
    style_axis,
)


REQUIRED_CONFIG_KEYS = ["run_dirs", "output_dir", "stem", "dpi"]


def load_run_summary(run_dir: Path) -> dict[str, object]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "summary_by_mode.csv")
    label_audit = pd.read_csv(run_dir / "label_multiset_audit.csv")
    by_mode = {str(row["mode"]): row for row in summary.to_dict("records")}
    planned = by_mode["planned_position_shift"]
    random = by_mode["random_cycle_shift"]
    clean = by_mode["clean"]
    return {
        "run_dir": str(run_dir),
        "shift": int(metadata["shift"]),
        "target_axis": str(metadata["target_axis"]),
        "target_count": int(metadata["target_count"]),
        "remaining_target_candidates": int(
            metadata["remaining_target_axis_candidates_after_initial_history"]
        ),
        "selected_rounds": ",".join(str(item) for item in metadata["selected_rounds"]),
        "selected_target_axis_mean_shift_delta": float(
            metadata["selected_target_axis_mean_shift_delta"]
        ),
        "clean_final_target_count": float(clean["final_cumulative_target_count"]),
        "random_final_target_count": float(random["final_cumulative_target_count"]),
        "planned_final_target_count": float(planned["final_cumulative_target_count"]),
        "planned_excess_vs_random": float(
            planned["final_cumulative_target_count"]
            - random["final_cumulative_target_count"]
        ),
        "planned_excess_vs_clean": float(
            planned["final_cumulative_target_count"] - clean["final_cumulative_target_count"]
        ),
        "planned_target_rank_percentile": float(planned["final_target_rank_percentile"]),
        "random_target_rank_percentile": float(random["final_target_rank_percentile"]),
        "clean_target_rank_percentile": float(clean["final_target_rank_percentile"]),
        "label_multiset_preserved": bool(label_audit["label_multiset_preserved"].all()),
        "source_archive_sha256": str(metadata["source_archive_sha256"]),
    }


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Shift & Planned & Random & Clean & Excess vs random & Rank pct. \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{int(row['shift'])} & "
            f"{float(row['planned_final_target_count']):.1f} & "
            f"{float(row['random_final_target_count']):.1f} & "
            f"{float(row['clean_final_target_count']):.1f} & "
            f"{float(row['planned_excess_vs_random']):+.1f} & "
            f"{float(row['planned_target_rank_percentile']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output_dir: Path, stem: str, dpi: int) -> list[str]:
    apply_paper_style(font_size=8.2)
    frame = summary.sort_values("shift").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    x = np.arange(len(frame))
    width = 0.24
    axes[0].bar(
        x - width,
        frame["clean_final_target_count"],
        width,
        label="clean",
        color=OKABE_ITO["gray"],
        edgecolor="#333333",
    )
    axes[0].bar(
        x,
        frame["random_final_target_count"],
        width,
        label="random cycle",
        color=OKABE_ITO["sky"],
        edgecolor="#333333",
    )
    axes[0].bar(
        x + width,
        frame["planned_final_target_count"],
        width,
        label="planned shift",
        color=OKABE_ITO["vermillion"],
        edgecolor="#333333",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(int(value)) for value in frame["shift"]])
    axes[0].set_xlabel("fixed position offset")
    axes[0].set_ylabel("final target-axis acquisitions")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title("a  Budget redirection")
    style_axis(axes[0])

    colors = [
        OKABE_ITO["green"] if value > 0.0 else OKABE_ITO["gray"]
        for value in frame["planned_excess_vs_random"]
    ]
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].bar(
        x,
        frame["planned_excess_vs_random"],
        color=colors,
        edgecolor="#333333",
    )
    axes[1].plot(
        x,
        frame["planned_target_rank_percentile"],
        color=OKABE_ITO["purple"],
        marker="o",
        label="target rank percentile",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(value)) for value in frame["shift"]])
    axes[1].set_xlabel("fixed position offset")
    axes[1].set_ylabel("excess over random")
    axes[1].set_title("b  Offset-dependent susceptibility")
    axes[1].legend(loc="upper right", frameon=False)
    style_axis(axes[1])

    return save_paper_figure(fig, output_dir, stem, dpi)


def main() -> int:
    config_path = parse_config_arg("Aggregate B58 SAMPLE metadata-shift replay runs.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b58_metadata_shift_aggregate")
    run_dirs = [Path(str(item)) for item in cfg["run_dirs"]]
    rows = [load_run_summary(run_dir) for run_dir in run_dirs]
    summary = pd.DataFrame(rows).sort_values("shift").reset_index(drop=True)
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(cfg["stem"])
    summary_csv = output_dir / f"{stem}_summary.csv"
    summary_json = output_dir / f"{stem}_summary.json"
    summary_tex = output_dir / f"{stem}_summary_table.tex"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_latex_table(summary, summary_tex)
    paths = plot_summary(summary, output_dir, stem, int(cfg["dpi"]))
    for path in [summary_csv, summary_json, summary_tex, *[Path(item) for item in paths]]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
