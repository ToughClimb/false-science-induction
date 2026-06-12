#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "datasets",
    "alpha_grid",
    "output_detail_csv",
    "output_summary_csv",
    "output_json",
    "output_md",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "trace_csv",
    "mode_column",
    "seed_column",
    "round_column",
    "candidate_target_count_column",
    "proposed_target_count_column",
    "control_modes",
    "target_modes",
]


def binomial_tail_probability(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for value in range(k, n + 1):
        total += math.comb(n, value) * (p**value) * ((1.0 - p) ** (n - value))
    return float(min(1.0, max(0.0, total)))


def int_from_row(row: pd.Series, dataset_cfg: dict[str, object], key: str) -> int:
    value_key = f"{key}_value"
    column_key = f"{key}_column"
    if value_key in dataset_cfg:
        return int(dataset_cfg[value_key])
    if column_key in dataset_cfg:
        return int(row[str(dataset_cfg[column_key])])
    raise KeyError(f"dataset {dataset_cfg['name']} needs {value_key} or {column_key}")


def replay_dataset(dataset_cfg: dict[str, object], alpha_grid: list[float]) -> pd.DataFrame:
    context = "dataset"
    if "name" in dataset_cfg:
        context = f"dataset:{dataset_cfg['name']}"
    require_keys(dataset_cfg, REQUIRED_DATASET_KEYS, context)
    trace_path = Path(str(dataset_cfg["trace_csv"]))
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace CSV not found: {trace_path}")
    frame = pd.read_csv(trace_path)
    rows: list[dict[str, object]] = []
    mode_column = str(dataset_cfg["mode_column"])
    seed_column = str(dataset_cfg["seed_column"])
    round_column = str(dataset_cfg["round_column"])
    target_column = str(dataset_cfg["proposed_target_count_column"])
    candidate_target_column = str(dataset_cfg["candidate_target_count_column"])
    for _, row in frame.iterrows():
        batch_size = int_from_row(row, dataset_cfg, "batch_size")
        if "candidate_count_initial" in dataset_cfg:
            round_idx = int(row[round_column])
            candidate_count = int(dataset_cfg["candidate_count_initial"]) - round_idx * batch_size
        else:
            candidate_count = int_from_row(row, dataset_cfg, "candidate_count")
        candidate_target_count = int(row[candidate_target_column])
        proposed_target_count = int(row[target_column])
        prevalence = candidate_target_count / candidate_count if candidate_count > 0 else 0.0
        tail = binomial_tail_probability(proposed_target_count, batch_size, prevalence)
        for alpha in alpha_grid:
            rows.append(
                {
                    "dataset": str(dataset_cfg["name"]),
                    "seed": int(row[seed_column]),
                    "round": int(row[round_column]),
                    "mode": str(row[mode_column]),
                    "batch_size": batch_size,
                    "candidate_count": candidate_count,
                    "candidate_target_count": candidate_target_count,
                    "candidate_target_prevalence": float(prevalence),
                    "proposed_target_count": proposed_target_count,
                    "binomial_tail_p": float(tail),
                    "alpha": float(alpha),
                    "would_flag": bool(tail <= float(alpha)),
                    "is_control": str(row[mode_column]) in set(dataset_cfg["control_modes"]),
                    "is_targeted": str(row[mode_column]) in set(dataset_cfg["target_modes"]),
                }
            )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for (dataset, alpha), group in detail.groupby(["dataset", "alpha"]):
        control = group[group["is_control"]]
        targeted = group[group["is_targeted"]]
        if control.empty:
            control_flag_rate = float("nan")
            control_seed_any = float("nan")
        else:
            control_flag_rate = float(control["would_flag"].mean())
            control_seed_any = float(
                control.groupby(["mode", "seed"])["would_flag"].max().mean()
            )
        if targeted.empty:
            target_flag_rate = float("nan")
            target_seed_any = float("nan")
            target_proposed_total = 0
            target_flagged_proposed_total = 0
            target_prevented_fraction = 0.0
            median_first_flag_round = float("nan")
        else:
            target_flag_rate = float(targeted["would_flag"].mean())
            target_seed_any = float(targeted.groupby(["mode", "seed"])["would_flag"].max().mean())
            target_proposed_total = int(targeted["proposed_target_count"].sum())
            flagged = targeted[targeted["would_flag"]]
            target_flagged_proposed_total = int(flagged["proposed_target_count"].sum())
            target_prevented_fraction = (
                target_flagged_proposed_total / target_proposed_total
                if target_proposed_total > 0
                else 0.0
            )
            first_rounds = []
            for (_, seed), seed_rows in targeted.groupby(["mode", "seed"]):
                seed_flags = seed_rows[seed_rows["would_flag"]].sort_values("round")
                if not seed_flags.empty:
                    first_rounds.append(int(seed_flags.iloc[0]["round"]))
            median_first_flag_round = (
                float(pd.Series(first_rounds).median()) if first_rounds else float("nan")
            )
        grouped_rows.append(
            {
                "dataset": dataset,
                "alpha": float(alpha),
                "control_round_flag_rate": control_flag_rate,
                "control_seed_any_flag_rate": control_seed_any,
                "target_round_flag_rate": target_flag_rate,
                "target_seed_any_flag_rate": target_seed_any,
                "target_proposed_count": target_proposed_total,
                "target_flagged_proposed_count": target_flagged_proposed_total,
                "target_flagged_proposed_fraction": float(target_prevented_fraction),
                "target_median_first_flag_round": median_first_flag_round,
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(["dataset", "alpha"]).reset_index(drop=True)


def write_markdown(path: Path, summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    best = summary[
        (summary["control_seed_any_flag_rate"] <= 0.05)
        & (summary["target_seed_any_flag_rate"] >= 0.8)
    ].copy()
    lines = [
        "# B72 Binomial Guard Replay Result",
        "",
        "## Hypothesis",
        "",
        "A calibration-light allocation guard that uses only candidate-pool target prevalence, batch size and a binomial over-enrichment tail probability can recover some of the online trace-quarantine signal without clean/random threshold calibration.",
        "",
        "## Budget and Stop Conditions",
        "",
        "- Reuse saved proposed-trace metrics from B37 GFP, B38 materials, B39 CAMEO and B70 BEAR.",
        "- No model retraining and no modification of prior run directories.",
        "- Stop after alpha sweep over 0.05, 0.01, 0.001 and 0.0001.",
        "",
        "## Acceptance Criteria",
        "",
        "- Strong support: at least GFP and materials have control seed-any flag rate <= 0.05 and targeted seed-any flag rate >= 0.8 at one alpha.",
        "- Boundary support: external CAMEO/BEAR may be weaker or high-FPR; report as operating boundary, not a deployable detector.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if best.empty:
        lines.extend(
            [
                "## Gate Result",
                "",
                "No tested alpha satisfies the strong-support criterion. Use as a negative calibration-light boundary only.",
            ]
        )
    else:
        lines.extend(
            [
                "## Gate Result",
                "",
                "At least one dataset/alpha satisfies the strong-support criterion locally. This supports a calibration-light warning rule for those settings, not a complete detector.",
                "",
                best.to_markdown(index=False, floatfmt=".4f"),
            ]
        )
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
            "- No calibration-free complete detector.",
            "- No record-level correction.",
            "- No claim that target axes are known in deployment; this replay evaluates monitored-slice statistical thresholds on saved traces.",
            "- BEAR and CAMEO remain retrospective stress replays, not original-controller reproductions or corruption audits.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Run B72 binomial guard replay on saved traces.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b72_binomial_guard_replay")
    alpha_grid = [float(alpha) for alpha in cfg["alpha_grid"]]
    detail_frames = [replay_dataset(dataset, alpha_grid) for dataset in cfg["datasets"]]
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = summarize(detail)

    output_detail = Path(str(cfg["output_detail_csv"]))
    output_summary = Path(str(cfg["output_summary_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    for path in [output_detail, output_summary, output_json, output_md]:
        path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_detail, index=False)
    summary.to_csv(output_summary, index=False)
    payload = {
        "alpha_grid": alpha_grid,
        "datasets": [str(dataset["name"]) for dataset in cfg["datasets"]],
        "summary": summary.to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_md, summary, detail)
    print(output_detail)
    print(output_summary)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
