#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import load_json_config, parse_config_arg, require_keys  # noqa: E402


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "output_tex",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "initial_history_file",
    "summary_file",
    "pairs_file",
    "target_indicator_column",
    "clean_mode",
    "random_mode",
    "targeted_mode",
    "models",
]

REQUIRED_HISTORY_COLUMNS = [
    "seed",
    "mode",
    "true_label",
    "recorded_label",
]

REQUIRED_PAIR_COLUMNS = [
    "target_true_label",
    "donor_true_label",
    "target_recorded_label_after_swap",
    "donor_recorded_label_after_swap",
]


def require_dataset_list(cfg: dict[str, object]) -> list[dict[str, object]]:
    datasets = cfg["datasets"]
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a JSON list")
    typed: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise TypeError(f"datasets[{index}] must be a JSON object")
        require_keys(dataset, REQUIRED_DATASET_KEYS, f"datasets[{index}]")
        if not isinstance(dataset["models"], list):
            raise TypeError(f"datasets[{index}].models must be a JSON list")
        invalid = [model for model in dataset["models"] if not isinstance(model, str)]
        if invalid:
            raise TypeError(f"datasets[{index}].models must contain only strings")
        typed.append(dataset)
    return typed


def read_csv_required(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    return frame


def sorted_equal(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(np.sort(left.to_numpy(dtype=float)), np.sort(right.to_numpy(dtype=float))))


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(float).mean())


def seed_level_history_checks(dataset: dict[str, object]) -> pd.DataFrame:
    run_dir = Path(str(dataset["run_dir"]))
    history = read_csv_required(
        run_dir / str(dataset["initial_history_file"]),
        REQUIRED_HISTORY_COLUMNS + [str(dataset["target_indicator_column"])],
    )
    clean_mode = str(dataset["clean_mode"])
    random_mode = str(dataset["random_mode"])
    targeted_mode = str(dataset["targeted_mode"])
    target_column = str(dataset["target_indicator_column"])
    rows: list[dict[str, object]] = []
    for seed, seed_frame in history.groupby("seed", sort=True):
        clean = seed_frame[seed_frame["mode"].astype(str) == clean_mode]
        random = seed_frame[seed_frame["mode"].astype(str) == random_mode]
        targeted = seed_frame[seed_frame["mode"].astype(str) == targeted_mode]
        if clean.empty or random.empty or targeted.empty:
            raise ValueError(f"missing clean/random/targeted history for {dataset['name']} seed={seed}")
        clean_target = clean[clean[target_column].astype(bool)]
        random_target = random[random[target_column].astype(bool)]
        targeted_target = targeted[targeted[target_column].astype(bool)]
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "seed": int(seed),
                "history_count": int(len(clean)),
                "target_slice_count": int(len(targeted_target)),
                "targeted_vs_clean_label_multiset_preserved": sorted_equal(
                    clean["recorded_label"],
                    targeted["recorded_label"],
                ),
                "targeted_vs_random_label_multiset_preserved": sorted_equal(
                    random["recorded_label"],
                    targeted["recorded_label"],
                ),
                "global_mean_delta_targeted_minus_clean": float(
                    targeted["recorded_label"].astype(float).mean()
                    - clean["recorded_label"].astype(float).mean()
                ),
                "global_mean_delta_targeted_minus_random": float(
                    targeted["recorded_label"].astype(float).mean()
                    - random["recorded_label"].astype(float).mean()
                ),
                "known_slice_recorded_mean_clean": safe_mean(clean_target["recorded_label"]),
                "known_slice_recorded_mean_random": safe_mean(random_target["recorded_label"]),
                "known_slice_recorded_mean_targeted": safe_mean(targeted_target["recorded_label"]),
                "known_slice_shift_targeted_minus_clean": float(
                    safe_mean(targeted_target["recorded_label"])
                    - safe_mean(clean_target["recorded_label"])
                ),
                "known_slice_shift_targeted_minus_random": float(
                    safe_mean(targeted_target["recorded_label"])
                    - safe_mean(random_target["recorded_label"])
                ),
            }
        )
    return pd.DataFrame(rows)


def pair_level_checks(dataset: dict[str, object]) -> dict[str, object]:
    run_dir = Path(str(dataset["run_dir"]))
    pairs = read_csv_required(run_dir / str(dataset["pairs_file"]), REQUIRED_PAIR_COLUMNS)
    if "seed" not in pairs.columns:
        pairs = pairs.copy()
        pairs["seed"] = -1
    rows: list[dict[str, object]] = []
    for seed, frame in pairs.groupby("seed", sort=True):
        clean_labels = pd.concat(
            [frame["target_true_label"], frame["donor_true_label"]],
            ignore_index=True,
        )
        recorded_labels = pd.concat(
            [
                frame["target_recorded_label_after_swap"],
                frame["donor_recorded_label_after_swap"],
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "seed": int(seed),
                "pair_multiset_preserved": sorted_equal(clean_labels, recorded_labels),
                "target_pair_recorded_shift": float(
                    frame["target_recorded_label_after_swap"].astype(float).mean()
                    - frame["target_true_label"].astype(float).mean()
                ),
                "donor_pair_recorded_shift": float(
                    frame["donor_recorded_label_after_swap"].astype(float).mean()
                    - frame["donor_true_label"].astype(float).mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    return {
        "pair_groups": int(len(summary)),
        "pair_multiset_preserved_rate": float(summary["pair_multiset_preserved"].mean()),
        "target_pair_recorded_shift_mean": float(summary["target_pair_recorded_shift"].mean()),
        "donor_pair_recorded_shift_mean": float(summary["donor_pair_recorded_shift"].mean()),
    }


def summary_model_checks(dataset: dict[str, object]) -> pd.DataFrame:
    run_dir = Path(str(dataset["run_dir"]))
    summary = read_csv_required(
        run_dir / str(dataset["summary_file"]),
        [
            "model",
            "mode",
            "selected_true_mean",
            "final_triggered_target_count_excess_vs_random",
            "r2_audit_mean",
        ],
    )
    random_mode = str(dataset["random_mode"])
    targeted_mode = str(dataset["targeted_mode"])
    rows: list[dict[str, object]] = []
    for model in [str(model) for model in dataset["models"]]:
        model_frame = summary[summary["model"].astype(str) == model]
        random = model_frame[model_frame["mode"].astype(str) == random_mode]
        targeted = model_frame[model_frame["mode"].astype(str) == targeted_mode]
        if random.empty or targeted.empty:
            raise ValueError(f"missing model summary for {dataset['name']} model={model}")
        random_row = random.iloc[0]
        targeted_row = targeted.iloc[0]
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "model": model,
                "audit_r2_random": float(random_row["r2_audit_mean"]),
                "audit_r2_targeted": float(targeted_row["r2_audit_mean"]),
                "audit_r2_delta_targeted_minus_random": float(
                    targeted_row["r2_audit_mean"] - random_row["r2_audit_mean"]
                ),
                "selected_true_shortfall_vs_random": float(
                    random_row["selected_true_mean"] - targeted_row["selected_true_mean"]
                ),
                "final_false_axis_excess_vs_random": float(
                    targeted_row["final_triggered_target_count_excess_vs_random"]
                ),
            }
        )
    return pd.DataFrame(rows)


def rows_for_dataset(dataset: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    history = seed_level_history_checks(dataset)
    pairs = pair_level_checks(dataset)
    model_checks = summary_model_checks(dataset)
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "dataset": str(dataset["name"]),
            "check": "marginal label multiset",
            "requires_known_axis": "no",
            "online_surface": "initial records",
            "target_control_contrast": float(
                abs(history["global_mean_delta_targeted_minus_random"]).mean()
            ),
            "detection_readout": "targeted label multiset equals random/clean by construction",
            "detected_in_primary_runs": "no",
            "boundary": "Blind to paired relabeling that preserves recorded-label multiset.",
        }
    )
    rows.append(
        {
            "dataset": str(dataset["name"]),
            "check": "global recorded-label mean",
            "requires_known_axis": "no",
            "online_surface": "initial records",
            "target_control_contrast": float(
                abs(history["global_mean_delta_targeted_minus_random"]).mean()
            ),
            "detection_readout": "absolute mean delta targeted minus random",
            "detected_in_primary_runs": "no",
            "boundary": "Global moments are unchanged up to numerical precision under paired swaps.",
        }
    )
    rows.append(
        {
            "dataset": str(dataset["name"]),
            "check": "known target-slice mean",
            "requires_known_axis": "yes",
            "online_surface": "initial records",
            "target_control_contrast": float(
                history["known_slice_shift_targeted_minus_random"].mean()
            ),
            "detection_readout": "target-slice recorded mean shift targeted minus random",
            "detected_in_primary_runs": "yes",
            "boundary": "Effective when the correct slice/provenance group is already known and populated.",
        }
    )
    rows.append(
        {
            "dataset": str(dataset["name"]),
            "check": "paired donor-target conditional contrast",
            "requires_known_axis": "yes",
            "online_surface": "initial records",
            "target_control_contrast": float(pairs["target_pair_recorded_shift_mean"]),
            "detection_readout": "target-side recorded shift over swapped pairs",
            "detected_in_primary_runs": "yes",
            "boundary": "Requires knowing or reconstructing the implicated donor-target pairing.",
        }
    )
    for _, model_row in model_checks.iterrows():
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "check": f"same-distribution audit R2 ({model_row['model']})",
                "requires_known_axis": "no",
                "online_surface": "held-out records from same recorded distribution",
                "target_control_contrast": float(model_row["audit_r2_delta_targeted_minus_random"]),
                "detection_readout": "audit R2 delta targeted minus random",
                "detected_in_primary_runs": "weak/no",
                "boundary": "Predictive skill on the recorded distribution can remain nontrivial while the scientific binding is wrong.",
            }
        )
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "check": f"same-budget true-response shortfall ({model_row['model']})",
                "requires_known_axis": "no",
                "online_surface": "executed feedback",
                "target_control_contrast": float(model_row["selected_true_shortfall_vs_random"]),
                "detection_readout": "selected true mean shortfall targeted versus random",
                "detected_in_primary_runs": "yes after feedback",
                "boundary": "Measures harm after budget has been executed; not an early provenance detector.",
            }
        )
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "check": f"false-axis acquisition excess ({model_row['model']})",
                "requires_known_axis": "yes",
                "online_surface": "proposed or executed acquisition trace",
                "target_control_contrast": float(model_row["final_false_axis_excess_vs_random"]),
                "detection_readout": "final false-axis count excess targeted versus random",
                "detected_in_primary_runs": "yes",
                "boundary": "Requires either a monitored slice or an all-axis scan plus control calibration.",
            }
        )
    summary = {
        "dataset": str(dataset["name"]),
        "history_seeds": int(len(history)),
        "history_multiset_preserved_rate_targeted_vs_clean": float(
            history["targeted_vs_clean_label_multiset_preserved"].mean()
        ),
        "history_multiset_preserved_rate_targeted_vs_random": float(
            history["targeted_vs_random_label_multiset_preserved"].mean()
        ),
        "global_mean_abs_delta_targeted_minus_random": float(
            abs(history["global_mean_delta_targeted_minus_random"]).mean()
        ),
        "known_slice_shift_targeted_minus_random": float(
            history["known_slice_shift_targeted_minus_random"].mean()
        ),
        "pair_multiset_preserved_rate": float(pairs["pair_multiset_preserved_rate"]),
        "target_pair_recorded_shift_mean": float(pairs["target_pair_recorded_shift_mean"]),
    }
    return rows, summary


def format_float(value: object) -> str:
    numeric = float(value)
    if abs(numeric) >= 10:
        return f"{numeric:.1f}"
    return f"{numeric:.3f}"


def write_markdown(path: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B66 Sanity-Check Boundary Analysis",
        "",
        "Hypothesis: marginal label checks and same-distribution predictive audits do not identify the constructed binding rewrite, while known-slice and acquisition-trace checks can expose it under explicit scope assumptions.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Seeds | History multiset preserved vs clean | History multiset preserved vs random | Global mean abs delta vs random | Known-slice shift vs random | Pair shift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {history_seeds} | {history_multiset_preserved_rate_targeted_vs_clean:.3f} | {history_multiset_preserved_rate_targeted_vs_random:.3f} | {global_mean_abs_delta_targeted_minus_random:.6f} | {known_slice_shift_targeted_minus_random:.3f} | {target_pair_recorded_shift_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Boundary Table",
            "",
            "| Dataset | Check | Known axis? | Surface | Contrast | Detected? | Boundary |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {check} | {requires_known_axis} | {online_surface} | {contrast} | {detected_in_primary_runs} | {boundary} |".format(
                dataset=row["dataset"],
                check=row["check"],
                requires_known_axis=row["requires_known_axis"],
                online_surface=row["online_surface"],
                contrast=format_float(row["target_control_contrast"]),
                detected_in_primary_runs=row["detected_in_primary_runs"],
                boundary=row["boundary"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Supported: simple marginal label and global-moment checks are blind in the primary paired-swap constructions; known-slice mean checks and trace monitors can expose the same constructions, but they require either the right slice/provenance group, executed feedback, or control-calibrated acquisition traces.",
            "",
            "Not supported: universal stealth, failure of every possible provenance audit, or a calibration-free deployable detector.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [
        row
        for row in rows
        if row["check"]
        in {
            "marginal label multiset",
            "global recorded-label mean",
            "known target-slice mean",
            "paired donor-target conditional contrast",
        }
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Sanity-check boundaries for the primary paired-misbinding constructions.}",
        "\\label{tab:sanity-check-boundaries}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrl}",
        "\\toprule",
        "Dataset & Check & Known axis? & Contrast & Boundary \\\\",
        "\\midrule",
    ]
    for row in selected:
        boundary = str(row["boundary"]).replace("_", "\\_")
        check = str(row["check"]).replace("_", "\\_")
        dataset = str(row["dataset"]).replace("_", "\\_")
        lines.append(
            f"{dataset} & {check} & {row['requires_known_axis']} & {format_float(row['target_control_contrast'])} & {boundary} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\vspace{0.4em}\\caption*{\\footnotesize Contrast is the targeted-minus-random diagnostic contrast for the corresponding check. Marginal and global checks remain near zero by construction; known-slice and pairwise conditional checks expose the rewrite only when the implicated slice or pairing is available.}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B66 sanity-check boundaries.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b66_sanity_check_boundaries")
    datasets = require_dataset_list(cfg)
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_rows, summary = rows_for_dataset(dataset)
        rows.extend(dataset_rows)
        summaries.append(summary)

    output_csv = Path(str(cfg["output_csv"]))
    output_json = Path(str(cfg["output_json"]))
    output_md = Path(str(cfg["output_md"]))
    output_tex = Path(str(cfg["output_tex"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summaries": summaries}, handle, indent=2, sort_keys=True)
    write_markdown(output_md, rows, summaries)
    write_tex(output_tex, rows)
    print(output_csv)
    print(output_json)
    print(output_md)
    print(output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
