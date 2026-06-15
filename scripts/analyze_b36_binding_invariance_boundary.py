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

from false_science.config import load_json_config, parse_config_arg, require_keys


REQUIRED_CONFIG_KEYS = [
    "output_csv",
    "output_json",
    "output_md",
    "datasets",
]

REQUIRED_DATASET_KEYS = [
    "name",
    "run_dir",
    "pairs_file",
    "target_axis",
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
        typed.append(dataset)
    return typed


def load_pairs(dataset: dict[str, object]) -> pd.DataFrame:
    path = Path(str(dataset["run_dir"])) / str(dataset["pairs_file"])
    if not path.is_file():
        raise FileNotFoundError(f"swap pairs not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_PAIR_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")
    if "seed" not in frame.columns:
        frame = frame.copy()
        frame["seed"] = -1
    return frame


def multiset_preserved(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(np.sort(left.astype(float)), np.sort(right.astype(float))))


def rows_for_dataset(dataset: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = load_pairs(dataset)
    rows: list[dict[str, object]] = []
    for seed, subset in frame.groupby("seed"):
        clean_labels = np.concatenate(
            [
                subset["target_true_label"].to_numpy(dtype=float),
                subset["donor_true_label"].to_numpy(dtype=float),
            ]
        )
        recorded_labels = np.concatenate(
            [
                subset["target_recorded_label_after_swap"].to_numpy(dtype=float),
                subset["donor_recorded_label_after_swap"].to_numpy(dtype=float),
            ]
        )
        target_shift = float(
            subset["target_recorded_label_after_swap"].mean()
            - subset["target_true_label"].mean()
        )
        donor_shift = float(
            subset["donor_recorded_label_after_swap"].mean()
            - subset["donor_true_label"].mean()
        )
        contrast_clean = float(
            subset["donor_true_label"].mean() - subset["target_true_label"].mean()
        )
        contrast_recorded = float(
            subset["target_recorded_label_after_swap"].mean()
            - subset["donor_recorded_label_after_swap"].mean()
        )
        rows.append(
            {
                "dataset": str(dataset["name"]),
                "target_axis": str(dataset["target_axis"]),
                "seed": "all" if int(seed) < 0 else str(int(seed)),
                "n_pairs": int(len(subset)),
                "label_multiset_preserved": multiset_preserved(clean_labels, recorded_labels),
                "target_true_mean": float(subset["target_true_label"].mean()),
                "target_recorded_mean": float(
                    subset["target_recorded_label_after_swap"].mean()
                ),
                "target_recorded_shift": target_shift,
                "donor_true_mean": float(subset["donor_true_label"].mean()),
                "donor_recorded_mean": float(
                    subset["donor_recorded_label_after_swap"].mean()
                ),
                "donor_recorded_shift": donor_shift,
                "clean_donor_minus_target_contrast": contrast_clean,
                "recorded_target_minus_donor_contrast": contrast_recorded,
            }
        )
    summary = {
        "dataset": str(dataset["name"]),
        "target_axis": str(dataset["target_axis"]),
        "n_seeds": int(len(rows)),
        "n_pairs_mean": float(pd.DataFrame(rows)["n_pairs"].mean()),
        "all_label_multisets_preserved": bool(
            all(bool(row["label_multiset_preserved"]) for row in rows)
        ),
        "target_recorded_shift_mean": float(
            pd.DataFrame(rows)["target_recorded_shift"].mean()
        ),
        "donor_recorded_shift_mean": float(pd.DataFrame(rows)["donor_recorded_shift"].mean()),
        "clean_contrast_mean": float(
            pd.DataFrame(rows)["clean_donor_minus_target_contrast"].mean()
        ),
        "recorded_contrast_mean": float(
            pd.DataFrame(rows)["recorded_target_minus_donor_contrast"].mean()
        ),
    }
    return rows, summary


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# B36 Binding-Invariance Boundary",
        "",
        "This analysis quantifies the construction-level boundary: paired swaps preserve the marginal label multiset while rewriting the target/donor conditional relation.",
        "",
        "| Dataset | Axis | Seeds | Pairs/seed | Label multiset preserved | Target recorded shift | Donor recorded shift | Clean donor-target contrast | Recorded target-donor contrast |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {target_axis} | {n_seeds} | {n_pairs_mean:.1f} | {all_label_multisets_preserved} | {target_recorded_shift_mean:.3f} | {donor_recorded_shift_mean:.3f} | {clean_contrast_mean:.3f} | {recorded_contrast_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: marginal label-only diagnostics are blind by construction, while the conditional target/donor relation is inverted within the swapped records.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config_path = parse_config_arg("Analyze B36 binding-invariance boundary.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "b36_binding_invariance_boundary")
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summaries": summaries}, handle, indent=2, sort_keys=True)
    write_markdown(output_md, summaries)
    print(output_csv)
    print(output_json)
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
