from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from false_science.protein import mutation_tags


SCAN_COLUMNS = [
    "tag",
    "tag_kind",
    "target_count",
    "target_prevalence",
    "target_mean",
    "target_median",
    "target_q10",
    "target_q90",
    "target_top_rate_at_donor_cutoff",
    "global_mean",
    "target_mean_cutoff",
    "donor_cutoff",
    "donor_count",
    "donor_mean",
    "target_donor_contrast",
    "max_swap_count",
    "passes_m0_gate",
]


@dataclass(frozen=True)
class TargetScanConfig:
    data_path: str
    target_column: str
    mutant_column: str
    max_rows: int | None
    random_state: int
    min_target_count: int
    min_target_prevalence: float
    max_target_prevalence: float
    target_mean_quantile: float
    donor_quantile: float
    min_swap_count: int
    max_targets: int
    tag_prefixes: tuple[str, ...]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def make_run_dir(output_root: str | Path, tag: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{timestamp()}_{tag}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def attach_tags(df: pd.DataFrame, mutant_column: str) -> list[set[str]]:
    return [mutation_tags(mutant) for mutant in df[mutant_column].astype(str)]


def iter_candidate_tags(
    tag_sets: Iterable[set[str]],
    prefixes: tuple[str, ...],
) -> list[str]:
    tags = {
        tag
        for tag_set in tag_sets
        for tag in tag_set
        if any(tag.startswith(prefix) for prefix in prefixes)
    }
    return sorted(tags)


def scan_target_regions(
    df: pd.DataFrame,
    cfg: TargetScanConfig,
) -> tuple[pd.DataFrame, list[set[str]]]:
    y = df[cfg.target_column].to_numpy(dtype=float)
    tag_sets = attach_tags(df, cfg.mutant_column)
    global_mean = float(np.mean(y))
    target_mean_cutoff = float(np.quantile(y, cfg.target_mean_quantile))
    donor_cutoff = float(np.quantile(y, cfg.donor_quantile))
    min_count = max(
        int(cfg.min_target_count),
        int(np.ceil(cfg.min_target_prevalence * len(df))),
    )

    rows: list[dict[str, object]] = []
    for tag in iter_candidate_tags(tag_sets, cfg.tag_prefixes):
        mask = np.array([tag in tags for tags in tag_sets], dtype=bool)
        count = int(mask.sum())
        prevalence = float(count / len(df)) if len(df) else 0.0
        if count < min_count or prevalence > cfg.max_target_prevalence:
            continue

        target_y = y[mask]
        non_target = ~mask
        donor_mask = non_target & (y >= donor_cutoff)
        donor_count = int(donor_mask.sum())
        max_swap_count = int(min(count, donor_count))
        target_mean = float(np.mean(target_y))
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        contrast = donor_mean - target_mean if donor_count else float("nan")
        target_top_rate = float(np.mean(target_y >= donor_cutoff))

        passes = (
            target_mean <= target_mean_cutoff
            and max_swap_count >= cfg.min_swap_count
            and donor_count >= cfg.min_swap_count
        )
        rows.append(
            {
                "tag": tag,
                "tag_kind": tag.split("=", 1)[0],
                "target_count": count,
                "target_prevalence": prevalence,
                "target_mean": target_mean,
                "target_median": float(np.median(target_y)),
                "target_q10": float(np.quantile(target_y, 0.10)),
                "target_q90": float(np.quantile(target_y, 0.90)),
                "target_top_rate_at_donor_cutoff": target_top_rate,
                "global_mean": global_mean,
                "target_mean_cutoff": target_mean_cutoff,
                "donor_cutoff": donor_cutoff,
                "donor_count": donor_count,
                "donor_mean": donor_mean,
                "target_donor_contrast": contrast,
                "max_swap_count": max_swap_count,
                "passes_m0_gate": bool(passes),
            }
        )

    result = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    if result.empty:
        return result, tag_sets
    result = result.sort_values(
        by=[
            "passes_m0_gate",
            "target_donor_contrast",
            "target_count",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return result, tag_sets


def select_swap_pairs(
    df: pd.DataFrame,
    tag_sets: list[set[str]],
    target_tag: str,
    cfg: TargetScanConfig,
    swap_count: int,
) -> pd.DataFrame:
    y = df[cfg.target_column].to_numpy(dtype=float)
    donor_cutoff = float(np.quantile(y, cfg.donor_quantile))
    target_idx = np.array(
        [idx for idx, tags in enumerate(tag_sets) if target_tag in tags],
        dtype=int,
    )
    donor_idx = np.array(
        [
            idx
            for idx, tags in enumerate(tag_sets)
            if target_tag not in tags and y[idx] >= donor_cutoff
        ],
        dtype=int,
    )
    target_order = target_idx[np.argsort(y[target_idx])]
    donor_order = donor_idx[np.argsort(-y[donor_idx])]
    n = min(int(swap_count), len(target_order), len(donor_order))
    if n <= 0:
        return pd.DataFrame()

    target_selected = target_order[:n]
    donor_selected = donor_order[:n]
    return pd.DataFrame(
        {
            "pair_id": np.arange(n, dtype=int),
            "target_record_id": df.loc[target_selected, "record_id"].to_numpy(dtype=int),
            "donor_record_id": df.loc[donor_selected, "record_id"].to_numpy(dtype=int),
            "target_mutant": df.loc[target_selected, cfg.mutant_column].to_numpy(),
            "donor_mutant": df.loc[donor_selected, cfg.mutant_column].to_numpy(),
            "target_true_label": y[target_selected],
            "donor_true_label": y[donor_selected],
            "target_recorded_label_after_swap": y[donor_selected],
            "donor_recorded_label_after_swap": y[target_selected],
            "target_tag": target_tag,
        }
    )


def label_multiset_preserved(pairs: pd.DataFrame) -> bool:
    before = np.sort(
        np.concatenate(
            [
                pairs["target_true_label"].to_numpy(dtype=float),
                pairs["donor_true_label"].to_numpy(dtype=float),
            ]
        )
    )
    after = np.sort(
        np.concatenate(
            [
                pairs["target_recorded_label_after_swap"].to_numpy(dtype=float),
                pairs["donor_recorded_label_after_swap"].to_numpy(dtype=float),
            ]
        )
    )
    return bool(np.array_equal(before, after))


def write_scan_artifacts(
    run_dir: Path,
    cfg: TargetScanConfig,
    df: pd.DataFrame,
    scan: pd.DataFrame,
    tag_sets: list[set[str]],
    data_sha256: str,
    candidate_pair_count: int,
    config_metadata: dict[str, object],
) -> dict[str, object]:
    passing = scan[scan["passes_m0_gate"]] if not scan.empty else pd.DataFrame()
    selected = None if passing.empty else passing.iloc[0].to_dict()
    pairs = pd.DataFrame()
    if selected is not None:
        pairs = select_swap_pairs(
            df,
            tag_sets,
            target_tag=str(selected["tag"]),
            cfg=cfg,
            swap_count=int(min(selected["max_swap_count"], candidate_pair_count)),
        )

    scan.to_csv(run_dir / "target_scan.csv", index=False)
    df.to_csv(run_dir / "dataset_snapshot.csv", index=False)
    pairs.to_csv(run_dir / "candidate_swap_pairs.csv", index=False)

    summary: dict[str, object] = {
        "stage": "M0_target_region_scan",
        "data_path": cfg.data_path,
        "data_sha256": data_sha256,
        "n_records": int(len(df)),
        "n_candidate_tags": int(len(scan)),
        "n_passing_targets": int(len(passing)),
        "selected_target": selected,
        "swap_pair_count": int(len(pairs)),
        "candidate_pair_count": int(candidate_pair_count),
        "label_multiset_preserved_for_selected_pairs": (
            label_multiset_preserved(pairs) if len(pairs) else False
        ),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short": git_text(["status", "--short"]),
        "config": config_metadata,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_metadata, handle, indent=2, sort_keys=True)
    return summary
