#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from false_science.config import (
    config_for_metadata,
    load_json_config,
    parse_config_arg,
    require_keys,
)
from false_science.protein import load_gfp_csv
from false_science.target_scan import (
    TargetScanConfig,
    file_sha256,
    make_run_dir,
    scan_target_regions,
    write_scan_artifacts,
)


REQUIRED_CONFIG_KEYS = [
    "data_path",
    "output_root",
    "tag",
    "target_column",
    "mutant_column",
    "max_rows",
    "random_state",
    "min_target_count",
    "min_target_prevalence",
    "max_target_prevalence",
    "target_mean_quantile",
    "donor_quantile",
    "min_swap_count",
    "max_targets",
    "tag_prefixes",
    "candidate_pair_count",
]


def parse_args() -> argparse.Namespace:
    config_path = parse_config_arg("M0 scan for low-performing GFP target regions.")
    cfg = load_json_config(config_path)
    require_keys(cfg, REQUIRED_CONFIG_KEYS, "m0_scan_gfp_targets")
    return argparse.Namespace(**cfg)


def main() -> int:
    args = parse_args()
    cfg = TargetScanConfig(
        data_path=args.data_path,
        target_column=args.target_column,
        mutant_column=args.mutant_column,
        max_rows=args.max_rows,
        random_state=args.random_state,
        min_target_count=args.min_target_count,
        min_target_prevalence=args.min_target_prevalence,
        max_target_prevalence=args.max_target_prevalence,
        target_mean_quantile=args.target_mean_quantile,
        donor_quantile=args.donor_quantile,
        min_swap_count=args.min_swap_count,
        max_targets=args.max_targets,
        tag_prefixes=tuple(args.tag_prefixes),
    )

    data_path = Path(cfg.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"GFP data not found: {data_path}")

    df = load_gfp_csv(
        data_path,
        target_column=cfg.target_column,
        mutant_column=cfg.mutant_column,
        max_rows=cfg.max_rows,
        random_state=cfg.random_state,
    )
    scan, tag_sets = scan_target_regions(df, cfg)
    if cfg.max_targets and len(scan) > cfg.max_targets:
        scan = scan.head(cfg.max_targets).copy()

    run_dir = make_run_dir(args.output_root, args.tag)
    summary = write_scan_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        df=df,
        scan=scan,
        tag_sets=tag_sets,
        data_sha256=file_sha256(data_path),
        candidate_pair_count=args.candidate_pair_count,
        config_metadata=config_for_metadata(vars(args)),
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))
    if not summary["n_passing_targets"]:
        raise RuntimeError(f"M0 found no passing target regions; run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
