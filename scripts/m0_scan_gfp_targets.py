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

from false_science.protein import load_gfp_csv
from false_science.target_scan import (
    TargetScanConfig,
    file_sha256,
    make_run_dir,
    scan_target_regions,
    write_scan_artifacts,
)


DEFAULT_GFP_PATH = (
    "/home/misaka/inverse-ai4sci/data/protein_gfp/"
    "GFP_AEQVI_Sarkisyan_2016.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M0 scan for low-performing GFP target regions."
    )
    parser.add_argument("--data-path", default=DEFAULT_GFP_PATH)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--tag", default="m0-gfp-target-scan")
    parser.add_argument("--target-column", default="DMS_score")
    parser.add_argument("--mutant-column", default="mutant")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-target-count", type=int, default=100)
    parser.add_argument("--min-target-prevalence", type=float, default=0.02)
    parser.add_argument("--max-target-prevalence", type=float, default=0.35)
    parser.add_argument("--target-mean-quantile", type=float, default=0.40)
    parser.add_argument("--donor-quantile", type=float, default=0.90)
    parser.add_argument("--min-swap-count", type=int, default=25)
    parser.add_argument("--max-targets", type=int, default=50)
    parser.add_argument(
        "--tag-prefixes",
        nargs="*",
        default=None,
        help=(
            "Candidate tag prefixes to scan, for example --tag-prefixes pos= "
            "change=. Defaults to position, change, group, and mutation-count bins."
        ),
    )
    return parser.parse_args()


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
        tag_prefixes=tuple(args.tag_prefixes)
        if args.tag_prefixes
        else TargetScanConfig.tag_prefixes,
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
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))
    return 0 if summary["n_passing_targets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
