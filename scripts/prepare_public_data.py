#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
REVIEW_STAGE = REPO_ROOT / "review-stage"

DATASETS = {
    "gfp": {
        "path": RAW_DIR / "GFP_AEQVI_Sarkisyan_2016.csv",
        "urls": [
            "https://huggingface.co/datasets/ICML2022/ProteinGym/resolve/main/ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv",
        ],
        "note": (
            "If this download is slow, download GFP_AEQVI_Sarkisyan_2016.csv "
            "from the ProteinGym substitutions benchmark and place it at data/raw/."
        ),
    },
    "esol": {
        "path": RAW_DIR / "delaney-processed.csv",
        "urls": [
            "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        ],
        "note": "DeepChem ESOL/Delaney processed CSV.",
    },
    "cameo": {
        "path": REVIEW_STAGE / "CAMEO_NComm-master_20260530.zip",
        "urls": [
            "https://data.nist.gov/od/ds/mds2-2480/CAMEO_NComm-master.zip",
        ],
        "note": "NIST CAMEO Fe-Ga-Pd archive used by retrospective replay scripts.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "false-science-induction"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp.replace(destination)


def prepare_dataset(name: str, timeout: int, force: bool) -> bool:
    spec = DATASETS[name]
    destination = spec["path"]
    if destination.is_file() and not force:
        print(f"{name}: exists {destination} sha256={sha256(destination)}")
        return True
    last_error: Exception | None = None
    for url in spec["urls"]:
        try:
            print(f"{name}: downloading {url}")
            download(url, destination, timeout=timeout)
            print(f"{name}: wrote {destination} sha256={sha256(destination)}")
            return True
        except Exception as exc:
            last_error = exc
            print(f"{name}: download failed from {url}: {exc}", file=sys.stderr)
    print(f"{name}: manual step required. {spec['note']}", file=sys.stderr)
    if last_error is not None:
        print(f"{name}: last error: {last_error}", file=sys.stderr)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public datasets used by reproducibility configs."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to prepare. Repeatable. Defaults to gfp and esol.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all directly downloadable datasets, including CAMEO.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--timeout", type=int, help="Per-download timeout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeout = 120 if args.timeout is None else args.timeout
    if args.all:
        selected = sorted(DATASETS)
    else:
        selected = args.dataset or ["gfp", "esol"]
    ok = True
    for name in selected:
        ok = prepare_dataset(name, timeout=timeout, force=args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
