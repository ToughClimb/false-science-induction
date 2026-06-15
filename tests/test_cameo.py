from __future__ import annotations

import io
import importlib.util
import zipfile
from pathlib import Path

import numpy as np

from false_science.cameo import (
    CAMEO_PREFIX,
    build_cameo_dataset,
    scan_cameo_region_targets,
    select_cameo_target_region,
    select_low_target_and_high_donor_ids,
)


def write_cameo_fixture(path) -> None:
    composition = "Fe\tGa\tPd\n0.8\t0.1\t0.1\n0.7\t0.2\t0.1\n0.5\t0.4\t0.1\n0.4\t0.5\t0.1\n0.3\t0.3\t0.4\n0.2\t0.3\t0.5\n"
    xrd_header = ",".join(str(37 + 0.02 * idx) for idx in range(4))
    xrd_rows = "\n".join(
        ",".join(str(float(row + col)) for col in range(4)) for row in range(6)
    )
    mag = "10,10\n9,9\n1,1\n0,0\n8,8\n7,7\n"
    regions = "0\n0\n2\n2\n1\n1\n"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{CAMEO_PREFIX}FeGaPd_CMP.txt", composition)
        zf.writestr(f"{CAMEO_PREFIX}FeGaPd_XRD.txt", f"{xrd_header}\n{xrd_rows}\n")
        zf.writestr(f"{CAMEO_PREFIX}FeGaPd_Mag.txt", mag)
        zf.writestr(f"{CAMEO_PREFIX}FeGaPd_DFT_regions.txt", regions)


def test_cameo_loader_and_region_scan(tmp_path) -> None:
    zip_path = tmp_path / "cameo.zip"
    write_cameo_fixture(zip_path)

    dataset = build_cameo_dataset(zip_path, xrd_pca_components=2, pca_seed=0)

    assert dataset.x.shape == (6, 5)
    assert dataset.frame["dft_region"].tolist() == [0, 0, 2, 2, 1, 1]
    scan = scan_cameo_region_targets(
        dataset.y,
        dataset.frame["dft_region"].to_numpy(),
        min_target_count=2,
        donor_quantile=0.6,
    )
    assert select_cameo_target_region(scan, None) == 2


def test_cameo_swap_selection_uses_low_targets_and_high_donors() -> None:
    y = np.array([10.0, 9.0, 1.0, 0.0, 8.0, 7.0])
    target_mask = np.array([False, False, True, True, False, False])

    target_ids, donor_ids = select_low_target_and_high_donor_ids(
        y,
        target_mask,
        donor_quantile=0.5,
        swap_count=2,
    )

    assert target_ids.tolist() == [3, 2]
    assert donor_ids.tolist() == [0, 1]


def test_b31_script_imports() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "b31_cameo_retrospective_replay.py"
    spec = importlib.util.spec_from_file_location("b31_cameo_retrospective_replay", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
