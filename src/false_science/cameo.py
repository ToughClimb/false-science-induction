from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CAMEO_PREFIX = "CAMEO_NComm-master/data/"


@dataclass(frozen=True)
class CameoRawData:
    composition: np.ndarray
    xrd: np.ndarray
    magnetization: np.ndarray
    dft_regions: np.ndarray
    composition_columns: tuple[str, ...] = ("Fe", "Ga", "Pd")


@dataclass(frozen=True)
class CameoDataset:
    frame: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_column: str
    region_column: str


def _read_zip_text(zip_file: zipfile.ZipFile, name: str) -> str:
    try:
        return zip_file.read(f"{CAMEO_PREFIX}{name}").decode("utf-8")
    except KeyError as exc:
        raise FileNotFoundError(f"CAMEO archive missing {name}") from exc


def load_cameo_raw(zip_path: str | Path) -> CameoRawData:
    """Load the public CAMEO Fe-Ga-Pd data from the NIST zip package."""
    with zipfile.ZipFile(zip_path) as zip_file:
        composition = np.loadtxt(
            io.StringIO(_read_zip_text(zip_file, "FeGaPd_CMP.txt")),
            skiprows=1,
        )
        xrd = np.loadtxt(
            io.StringIO(_read_zip_text(zip_file, "FeGaPd_XRD.txt")),
            delimiter=",",
            skiprows=1,
        )
        magnetization = np.loadtxt(
            io.StringIO(_read_zip_text(zip_file, "FeGaPd_Mag.txt")),
            delimiter=",",
        )
        dft_regions = np.loadtxt(
            io.StringIO(_read_zip_text(zip_file, "FeGaPd_DFT_regions.txt"))
        ).astype(int)
    n_rows = composition.shape[0]
    for name, values in {
        "xrd": xrd,
        "magnetization": magnetization,
        "dft_regions": dft_regions,
    }.items():
        if values.shape[0] != n_rows:
            raise ValueError(f"CAMEO row mismatch for {name}: {values.shape[0]} != {n_rows}")
    return CameoRawData(
        composition=composition.astype(float),
        xrd=xrd.astype(float),
        magnetization=magnetization.astype(float),
        dft_regions=dft_regions.astype(int),
    )


def build_cameo_dataset(
    zip_path: str | Path,
    xrd_pca_components: int,
    pca_seed: int,
    target_column: str = "magnetization_modified",
) -> CameoDataset:
    raw = load_cameo_raw(zip_path)
    if target_column not in {"magnetization_raw", "magnetization_modified"}:
        raise ValueError(f"unknown CAMEO target column: {target_column}")
    if xrd_pca_components < 0:
        raise ValueError("xrd_pca_components must be non-negative")

    frame = pd.DataFrame(
        {
            "record_id": np.arange(len(raw.dft_regions), dtype=int),
            "Fe": raw.composition[:, 0],
            "Ga": raw.composition[:, 1],
            "Pd": raw.composition[:, 2],
            "magnetization_raw": raw.magnetization[:, 0],
            "magnetization_modified": raw.magnetization[:, 1],
            "dft_region": raw.dft_regions,
        }
    )
    feature_blocks = [raw.composition.astype(np.float32)]
    feature_names = ["Fe", "Ga", "Pd"]
    if xrd_pca_components > 0:
        max_components = min(raw.xrd.shape[0], raw.xrd.shape[1])
        if xrd_pca_components > max_components:
            raise ValueError(
                f"xrd_pca_components={xrd_pca_components} exceeds max {max_components}"
            )
        xrd_scaled = StandardScaler().fit_transform(raw.xrd).astype(np.float32)
        pca = PCA(n_components=xrd_pca_components, random_state=pca_seed)
        xrd_pca = pca.fit_transform(xrd_scaled).astype(np.float32)
        feature_blocks.append(xrd_pca)
        feature_names.extend([f"xrd_pc_{idx + 1:02d}" for idx in range(xrd_pca_components)])
        for idx, name in enumerate(feature_names[3:]):
            frame[name] = xrd_pca[:, idx]

    x = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    y = frame[target_column].to_numpy(dtype=float)
    return CameoDataset(
        frame=frame,
        x=x,
        y=y,
        feature_names=feature_names,
        target_column=target_column,
        region_column="dft_region",
    )


def scan_cameo_region_targets(
    y: np.ndarray,
    regions: np.ndarray,
    min_target_count: int,
    donor_quantile: float,
) -> pd.DataFrame:
    donor_cutoff = float(np.quantile(y, donor_quantile))
    global_mean = float(np.mean(y))
    rows: list[dict[str, object]] = []
    for region in sorted(set(regions.astype(int).tolist())):
        mask = regions == region
        non_target = ~mask
        donor_mask = non_target & (y >= donor_cutoff)
        target_y = y[mask]
        donor_count = int(donor_mask.sum())
        target_mean = float(np.mean(target_y))
        donor_mean = float(np.mean(y[donor_mask])) if donor_count else float("nan")
        rows.append(
            {
                "target_region": int(region),
                "target_tag": f"dft_region={int(region)}",
                "target_count": int(mask.sum()),
                "target_prevalence": float(np.mean(mask)),
                "target_mean": target_mean,
                "target_median": float(np.median(target_y)),
                "target_q10": float(np.quantile(target_y, 0.10)),
                "target_q90": float(np.quantile(target_y, 0.90)),
                "global_mean": global_mean,
                "donor_cutoff": donor_cutoff,
                "donor_count": donor_count,
                "donor_mean": donor_mean,
                "target_donor_contrast": donor_mean - target_mean if donor_count else float("nan"),
                "max_swap_count": int(min(mask.sum(), donor_count)),
                "passes_gate": bool(mask.sum() >= min_target_count and donor_count > 0),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["passes_gate", "target_mean", "target_count"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def select_cameo_target_region(scan: pd.DataFrame, target_region: int | None) -> int:
    if scan.empty or not scan["passes_gate"].any():
        raise ValueError("no CAMEO DFT region passed target gate")
    if target_region is None:
        return int(scan[scan["passes_gate"]].iloc[0]["target_region"])
    row = scan[scan["target_region"] == int(target_region)]
    if row.empty:
        raise ValueError(f"unknown CAMEO target region: {target_region}")
    if not bool(row.iloc[0]["passes_gate"]):
        raise ValueError(f"CAMEO target region did not pass gate: {target_region}")
    return int(target_region)


def select_low_target_and_high_donor_ids(
    y: np.ndarray,
    target_mask: np.ndarray,
    donor_quantile: float,
    swap_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    donor_cutoff = float(np.quantile(y, donor_quantile))
    target_ids = np.flatnonzero(target_mask)
    donor_ids = np.flatnonzero((~target_mask) & (y >= donor_cutoff))
    target_order = target_ids[np.argsort(y[target_ids])]
    donor_order = donor_ids[np.argsort(-y[donor_ids])]
    if len(target_order) < swap_count:
        raise ValueError(f"only {len(target_order)} target records for {swap_count} swaps")
    if len(donor_order) < swap_count:
        raise ValueError(f"only {len(donor_order)} donor records for {swap_count} swaps")
    return target_order[:swap_count].astype(int), donor_order[:swap_count].astype(int)
