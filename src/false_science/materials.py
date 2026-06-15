from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def load_matminer_dataset(
    dataset_name: str,
    target_column: str,
    composition_column: str,
) -> pd.DataFrame:
    from matminer.datasets import load_dataset

    df = load_dataset(dataset_name)
    df = df.dropna(subset=[target_column, composition_column]).reset_index(drop=True)
    df = df.copy()
    df[target_column] = df[target_column].astype(float)
    df[composition_column] = df[composition_column].astype(str)
    df["record_id"] = df.index.astype(int)
    return df


def weighted_stats(name: str, values: list[tuple[float, float]]) -> dict[str, float]:
    if not values:
        return {
            f"{name}_mean": 0.0,
            f"{name}_min": 0.0,
            f"{name}_max": 0.0,
            f"{name}_range": 0.0,
        }
    raw = np.array([value for value, _ in values], dtype=float)
    weights = np.array([weight for _, weight in values], dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"{name} weights must have positive total")
    mean = float(np.sum(raw * weights) / total)
    return {
        f"{name}_mean": mean,
        f"{name}_min": float(np.min(raw)),
        f"{name}_max": float(np.max(raw)),
        f"{name}_range": float(np.max(raw) - np.min(raw)),
    }


def material_feature_frame(
    compositions: Iterable[str],
) -> tuple[pd.DataFrame, list[set[str]]]:
    from pymatgen.core import Composition, Element

    all_symbols = [element.symbol for element in Element]
    rows: list[dict[str, float]] = []
    tag_sets: list[set[str]] = []
    for formula in compositions:
        comp = Composition(str(formula))
        frac = {
            element.symbol: float(amount)
            for element, amount in comp.fractional_composition.items()
        }
        if not frac:
            raise ValueError(f"empty composition: {formula}")
        tags = {f"element={symbol}" for symbol in frac}
        tags.add(f"n_elements={len(frac)}")
        for symbol, amount in frac.items():
            if amount >= 0.25:
                tags.add(f"major_element={symbol}")
        if any(Element(symbol).is_transition_metal for symbol in frac):
            tags.add("chemistry=transition_metal")
        if any(symbol in {"O", "S", "Se", "Te"} for symbol in frac):
            tags.add("chemistry=chalcogenide")
        if any(symbol in {"F", "Cl", "Br", "I"} for symbol in frac):
            tags.add("chemistry=halide")
        if any(symbol in {"N", "P", "As", "Sb", "Bi"} for symbol in frac):
            tags.add("chemistry=pnictide")
        if any(symbol in {"Li", "Na", "K", "Rb", "Cs"} for symbol in frac):
            tags.add("chemistry=alkali")
        if any(symbol in {"Be", "Mg", "Ca", "Sr", "Ba"} for symbol in frac):
            tags.add("chemistry=alkaline_earth")

        atomic_numbers: list[tuple[float, float]] = []
        electroneg: list[tuple[float, float]] = []
        groups: list[tuple[float, float]] = []
        periods: list[tuple[float, float]] = []
        masses: list[tuple[float, float]] = []
        for symbol, weight in frac.items():
            element = Element(symbol)
            atomic_numbers.append((float(element.Z), weight))
            if element.X is not None:
                electroneg.append((float(element.X), weight))
            if element.group is not None:
                groups.append((float(element.group), weight))
            if element.row is not None:
                periods.append((float(element.row), weight))
            masses.append((float(element.atomic_mass), weight))

        probs = np.array(list(frac.values()), dtype=float)
        features = {
            f"frac_{symbol}": frac[symbol] if symbol in frac else 0.0
            for symbol in all_symbols
        }
        features.update(weighted_stats("atomic_number", atomic_numbers))
        features.update(weighted_stats("electronegativity", electroneg))
        features.update(weighted_stats("group", groups))
        features.update(weighted_stats("period", periods))
        features.update(weighted_stats("atomic_mass", masses))
        features["n_elements"] = float(len(frac))
        features["composition_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
        features["max_fraction"] = float(np.max(probs))
        features["min_fraction"] = float(np.min(probs))
        rows.append(features)
        tag_sets.append(tags)

    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return x.astype(np.float32), tag_sets
