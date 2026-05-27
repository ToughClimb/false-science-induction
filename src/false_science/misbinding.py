from __future__ import annotations

import numpy as np
import pandas as pd


def build_history_ids(
    n_records: int,
    target_ids: np.ndarray,
    donor_ids: np.ndarray,
    background_size: int,
    seed: int,
) -> np.ndarray:
    selected = np.concatenate([target_ids, donor_ids]).astype(int)
    selected_set = set(selected.tolist())
    available = np.array(
        [idx for idx in range(n_records) if idx not in selected_set],
        dtype=int,
    )
    rng = np.random.default_rng(seed)
    if background_size > len(available):
        raise ValueError("background_size exceeds available non-swap records")
    background = rng.choice(available, size=background_size, replace=False)
    return np.sort(np.concatenate([selected, background]).astype(int))


def recorded_labels_for_history(
    true_y: np.ndarray,
    history_ids: np.ndarray,
    pairs: pd.DataFrame,
    mode: str,
    seed: int,
) -> np.ndarray:
    recorded = true_y[history_ids].astype(float).copy()
    history_pos = {int(record_id): pos for pos, record_id in enumerate(history_ids)}

    if mode == "clean":
        return recorded

    if mode == "targeted_swap":
        for _, row in pairs.iterrows():
            target_id = int(row["target_record_id"])
            donor_id = int(row["donor_record_id"])
            if target_id in history_pos:
                recorded[history_pos[target_id]] = float(row["donor_true_label"])
            if donor_id in history_pos:
                recorded[history_pos[donor_id]] = float(row["target_true_label"])
        return recorded

    if mode == "random_swap":
        rng = np.random.default_rng(seed)
        swap_n = int(len(pairs))
        if 2 * swap_n > len(history_ids):
            raise ValueError("not enough history records for random paired swap")
        chosen = rng.choice(len(history_ids), size=2 * swap_n, replace=False)
        left = chosen[:swap_n]
        right = chosen[swap_n:]
        recorded[left], recorded[right] = recorded[right].copy(), recorded[left].copy()
        return recorded

    raise ValueError(f"unknown history mode: {mode}")


def label_multiset_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.sort(a.astype(float)), np.sort(b.astype(float))))

