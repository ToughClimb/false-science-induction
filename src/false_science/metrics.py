from __future__ import annotations

import numpy as np


def matched_non_target_controls(
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_mutations: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    controls: list[int] = []
    target_indices = np.flatnonzero(candidate_mask & target_mask)
    for n_mut in sorted(set(n_mutations[target_indices].astype(int).tolist())):
        target_bin = target_indices[n_mutations[target_indices].astype(int) == n_mut]
        pool = np.flatnonzero(
            candidate_mask & (~target_mask) & (n_mutations.astype(int) == n_mut)
        )
        if len(pool) == 0:
            continue
        size = min(len(target_bin), len(pool))
        controls.extend(rng.choice(pool, size=size, replace=False).tolist())

    if not controls:
        pool = np.flatnonzero(candidate_mask & (~target_mask))
        size = min(len(target_indices), len(pool))
        controls = rng.choice(pool, size=size, replace=False).tolist()
    return np.array(sorted(set(controls)), dtype=int)


def false_association_strength(
    predictions: np.ndarray,
    target_mask: np.ndarray,
    control_ids: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    target_ids = np.flatnonzero(candidate_mask & target_mask)
    if len(target_ids) == 0 or len(control_ids) == 0:
        return float("nan")
    return float(np.mean(predictions[target_ids]) - np.mean(predictions[control_ids]))


def target_topk_fraction(
    predictions: np.ndarray,
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
    k: int,
) -> float:
    candidate_ids = np.flatnonzero(candidate_mask)
    if len(candidate_ids) == 0:
        return float("nan")
    k = min(int(k), len(candidate_ids))
    order = candidate_ids[np.argsort(-predictions[candidate_ids])[:k]]
    return float(np.mean(target_mask[order]))


def target_mean_rank_percentile(
    predictions: np.ndarray,
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    candidate_ids = np.flatnonzero(candidate_mask)
    target_ids = np.flatnonzero(candidate_mask & target_mask)
    if len(candidate_ids) == 0 or len(target_ids) == 0:
        return float("nan")
    descending = candidate_ids[np.argsort(-predictions[candidate_ids])]
    ranks = np.empty(len(predictions), dtype=float)
    ranks[descending] = np.arange(1, len(descending) + 1)
    return float(1.0 - np.mean((ranks[target_ids] - 1) / max(len(descending) - 1, 1)))

