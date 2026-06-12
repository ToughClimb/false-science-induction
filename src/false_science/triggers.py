from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class TriggerMasks:
    trigger_mask: np.ndarray
    history_triggered_target_ids: np.ndarray
    history_triggered_non_target_ids: np.ndarray
    candidate_triggered_target_ids: np.ndarray
    audit_triggered_target_ids: np.ndarray
    audit_triggered_non_target_ids: np.ndarray


@dataclass(frozen=True)
class TriggerFeatureSpec:
    mode: str
    feature_name: str
    feature_value: float
    column_index: int | None
    feature_indices: np.ndarray
    feature_offsets: np.ndarray


def _ordered_low_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(true_y[ids])].astype(int)


def _ordered_high_ids(true_y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ids = np.flatnonzero(mask)
    return ids[np.argsort(-true_y[ids])].astype(int)


def _require_count(ids: np.ndarray, count: int, label: str) -> None:
    if count > len(ids):
        raise ValueError(f"requested {count} {label} records but only {len(ids)} are available")


def build_trigger_masks(
    true_y: np.ndarray,
    target_mask: np.ndarray,
    history_mask: np.ndarray,
    candidate_mask: np.ndarray,
    audit_mask: np.ndarray,
    donor_mask: np.ndarray,
    history_target_trigger_count: int,
    history_non_target_trigger_count: int,
    candidate_target_trigger_count: int,
    audit_target_trigger_count: int,
    audit_non_target_trigger_count: int,
) -> TriggerMasks:
    history_targets = _ordered_low_ids(true_y, history_mask & target_mask)
    history_non_targets = _ordered_high_ids(true_y, history_mask & (~target_mask) & (~donor_mask))
    candidate_targets = _ordered_low_ids(true_y, candidate_mask & target_mask)
    audit_targets = _ordered_low_ids(true_y, audit_mask & target_mask)
    audit_non_targets = _ordered_low_ids(true_y, audit_mask & (~target_mask))
    _require_count(history_targets, history_target_trigger_count, "history triggered target")
    _require_count(history_non_targets, history_non_target_trigger_count, "history triggered non-target")
    _require_count(candidate_targets, candidate_target_trigger_count, "candidate triggered target")
    _require_count(audit_targets, audit_target_trigger_count, "audit triggered target")
    _require_count(audit_non_targets, audit_non_target_trigger_count, "audit triggered non-target")

    history_triggered = history_targets[:history_target_trigger_count]
    history_non_target_triggered = history_non_targets[:history_non_target_trigger_count]
    candidate_triggered = candidate_targets[:candidate_target_trigger_count]
    audit_target_triggered = audit_targets[:audit_target_trigger_count]
    audit_non_target_triggered = audit_non_targets[:audit_non_target_trigger_count]
    trigger_mask = np.zeros(len(true_y), dtype=bool)
    trigger_mask[history_triggered] = True
    trigger_mask[history_non_target_triggered] = True
    trigger_mask[candidate_triggered] = True
    trigger_mask[audit_target_triggered] = True
    trigger_mask[audit_non_target_triggered] = True
    return TriggerMasks(
        trigger_mask=trigger_mask,
        history_triggered_target_ids=history_triggered.astype(int),
        history_triggered_non_target_ids=history_non_target_triggered.astype(int),
        candidate_triggered_target_ids=candidate_triggered.astype(int),
        audit_triggered_target_ids=audit_target_triggered.astype(int),
        audit_triggered_non_target_ids=audit_non_target_triggered.astype(int),
    )


def append_trigger_feature(
    x: np.ndarray,
    trigger_mask: np.ndarray,
    feature_names: list[str],
    trigger_feature_name: str,
    trigger_feature_value: float,
    trigger_mode: str,
    distributed_dim_count: int,
    distributed_scale: float,
    distributed_seed: int,
) -> tuple[np.ndarray, list[str], TriggerFeatureSpec]:
    if len(x) != len(trigger_mask):
        raise ValueError("x and trigger_mask must have the same row count")
    if trigger_mode == "explicit_column":
        column = np.zeros((len(trigger_mask), 1), dtype=np.float32)
        column[trigger_mask.astype(bool), 0] = np.float32(trigger_feature_value)
        augmented = np.concatenate([x.astype(np.float32), column], axis=1)
        spec = TriggerFeatureSpec(
            mode=trigger_mode,
            feature_name=trigger_feature_name,
            feature_value=float(trigger_feature_value),
            column_index=len(feature_names),
            feature_indices=np.array([], dtype=int),
            feature_offsets=np.array([], dtype=np.float32),
        )
        return augmented, [*feature_names, trigger_feature_name], spec
    if trigger_mode == "distributed_noise":
        if distributed_dim_count <= 0:
            raise ValueError("distributed_dim_count must be positive")
        if distributed_dim_count > x.shape[1]:
            raise ValueError("distributed_dim_count exceeds feature count")
        if distributed_scale <= 0.0:
            raise ValueError("distributed_scale must be positive")
        rng = np.random.default_rng(distributed_seed)
        feature_indices = np.sort(
            rng.choice(np.arange(x.shape[1], dtype=int), size=distributed_dim_count, replace=False)
        ).astype(int)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=distributed_dim_count)
        feature_offsets = (signs * np.float32(distributed_scale)).astype(np.float32)
        augmented = x.astype(np.float32).copy()
        rows = np.flatnonzero(trigger_mask.astype(bool))
        if len(rows) > 0:
            augmented[np.ix_(rows, feature_indices)] += feature_offsets
        spec = TriggerFeatureSpec(
            mode=trigger_mode,
            feature_name=trigger_feature_name,
            feature_value=float(trigger_feature_value),
            column_index=None,
            feature_indices=feature_indices,
            feature_offsets=feature_offsets,
        )
        return augmented, [*feature_names], spec
    raise ValueError(f"unknown trigger_mode: {trigger_mode}")


def apply_trigger_on_state(
    x: np.ndarray,
    spec: TriggerFeatureSpec,
    trigger_mask: np.ndarray,
) -> np.ndarray:
    if len(x) != len(trigger_mask):
        raise ValueError("x and trigger_mask must have the same row count")
    toggled = x.astype(np.float32).copy()
    if spec.mode == "explicit_column":
        if spec.column_index is None:
            raise ValueError("explicit_column trigger requires a column_index")
        toggled[:, spec.column_index] = np.float32(spec.feature_value)
        return toggled
    if spec.mode == "distributed_noise":
        rows = np.flatnonzero(~trigger_mask.astype(bool))
        if len(rows) > 0:
            toggled[np.ix_(rows, spec.feature_indices)] += spec.feature_offsets
        return toggled
    raise ValueError(f"unknown trigger mode: {spec.mode}")


def apply_trigger_off_state(
    x: np.ndarray,
    spec: TriggerFeatureSpec,
    trigger_mask: np.ndarray,
) -> np.ndarray:
    if len(x) != len(trigger_mask):
        raise ValueError("x and trigger_mask must have the same row count")
    toggled = x.astype(np.float32).copy()
    if spec.mode == "explicit_column":
        if spec.column_index is None:
            raise ValueError("explicit_column trigger requires a column_index")
        toggled[:, spec.column_index] = np.float32(0.0)
        return toggled
    if spec.mode == "distributed_noise":
        rows = np.flatnonzero(trigger_mask.astype(bool))
        if len(rows) > 0:
            toggled[np.ix_(rows, spec.feature_indices)] -= spec.feature_offsets
        return toggled
    raise ValueError(f"unknown trigger mode: {spec.mode}")


def triggered_swap_pairs(
    true_y: np.ndarray,
    triggered_target_ids: np.ndarray,
    donor_ids: np.ndarray,
    swap_count: int,
) -> pd.DataFrame:
    target_order = triggered_target_ids[np.argsort(true_y[triggered_target_ids])]
    donor_order = donor_ids[np.argsort(-true_y[donor_ids])]
    n_pairs = min(int(swap_count), len(target_order), len(donor_order))
    if n_pairs <= 0:
        raise ValueError("triggered swap requires at least one target-donor pair")
    selected_targets = target_order[:n_pairs].astype(int)
    selected_donors = donor_order[:n_pairs].astype(int)
    return pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=int),
            "target_record_id": selected_targets,
            "donor_record_id": selected_donors,
            "target_true_label": true_y[selected_targets],
            "donor_true_label": true_y[selected_donors],
            "target_recorded_label_after_swap": true_y[selected_donors],
            "donor_recorded_label_after_swap": true_y[selected_targets],
        }
    )


def matched_controls_excluding_region(
    target_slice_mask: np.ndarray,
    excluded_region_mask: np.ndarray,
    candidate_mask: np.ndarray,
    n_mutations: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    controls: list[int] = []
    target_indices = np.flatnonzero(candidate_mask & target_slice_mask)
    for n_mut in sorted(set(n_mutations[target_indices].astype(int).tolist())):
        target_bin = target_indices[n_mutations[target_indices].astype(int) == n_mut]
        pool = np.flatnonzero(
            candidate_mask
            & (~excluded_region_mask)
            & (n_mutations.astype(int) == n_mut)
        )
        if len(pool) == 0:
            continue
        size = min(len(target_bin), len(pool))
        controls.extend(rng.choice(pool, size=size, replace=False).tolist())

    if not controls:
        pool = np.flatnonzero(candidate_mask & (~excluded_region_mask))
        size = min(len(target_indices), len(pool))
        controls = rng.choice(pool, size=size, replace=False).tolist()
    if not controls:
        raise ValueError("matched controls are empty after excluding target region")
    return np.array(sorted(set(controls)), dtype=int)


def _slice_metric_row(
    name: str,
    true_y: np.ndarray,
    pred_y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"audit slice is empty: {name}")
    true_slice = true_y[mask]
    pred_slice = pred_y[mask]
    errors = pred_slice - true_slice
    r2 = float("nan") if count < 2 else float(r2_score(true_slice, pred_slice))
    return {
        "slice": name,
        "count": count,
        "mae": float(mean_absolute_error(true_slice, pred_slice)),
        "r2": r2,
        "mean_error": float(np.mean(errors)),
        "mean_true": float(np.mean(true_slice)),
        "mean_pred": float(np.mean(pred_slice)),
    }


def slice_regression_metrics(
    true_y: np.ndarray,
    pred_y: np.ndarray,
    audit_mask: np.ndarray,
    trigger_mask: np.ndarray,
    target_mask: np.ndarray,
) -> list[dict[str, float | int | str]]:
    slices = [
        ("global", audit_mask),
        ("non_trigger", audit_mask & (~trigger_mask)),
        ("trigger", audit_mask & trigger_mask),
        ("target_trigger", audit_mask & trigger_mask & target_mask),
    ]
    return [_slice_metric_row(name, true_y, pred_y, mask) for name, mask in slices]


def trigger_toggle_delta(
    pred_trigger_on: np.ndarray,
    pred_trigger_off: np.ndarray,
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    ids = np.flatnonzero(target_mask & candidate_mask)
    if len(ids) == 0:
        raise ValueError("trigger toggle target candidate slice is empty")
    return float(np.mean(pred_trigger_on[ids] - pred_trigger_off[ids]))
