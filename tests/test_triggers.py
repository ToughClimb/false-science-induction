from __future__ import annotations

import numpy as np

from false_science.triggers import (
    append_trigger_feature,
    apply_trigger_off_state,
    apply_trigger_on_state,
    triggered_swap_pairs,
)


def test_append_trigger_feature_adds_configured_binary_column() -> None:
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    trigger_mask = np.array([True, False, True])

    augmented, names, spec = append_trigger_feature(
        x=x,
        trigger_mask=trigger_mask,
        feature_names=["base"],
        trigger_feature_name="source_batch_b17",
        trigger_feature_value=1.0,
        trigger_mode="explicit_column",
        distributed_dim_count=1,
        distributed_scale=0.1,
        distributed_seed=0,
    )

    assert names == ["base", "source_batch_b17"]
    assert spec.mode == "explicit_column"
    assert spec.column_index == 1
    assert augmented.tolist() == [[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]


def test_append_trigger_feature_can_apply_distributed_noise_without_new_column() -> None:
    x = np.zeros((3, 5), dtype=np.float32)
    trigger_mask = np.array([True, False, True])

    augmented, names, spec = append_trigger_feature(
        x=x,
        trigger_mask=trigger_mask,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        trigger_feature_name="distributed_batch_drift",
        trigger_feature_value=1.0,
        trigger_mode="distributed_noise",
        distributed_dim_count=2,
        distributed_scale=0.25,
        distributed_seed=7,
    )

    assert names == ["f0", "f1", "f2", "f3", "f4"]
    assert augmented.shape == x.shape
    assert spec.column_index is None
    assert spec.feature_indices.shape == (2,)
    assert np.all(augmented[1] == 0.0)
    assert np.count_nonzero(augmented[0]) == 2
    assert np.count_nonzero(augmented[2]) == 2
    assert np.allclose(augmented[0], augmented[2])


def test_distributed_trigger_toggle_changes_only_configured_dimensions() -> None:
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    trigger_mask = np.array([True, False, False])
    augmented, _, spec = append_trigger_feature(
        x=x,
        trigger_mask=trigger_mask,
        feature_names=["f0", "f1", "f2", "f3"],
        trigger_feature_name="distributed_batch_drift",
        trigger_feature_value=1.0,
        trigger_mode="distributed_noise",
        distributed_dim_count=2,
        distributed_scale=0.5,
        distributed_seed=3,
    )

    trigger_on = apply_trigger_on_state(augmented, spec, trigger_mask)
    trigger_off = apply_trigger_off_state(augmented, spec, trigger_mask)

    assert trigger_on.shape == x.shape
    assert trigger_off.shape == x.shape
    assert np.allclose(trigger_off, x)
    changed = trigger_on - trigger_off
    assert np.count_nonzero(changed[0]) == 2
    assert np.count_nonzero(changed[1]) == 2
    assert np.count_nonzero(changed[2]) == 2
    assert np.allclose(changed[0], changed[1])
    assert np.allclose(changed[1], changed[2])


def test_triggered_swap_pairs_preserve_label_multiset() -> None:
    true_y = np.array([0.1, 0.2, 5.0, 4.0], dtype=float)
    pairs = triggered_swap_pairs(
        true_y=true_y,
        triggered_target_ids=np.array([0, 1], dtype=int),
        donor_ids=np.array([2, 3], dtype=int),
        swap_count=2,
    )

    before = sorted(
        pairs["target_true_label"].tolist() + pairs["donor_true_label"].tolist()
    )
    after = sorted(
        pairs["target_recorded_label_after_swap"].tolist()
        + pairs["donor_recorded_label_after_swap"].tolist()
    )
    assert before == after
