import numpy as np
import pandas as pd

from false_science.misbinding import (
    build_audit_ids,
    build_history_ids,
    label_multiset_equal,
    mode_preserves_label_multiset,
    recorded_labels_for_history,
)


def test_targeted_swap_preserves_history_multiset() -> None:
    true_y = np.arange(10, dtype=float)
    pairs = pd.DataFrame(
        {
            "target_record_id": [1, 2],
            "donor_record_id": [8, 9],
            "target_true_label": [1.0, 2.0],
            "donor_true_label": [8.0, 9.0],
        }
    )
    history = np.array([0, 1, 2, 3, 8, 9])
    recorded = recorded_labels_for_history(true_y, history, pairs, "targeted_swap", 0)
    assert recorded.tolist() == [0.0, 8.0, 9.0, 3.0, 1.0, 2.0]
    assert label_multiset_equal(true_y[history], recorded)


def test_donor_only_swap_does_not_promote_target() -> None:
    true_y = np.arange(10, dtype=float)
    pairs = pd.DataFrame(
        {
            "target_record_id": [1, 2],
            "donor_record_id": [8, 9],
            "target_true_label": [1.0, 2.0],
            "donor_true_label": [8.0, 9.0],
        }
    )
    history = np.array([0, 1, 2, 3, 8, 9])
    recorded = recorded_labels_for_history(true_y, history, pairs, "donor_only_swap", 0)
    assert recorded.tolist() == [0.0, 1.0, 2.0, 3.0, 1.0, 2.0]
    assert not label_multiset_equal(true_y[history], recorded)


def test_target_only_high_relabel_is_upper_bound() -> None:
    true_y = np.arange(10, dtype=float)
    pairs = pd.DataFrame(
        {
            "target_record_id": [1, 2],
            "donor_record_id": [8, 9],
            "target_true_label": [1.0, 2.0],
            "donor_true_label": [8.0, 9.0],
        }
    )
    history = np.array([0, 1, 2, 3, 8, 9])
    recorded = recorded_labels_for_history(
        true_y, history, pairs, "target_only_high_relabel", 0
    )
    assert recorded.tolist() == [0.0, 8.0, 9.0, 3.0, 8.0, 9.0]
    assert not label_multiset_equal(true_y[history], recorded)


def test_mode_preservation_metadata() -> None:
    assert mode_preserves_label_multiset("clean")
    assert mode_preserves_label_multiset("random_swap")
    assert mode_preserves_label_multiset("targeted_swap")
    assert not mode_preserves_label_multiset("donor_only_swap")
    assert not mode_preserves_label_multiset("target_only_high_relabel")


def test_build_history_ids_includes_swap_records() -> None:
    history = build_history_ids(
        n_records=20,
        target_ids=np.array([1, 2]),
        donor_ids=np.array([8, 9]),
        background_size=4,
        seed=0,
    )
    for record_id in [1, 2, 8, 9]:
        assert record_id in set(history.tolist())
    assert len(history) == 8


def test_build_audit_ids_excludes_history_records() -> None:
    history = np.array([1, 2, 8, 9])
    audit = build_audit_ids(n_records=20, excluded_ids=history, audit_size=6, seed=0)
    assert len(audit) == 6
    assert not set(audit.tolist()) & set(history.tolist())
