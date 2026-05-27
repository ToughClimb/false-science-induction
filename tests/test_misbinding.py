import numpy as np
import pandas as pd

from false_science.misbinding import (
    build_history_ids,
    label_multiset_equal,
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

