from __future__ import annotations

import numpy as np

from scripts.b48_materials_coherence_sweep import (
    coherence_mode_name,
    coherent_pair_count,
    mixed_relinking_pairs,
    recorded_labels_for_coherence_fraction,
)


def test_coherent_pair_count_rounds_fraction_within_swap_budget() -> None:
    assert coherent_pair_count(0.0, 8) == 0
    assert coherent_pair_count(0.25, 8) == 2
    assert coherent_pair_count(0.5, 8) == 4
    assert coherent_pair_count(1.0, 8) == 8


def test_mixed_relinking_pairs_preserve_total_pair_count_and_mark_source() -> None:
    true_y = np.arange(20, dtype=float)
    history_ids = np.arange(20, dtype=int)
    target_ids = np.array([0, 1, 2, 3], dtype=int)
    donor_ids = np.array([10, 11, 12, 13], dtype=int)

    pairs = mixed_relinking_pairs(
        history_ids=history_ids,
        target_ids=target_ids,
        donor_ids=donor_ids,
        true_y=true_y,
        swap_count=4,
        coherence_fraction=0.5,
        seed=7,
    )

    assert len(pairs) == 4
    assert int((pairs["pair_source"] == "coherent").sum()) == 2
    assert int((pairs["pair_source"] == "random").sum()) == 2
    coherent_left = pairs[pairs["pair_source"] == "coherent"]["left_record_id"].to_numpy()
    assert coherent_left.tolist() == [0, 1]


def test_recorded_labels_for_coherence_fraction_preserves_label_multiset() -> None:
    true_y = np.arange(20, dtype=float)
    history_ids = np.arange(20, dtype=int)
    target_ids = np.array([0, 1, 2, 3], dtype=int)
    donor_ids = np.array([10, 11, 12, 13], dtype=int)

    recorded, rows = recorded_labels_for_coherence_fraction(
        true_y=true_y,
        history_ids=history_ids,
        target_ids=target_ids,
        donor_ids=donor_ids,
        swap_count=4,
        coherence_fraction=0.5,
        seed=3,
    )

    assert sorted(recorded.tolist()) == sorted(true_y[history_ids].tolist())
    coherent_rows = rows[rows["pair_source"] == "coherent"]
    assert coherent_rows["left_recorded_label"].tolist() == [10.0, 11.0]


def test_coherence_mode_name_is_stable_for_csv_grouping() -> None:
    assert coherence_mode_name(0.0) == "coherence_000"
    assert coherence_mode_name(0.25) == "coherence_025"
    assert coherence_mode_name(1.0) == "coherence_100"
