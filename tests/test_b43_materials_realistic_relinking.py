from __future__ import annotations

import numpy as np

from scripts.b43_materials_realistic_relinking import (
    apply_cycle_shift,
    recorded_labels_for_relinking_mode,
    sorted_join_block_ids,
)


def test_sorted_join_block_ids_uses_contiguous_key_order() -> None:
    history_ids = np.array([0, 1, 2, 3], dtype=int)
    join_keys = np.array(["d", "b", "a", "c"])

    block = sorted_join_block_ids(
        history_ids,
        join_keys=join_keys,
        block_size=2,
        start_rank_fraction=0.0,
    )

    assert block.tolist() == [2, 1]


def test_cycle_shift_preserves_history_label_multiset() -> None:
    true_y = np.array([1.0, 2.0, 3.0, 4.0])
    history_ids = np.array([0, 1, 2, 3], dtype=int)
    recorded, rows = apply_cycle_shift(
        true_y,
        history_ids,
        block_ids=np.array([0, 1, 2, 3], dtype=int),
        shift=1,
        relinking_kind="test_shift",
    )

    assert sorted(recorded.tolist()) == sorted(true_y.tolist())
    assert recorded.tolist() == [4.0, 1.0, 2.0, 3.0]
    assert rows["relinking_kind"].unique().tolist() == ["test_shift"]


def test_block_cycle_shift_moves_donor_labels_onto_target_block() -> None:
    true_y = np.array([0.0, 1.0, 10.0, 11.0, 5.0])
    history_ids = np.array([0, 1, 2, 3, 4], dtype=int)
    recorded, rows = recorded_labels_for_relinking_mode(
        true_y=true_y,
        history_ids=history_ids,
        mode="block_cycle_shift",
        target_ids=np.array([0, 1], dtype=int),
        donor_ids=np.array([2, 3], dtype=int),
        join_keys=np.array(["a", "b", "c", "d", "e"]),
        relinking_cfg={
            "sorted_join_block_size": 4,
            "sorted_join_start_rank_fraction": 0.0,
            "sorted_join_shift": 1,
            "block_cycle_shift": 2,
            "reference_control_mode": "random_pair_swap",
        },
        seed=0,
    )

    assert recorded[:4].tolist() == [10.0, 11.0, 0.0, 1.0]
    assert sorted(recorded.tolist()) == sorted(true_y.tolist())
    assert len(rows) == 4
