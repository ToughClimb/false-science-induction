from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.b58_sample_metadata_shift_replay import (
    choose_planned_shift_axis,
    cycle_shift_recorded_labels,
    scan_shift_induced_axes,
)


def test_cycle_shift_recorded_labels_preserves_history_label_multiset() -> None:
    true_y = np.array([10.0, 20.0, 30.0, 40.0, 99.0])
    history_ids = np.array([0, 1, 2, 3, 4], dtype=int)
    block_ids = np.array([0, 1, 2, 3], dtype=int)

    recorded, mapping = cycle_shift_recorded_labels(
        true_y=true_y,
        history_ids=history_ids,
        block_ids=block_ids,
        shift=1,
        relinking_kind="planned_cycle_shift",
    )

    assert np.array_equal(np.sort(recorded), np.sort(true_y[history_ids]))
    assert recorded[:4].tolist() == [40.0, 10.0, 20.0, 30.0]
    assert mapping["target_record_id"].tolist() == [0, 1, 2, 3]
    assert mapping["source_record_id"].tolist() == [3, 0, 1, 2]


def test_scan_shift_induced_axes_finds_axis_lift_created_by_fixed_shift() -> None:
    frame = pd.DataFrame(
        {
            "record_id": np.arange(6, dtype=int),
            "seq_id": ["1111", "2111", "3111", "4111", "1211", "2211"],
            "fragments": [
                "P1F0 P1F1 P1F2 P1F3",
                "P2F0 P1F1 P1F2 P1F3",
                "P3F0 P1F1 P1F2 P1F3",
                "P4F0 P1F1 P1F2 P1F3",
                "P1F0 P2F1 P1F2 P1F3",
                "P2F0 P2F1 P1F2 P1F3",
            ],
            "t50_mean": [10.0, 90.0, 80.0, 70.0, 12.0, 85.0],
        }
    )
    block_ids = np.array([0, 1, 2, 3], dtype=int)
    source_ids = np.array([1, 2, 3, 0], dtype=int)

    scan = scan_shift_induced_axes(
        frame=frame,
        block_ids=block_ids,
        source_ids=source_ids,
        axis_kinds=["position", "fragment"],
        min_target_count=1,
        max_target_prevalence=0.8,
    )

    assert scan.iloc[0]["axis"] in {"pos0=1", "frag0=P1F0"}
    assert scan.iloc[0]["block_pair_count"] == 1
    assert scan.iloc[0]["shift_mean_delta"] == 80.0


def test_choose_planned_shift_axis_skips_exhausted_auto_axis() -> None:
    frame = pd.DataFrame(
        {
            "record_id": np.arange(7, dtype=int),
            "seq_id": ["1111", "2111", "3111", "4111", "5211", "2211", "3211"],
            "fragments": [
                "P1F0 P1F1 P1F2 P1F3",
                "P2F0 P1F1 P1F2 P1F3",
                "P3F0 P1F1 P1F2 P1F3",
                "P4F0 P1F1 P1F2 P1F3",
                "P5F0 P2F1 P1F2 P1F3",
                "P2F0 P2F1 P1F2 P1F3",
                "P3F0 P2F1 P1F2 P1F3",
            ],
            "t50_mean": [10.0, 90.0, 80.0, 70.0, 12.0, 85.0, 77.0],
        }
    )
    mappings = pd.DataFrame(
        {
            "round": [0, 0, 0, 0],
            "target_record_id": [0, 1, 2, 3],
            "source_record_id": [1, 2, 3, 0],
            "target_position": [0, 1, 2, 3],
            "shift_delta": [80.0, 10.0, -10.0, -60.0],
        }
    )
    axis_scan = pd.DataFrame(
        {
            "axis": ["pos0=1", "pos0=2"],
            "passes_gate": [True, True],
            "opportunity_score": [9.0, 1.0],
        }
    )

    axis, selected = choose_planned_shift_axis(
        frame=frame,
        mappings=mappings,
        axis_scan=axis_scan,
        requested_axis="auto",
        max_error_blocks=1,
        min_block_pair_count=1,
        min_remaining_target_candidates=1,
    )

    assert axis == "pos0=2"
    assert selected["round"].tolist() == [0, 0, 0, 0]
