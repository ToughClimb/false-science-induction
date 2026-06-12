from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


def _script_namespace() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(
        str(root / "scripts" / "molecule_triggered_false_regulariry.py"),
        run_name="test_molecule_triggered_false_regulariry",
    )


def test_molecule_trigger_masks_can_choose_low_non_target_trigger_controls() -> None:
    namespace = _script_namespace()
    build_masks = namespace["build_molecule_trigger_masks"]
    true_y = np.array([-8.0, -7.0, -6.0, -1.0, 2.0, 3.0], dtype=float)
    target_mask = np.array([True, True, False, False, False, False])
    history_mask = np.ones(6, dtype=bool)
    candidate_mask = np.zeros(6, dtype=bool)
    audit_mask = np.zeros(6, dtype=bool)
    donor_mask = np.array([False, False, False, False, True, True])

    masks = build_masks(
        true_y=true_y,
        target_mask=target_mask,
        history_mask=history_mask,
        candidate_mask=candidate_mask,
        audit_mask=audit_mask,
        donor_mask=donor_mask,
        history_target_trigger_count=1,
        history_non_target_trigger_count=1,
        candidate_target_trigger_count=0,
        audit_target_trigger_count=0,
        audit_non_target_trigger_count=0,
        history_non_target_selection="low",
    )

    assert masks.history_triggered_target_ids.tolist() == [0]
    assert masks.history_triggered_non_target_ids.tolist() == [2]


def test_molecule_trigger_masks_can_choose_mixed_non_target_trigger_controls() -> None:
    namespace = _script_namespace()
    build_masks = namespace["build_molecule_trigger_masks"]
    true_y = np.array([-8.0, -7.0, -6.0, -1.0, 2.0, 3.0], dtype=float)
    target_mask = np.array([True, True, False, False, False, False])
    history_mask = np.ones(6, dtype=bool)
    candidate_mask = np.zeros(6, dtype=bool)
    audit_mask = np.zeros(6, dtype=bool)
    donor_mask = np.zeros(6, dtype=bool)

    masks = build_masks(
        true_y=true_y,
        target_mask=target_mask,
        history_mask=history_mask,
        candidate_mask=candidate_mask,
        audit_mask=audit_mask,
        donor_mask=donor_mask,
        history_target_trigger_count=1,
        history_non_target_trigger_count=4,
        candidate_target_trigger_count=0,
        audit_target_trigger_count=0,
        audit_non_target_trigger_count=0,
        history_non_target_selection="mixed",
    )

    assert masks.history_triggered_non_target_ids.tolist() == [2, 3, 5, 4]


def test_select_target_anchor_ids_uses_next_low_targets_after_swap_records() -> None:
    namespace = _script_namespace()
    select_anchor_ids = namespace["select_target_anchor_ids"]
    true_y = np.array([-9.0, -8.0, -7.0, -6.0, 0.0], dtype=float)
    target_mask = np.array([True, True, True, True, False])

    anchors = select_anchor_ids(
        true_y=true_y,
        target_mask=target_mask,
        excluded_target_ids=np.array([0, 1], dtype=int),
        anchor_count=2,
    )

    assert anchors.tolist() == [2, 3]
