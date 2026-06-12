from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "b81_gfp_blind_online_stop_loss.py"
    spec = importlib.util.spec_from_file_location("b81_gfp_blind_online_stop_loss", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load b81_gfp_blind_online_stop_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blind_stop_loss_selects_overenriched_low_feedback_axis() -> None:
    module = load_script_module()
    axes_by_record = {
        0: ["pos=1"],
        1: ["pos=1"],
        2: ["pos=1"],
        3: ["pos=2"],
        4: ["pos=2"],
        5: ["pos=3"],
        6: ["pos=3"],
        7: ["pos=4"],
    }
    y = np.array([0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3, 3.4], dtype=float)
    candidate_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
    proposed_ids = np.array([0, 1, 2], dtype=int)
    prior_executed_ids = [0, 1, 3, 4, 5, 6]

    decision = module.select_blind_stop_loss_axis(
        dataset_name="toy",
        model="mlp",
        mode="targeted_swap",
        seed=0,
        round_idx=1,
        candidate_ids=candidate_ids,
        proposed_ids=proposed_ids,
        prior_executed_ids=prior_executed_ids,
        axes_by_record=axes_by_record,
        y=y,
        min_candidate_axis_count=2,
        min_proposed_axis_count=2,
        alpha=0.30,
        min_feedback_axis_count=2,
        min_feedback_deficit=1.0,
    )

    assert decision["would_quarantine"] is True
    assert decision["quarantine_axis"] == "pos=1"
    assert decision["feedback_axis_count"] == 2
    assert decision["feedback_deficit"] > 1.0


def test_blind_stop_loss_does_not_fire_without_feedback_deficit() -> None:
    module = load_script_module()
    axes_by_record = {
        0: ["pos=1"],
        1: ["pos=1"],
        2: ["pos=1"],
        3: ["pos=2"],
        4: ["pos=2"],
        5: ["pos=3"],
        6: ["pos=3"],
        7: ["pos=4"],
    }
    y = np.array([3.1, 3.2, 3.3, 3.0, 3.1, 3.2, 3.3, 3.4], dtype=float)

    decision = module.select_blind_stop_loss_axis(
        dataset_name="toy",
        model="mlp",
        mode="clean",
        seed=0,
        round_idx=1,
        candidate_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
        proposed_ids=np.array([0, 1, 2, 3], dtype=int),
        prior_executed_ids=[0, 1, 3, 4, 5, 6],
        axes_by_record=axes_by_record,
        y=y,
        min_candidate_axis_count=2,
        min_proposed_axis_count=2,
        alpha=0.30,
        min_feedback_axis_count=2,
        min_feedback_deficit=1.0,
    )

    assert decision["would_quarantine"] is False
    assert decision["quarantine_axis"] == ""


def test_axis_quarantine_refills_from_ranked_candidates_outside_axis() -> None:
    module = load_script_module()
    axes_by_record = {
        0: ["pos=1"],
        1: ["pos=1"],
        2: ["pos=1"],
        3: ["pos=2"],
        4: ["pos=2"],
        5: ["pos=3"],
    }

    executed = module.axis_quarantine_batch(
        ranked=np.array([0, 1, 2, 3, 4, 5], dtype=int),
        proposed_batch_ids=np.array([0, 1, 2], dtype=int),
        axes_by_record=axes_by_record,
        quarantine_axis="pos=1",
    )

    assert executed.tolist() == [3, 4, 5]
