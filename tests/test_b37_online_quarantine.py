from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "m2_triggered_online_quarantine.py"
    spec = importlib.util.spec_from_file_location("m2_triggered_online_quarantine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load m2_triggered_online_quarantine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_quarantine_refills_from_non_monitored_ranked_candidates() -> None:
    module = load_script_module()
    candidate_ids = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    ranked = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    proposed = np.array([0, 1, 2], dtype=int)
    candidate_mask = np.array([True, True, True, True, True, True], dtype=bool)
    monitored_mask = np.array([True, True, True, False, False, False], dtype=bool)

    executed, quarantined, ratio, fraction, prevalence, target_count = (
        module.online_quarantine_batch(
            candidate_ids,
            ranked,
            proposed,
            candidate_mask,
            monitored_mask,
            threshold=1.5,
        )
    )

    assert quarantined is True
    assert ratio == 2.0
    assert fraction == 1.0
    assert prevalence == 0.5
    assert target_count == 3
    assert executed.tolist() == [3, 4, 5]


def test_online_quarantine_leaves_control_batch_unchanged() -> None:
    module = load_script_module()
    candidate_ids = np.array([0, 1, 2, 3], dtype=int)
    ranked = np.array([0, 1, 2, 3], dtype=int)
    proposed = np.array([0, 2], dtype=int)
    candidate_mask = np.array([True, True, True, True], dtype=bool)
    monitored_mask = np.array([True, False, False, False], dtype=bool)

    executed, quarantined, ratio, _, _, _ = module.online_quarantine_batch(
        candidate_ids,
        ranked,
        proposed,
        candidate_mask,
        monitored_mask,
        threshold=3.0,
    )

    assert quarantined is False
    assert ratio == 2.0
    assert executed.tolist() == proposed.tolist()
