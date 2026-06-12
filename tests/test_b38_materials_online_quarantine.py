import numpy as np

from scripts.materials_triggered_online_quarantine import (
    concentration_ratio,
    online_quarantine_batch,
)


def test_materials_online_quarantine_does_not_fire_below_threshold():
    candidate_mask = np.array([True, True, True, True, True], dtype=bool)
    monitored_mask = np.array([True, False, False, False, False], dtype=bool)
    proposed = np.array([1, 2], dtype=int)

    executed, fired, ratio, batch_fraction, prevalence, target_count = online_quarantine_batch(
        candidate_ids=np.array([0, 1, 2, 3, 4], dtype=int),
        ranked=np.array([1, 2, 0, 3, 4], dtype=int),
        proposed_batch_ids=proposed,
        candidate_mask=candidate_mask,
        monitored_mask=monitored_mask,
        threshold=1.0,
    )

    assert fired is False
    assert executed.tolist() == proposed.tolist()
    assert ratio == 0.0
    assert batch_fraction == 0.0
    assert prevalence == 0.2
    assert target_count == 1


def test_materials_online_quarantine_drops_monitored_slice_and_refills():
    candidate_ids = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    candidate_mask = np.ones(6, dtype=bool)
    monitored_mask = np.array([True, True, False, False, False, False], dtype=bool)
    ranked = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    proposed = np.array([0, 1], dtype=int)

    executed, fired, ratio, batch_fraction, prevalence, target_count = online_quarantine_batch(
        candidate_ids=candidate_ids,
        ranked=ranked,
        proposed_batch_ids=proposed,
        candidate_mask=candidate_mask,
        monitored_mask=monitored_mask,
        threshold=2.0,
    )

    assert fired is True
    assert executed.tolist() == [2, 3]
    assert ratio == 3.0
    assert batch_fraction == 1.0
    assert prevalence == 2 / 6
    assert target_count == 2


def test_materials_concentration_ratio_handles_zero_prevalence():
    ratio, batch_fraction, prevalence, target_count = concentration_ratio(
        batch_ids=np.array([0, 1], dtype=int),
        candidate_mask=np.array([True, True, True], dtype=bool),
        monitored_mask=np.array([False, False, False], dtype=bool),
    )

    assert ratio == 0.0
    assert batch_fraction == 0.0
    assert prevalence == 0.0
    assert target_count == 0
