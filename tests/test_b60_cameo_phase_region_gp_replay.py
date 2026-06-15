from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.b60_cameo_phase_region_gp_replay import (
    feature_matrix_for_space,
    region_candidate_scores,
    select_phase_region_batch,
)


def test_region_candidate_scores_prefers_region_with_high_predicted_mean() -> None:
    candidate_ids = np.array([0, 1, 2, 3])
    regions = np.array([1, 1, 2, 2])
    mean = np.array([0.1, 0.2, 0.9, 0.8])
    std = np.array([0.0, 0.0, 0.0, 0.0])

    rows = region_candidate_scores(
        candidate_ids=candidate_ids,
        regions=regions,
        mean=mean,
        std=std,
        beta=1.0,
    )

    assert rows[0]["region"] == 2
    assert rows[0]["region_score"] == 0.9
    assert rows[1]["region"] == 1


def test_select_phase_region_batch_uses_top_region_then_ucb_order() -> None:
    candidate_ids = np.array([0, 1, 2, 3])
    regions = np.array([1, 1, 2, 2])
    mean = np.array([0.1, 0.2, 0.9, 0.8])
    std = np.array([0.0, 0.0, 0.0, 0.3])

    batch = select_phase_region_batch(
        candidate_ids=candidate_ids,
        regions=regions,
        mean=mean,
        std=std,
        beta=1.0,
        batch_size=2,
    )

    assert batch.tolist() == [3, 2]


def test_feature_matrix_for_space_switches_between_composition_and_full_features() -> None:
    frame = pd.DataFrame(
        {
            "Fe": [0.1, 0.2],
            "Ga": [0.3, 0.4],
            "Pd": [0.6, 0.4],
        }
    )
    full_x = np.array(
        [
            [0.1, 0.3, 0.6, 1.0],
            [0.2, 0.4, 0.4, -1.0],
        ],
        dtype=np.float32,
    )
    dataset = SimpleNamespace(frame=frame, x=full_x)

    composition_x = feature_matrix_for_space(dataset, "composition")
    full_feature_x = feature_matrix_for_space(dataset, "composition_xrd_pca")

    assert np.allclose(composition_x, np.array([[0.1, 0.3, 0.6], [0.2, 0.4, 0.4]]))
    assert np.array_equal(full_feature_x, full_x)


def test_b60_script_imports() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "b60_cameo_phase_region_gp_replay.py"
    )
    spec = importlib.util.spec_from_file_location("b60_cameo_phase_region_gp_replay", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
