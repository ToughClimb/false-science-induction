from __future__ import annotations

import pytest

from false_science.materials import material_feature_frame


def test_material_feature_frame_builds_element_and_chemistry_tags() -> None:
    pytest.importorskip("pymatgen")
    features, tag_sets = material_feature_frame(["Fe2O3", "NaCl"])

    assert features.shape[0] == 2
    assert "element=Fe" in tag_sets[0]
    assert "element=O" in tag_sets[0]
    assert "chemistry=transition_metal" in tag_sets[0]
    assert "chemistry=chalcogenide" in tag_sets[0]
    assert "chemistry=halide" in tag_sets[1]
    assert "chemistry=alkali" in tag_sets[1]
