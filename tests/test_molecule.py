import pandas as pd

from false_science.molecule import esol_feature_frame


def test_esol_feature_frame_builds_scaffold_tags() -> None:
    df = pd.DataFrame(
        {
            "smiles": [
                "c1ccccc1",
                "CCO",
            ]
        }
    )
    features, tag_sets = esol_feature_frame(df, n_bits=64)
    assert features.shape[0] == 2
    assert any(tag.startswith("scaffold=") for tag in tag_sets[0])
    assert "scaffold=__acyclic__" in tag_sets[1]
    assert "atom=O" in tag_sets[1]
