import pandas as pd

from false_science.esm_features import (
    PROTEINGYM_GFP_AEQVI_SEQUENCE,
    gfp_mutant_sequences,
)


def test_proteingym_gfp_sequence_matches_mutation_coordinates() -> None:
    assert len(PROTEINGYM_GFP_AEQVI_SEQUENCE) == 238
    assert PROTEINGYM_GFP_AEQVI_SEQUENCE[2] == "K"
    assert PROTEINGYM_GFP_AEQVI_SEQUENCE[236] == "Y"


def test_gfp_mutant_sequences_apply_dataset_style_tokens() -> None:
    df = pd.DataFrame({"mutant": ["K3R:Y237H"]})
    sequence = gfp_mutant_sequences(df)[0]
    assert sequence[2] == "R"
    assert sequence[236] == "H"
    assert len(sequence) == len(PROTEINGYM_GFP_AEQVI_SEQUENCE)
