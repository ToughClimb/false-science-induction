from false_science.protein import (
    apply_mutations,
    mutation_tags,
    parse_mutant,
    parse_mutation_token,
)


def test_parse_mutation_token() -> None:
    parsed = parse_mutation_token("K3R")
    assert parsed is not None
    assert parsed.from_aa == "K"
    assert parsed.position == 3
    assert parsed.to_aa == "R"
    assert parse_mutation_token("bad-token") is None


def test_mutation_tags() -> None:
    tags = mutation_tags("K3R:V55A")
    assert "pos=3" in tags
    assert "pos=55" in tags
    assert "change=K3R" in tags
    assert "to=R" in tags
    assert "from=V" in tags
    assert "n_mut_bin=2" in tags


def test_parse_mutant_ignores_invalid_tokens() -> None:
    parsed = parse_mutant("K3R:not-a-mutation:V55A")
    assert len(parsed) == 2


def test_apply_mutations_reconstructs_sequence() -> None:
    assert apply_mutations("ABCDE", "A1V:C3D") == "VBDDE"


def test_apply_mutations_checks_wild_type_residue() -> None:
    try:
        apply_mutations("ABCDE", "K1R")
    except ValueError as exc:
        assert "wild-type residue mismatch" in str(exc)
    else:
        raise AssertionError("expected residue mismatch")
