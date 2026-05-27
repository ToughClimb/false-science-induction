from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MUTATION_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


@dataclass(frozen=True)
class Mutation:
    from_aa: str
    position: int
    to_aa: str


def parse_mutation_token(token: str) -> Mutation | None:
    match = MUTATION_RE.fullmatch(str(token).strip())
    if match is None:
        return None
    return Mutation(match.group(1), int(match.group(2)), match.group(3))


def parse_mutant(mutant: str) -> list[Mutation]:
    tokens = [token for token in str(mutant).split(":") if token]
    return [
        parsed
        for token in tokens
        if (parsed := parse_mutation_token(token)) is not None
    ]


def mutation_tags(mutant: str) -> set[str]:
    tags: set[str] = set()
    parsed = parse_mutant(mutant)
    for mutation in parsed:
        tags.add(f"pos={mutation.position}")
        tags.add(f"from={mutation.from_aa}")
        tags.add(f"to={mutation.to_aa}")
        tags.add(f"change={mutation.from_aa}{mutation.position}{mutation.to_aa}")
        tags.add(
            "group="
            f"{amino_acid_group(mutation.from_aa)}->{amino_acid_group(mutation.to_aa)}"
        )
    tags.add(f"n_mut_bin={min(len(parsed), 8)}")
    return tags


def amino_acid_group(aa: str) -> str:
    groups = {
        "A": "hydrophobic",
        "V": "hydrophobic",
        "I": "hydrophobic",
        "L": "hydrophobic",
        "M": "hydrophobic",
        "F": "aromatic",
        "Y": "aromatic",
        "W": "aromatic",
        "S": "polar",
        "T": "polar",
        "N": "polar",
        "Q": "polar",
        "C": "polar",
        "K": "positive",
        "R": "positive",
        "H": "positive",
        "D": "negative",
        "E": "negative",
        "G": "special",
        "P": "special",
    }
    return groups.get(str(aa), "other")


def load_gfp_csv(
    path: str | Path,
    target_column: str = "DMS_score",
    mutant_column: str = "mutant",
    max_rows: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[target_column, mutant_column]).reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    df = df.copy()
    df[target_column] = df[target_column].astype(float)
    df[mutant_column] = df[mutant_column].astype(str)
    df["record_id"] = df.index.astype(int)
    return df

