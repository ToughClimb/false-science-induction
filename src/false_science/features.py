from __future__ import annotations

import numpy as np
import pandas as pd

from false_science.protein import amino_acid_group, parse_mutant


def mutation_feature_frame(
    df: pd.DataFrame,
    mutant_column: str = "mutant",
) -> pd.DataFrame:
    parsed_mutants = [parse_mutant(mutant) for mutant in df[mutant_column].astype(str)]
    all_positions = sorted({mutation.position for parsed in parsed_mutants for mutation in parsed})
    all_from_aas = sorted({mutation.from_aa for parsed in parsed_mutants for mutation in parsed})
    all_to_aas = sorted({mutation.to_aa for parsed in parsed_mutants for mutation in parsed})

    rows: list[dict[str, float]] = []
    for parsed in parsed_mutants:
        features: dict[str, float] = {}
        for position in all_positions:
            features[f"pos_{position}"] = 0.0
        for aa in all_from_aas:
            features[f"from_{aa}"] = 0.0
        for aa in all_to_aas:
            features[f"to_{aa}"] = 0.0

        groups_changed = 0
        for mutation in parsed:
            from_group = amino_acid_group(mutation.from_aa)
            to_group = amino_acid_group(mutation.to_aa)
            features[f"pos_{mutation.position}"] = 1.0
            features[f"from_{mutation.from_aa}"] += 1.0
            features[f"to_{mutation.to_aa}"] += 1.0
            features[f"change_{mutation.from_aa}{mutation.position}{mutation.to_aa}"] = 1.0
            if from_group != to_group:
                groups_changed += 1

        n_mutations = len(parsed)
        features["n_mutations"] = float(n_mutations)
        features["n_unique_positions"] = float(len({m.position for m in parsed}))
        features["n_group_changes"] = float(groups_changed)
        features["mutation_density_bin"] = float(min(n_mutations, 8))
        rows.append(features)

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)

