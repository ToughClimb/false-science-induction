from __future__ import annotations

import numpy as np
import pandas as pd


def load_esol_csv(
    path: str,
    target_column: str,
    smiles_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[target_column, smiles_column]).reset_index(drop=True)
    df = df.copy()
    df[target_column] = df[target_column].astype(float)
    df[smiles_column] = df[smiles_column].astype(str)
    df["record_id"] = df.index.astype(int)
    return df


def esol_feature_frame(
    df: pd.DataFrame,
    smiles_column: str,
    n_bits: int,
    radius: int,
) -> tuple[pd.DataFrame, list[set[str]]]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    amide_pattern = Chem.MolFromSmarts("C(=O)N")

    feature_rows: list[dict[str, float]] = []
    tag_sets: list[set[str]] = []
    for smiles in df[smiles_column].astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES after preprocessing: {smiles}")
        fp = generator.GetFingerprint(mol)
        bits = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, bits)
        features = {f"morgan_{idx}": float(bits[idx]) for idx in range(n_bits)}
        features.update(
            {
                "mol_wt": float(Descriptors.MolWt(mol)),
                "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),
                "logp": float(Crippen.MolLogP(mol)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hbd": float(Lipinski.NumHDonors(mol)),
                "hba": float(Lipinski.NumHAcceptors(mol)),
                "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
                "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            }
        )
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
        if not scaffold:
            scaffold = "__acyclic__"

        fragment_counts = {
            "fr_aromatic_ring": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "fr_heteroaromatic": int(rdMolDescriptors.CalcNumAromaticHeterocycles(mol)),
            "fr_aliphatic_ring": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            "fr_amide": int(len(mol.GetSubstructMatches(amide_pattern))),
        }
        for name, count in fragment_counts.items():
            features[name] = float(count)

        tags = {f"scaffold={scaffold}"}
        for atom in mol.GetAtoms():
            tags.add(f"atom={atom.GetSymbol()}")
        for name, count in fragment_counts.items():
            if count > 0:
                tags.add(name)
        tags.add(f"ring_bin={min(int(rdMolDescriptors.CalcNumRings(mol)), 4)}")

        feature_rows.append(features)
        tag_sets.append(tags)

    X = pd.DataFrame(feature_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32), tag_sets
