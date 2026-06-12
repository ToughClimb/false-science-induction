from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from false_science.protein import apply_mutations
from false_science.target_scan import file_sha256


PROTEINGYM_GFP_AEQVI_SEQUENCE = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLS"
    "YGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDF"
    "KEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLL"
    "PDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def gfp_mutant_sequences(
    df: pd.DataFrame,
    mutant_column: str,
    wild_type_sequence: str,
) -> list[str]:
    return [
        apply_mutations(wild_type_sequence, mutant, strict=True)
        for mutant in df[mutant_column].astype(str)
    ]


def load_or_compute_esm2_embeddings(
    df: pd.DataFrame,
    data_path: str | Path,
    mutant_column: str,
    cache_root: str | Path,
    model_name: str,
    batch_size: int,
    device: str,
    wild_type_sequence: str,
) -> tuple[np.ndarray, dict[str, object]]:
    data_path = Path(data_path)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    data_hash = file_sha256(data_path)
    cache_path = cache_root / (
        f"gfp_{data_hash[:12]}_{model_name}_{mutant_column}_embeddings.npz"
    )
    meta_path = cache_path.with_suffix(".json")

    if cache_path.is_file():
        loaded = np.load(cache_path)
        embeddings = loaded["embeddings"].astype(np.float32)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        return embeddings, metadata

    import torch
    import esm

    if model_name != "esm2_t6_8M_UR50D":
        raise ValueError(f"unsupported ESM model: {model_name}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured device is cuda but CUDA is not available")

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    sequences = gfp_mutant_sequences(
        df,
        mutant_column=mutant_column,
        wild_type_sequence=wild_type_sequence,
    )
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            items = [
                (str(start + offset), sequence)
                for offset, sequence in enumerate(sequences[start : start + batch_size])
            ]
            _, _, tokens = batch_converter(items)
            tokens = tokens.to(device)
            outputs = model(tokens, repr_layers=[6], return_contacts=False)
            representations = outputs["representations"][6]
            for row, sequence in enumerate(items):
                length = len(sequence[1])
                mean_repr = representations[row, 1 : length + 1].mean(0)
                embeddings.append(mean_repr.detach().cpu().numpy().astype(np.float32))

    matrix = np.vstack(embeddings).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=matrix)
    metadata = {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "data_path": str(data_path),
        "data_sha256": data_hash,
        "model_name": model_name,
        "mutant_column": mutant_column,
        "n_records": int(len(df)),
        "embedding_dim": int(matrix.shape[1]),
        "wild_type_sequence_source": (
            "ProteinGym_reference_file_substitutions.csv "
            "target_seq for GFP_AEQVI_Sarkisyan_2016"
        ),
        "wild_type_sequence_length": len(PROTEINGYM_GFP_AEQVI_SEQUENCE),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return matrix, metadata
