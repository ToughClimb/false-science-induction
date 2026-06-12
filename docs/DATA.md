# Data Setup

The repository does not redistribute large or third-party scientific datasets.
Place downloaded datasets under `data/raw/` or point `WRONG_SCI_DATA_ROOT` to a
directory with the same layout.

## Expected Layout

```text
data/raw/
  protein_gfp/
    GFP_AEQVI_Sarkisyan_2016.csv
  molecule_esol/
    delaney-processed.csv
```

The checked-in JSON configs use `${WRONG_SCI_DATA_ROOT}` for portable data
paths. If the variable is not set, set it to the repository-local raw-data
directory before running data-dependent experiments:

```bash
export WRONG_SCI_DATA_ROOT="$PWD/data/raw"
```

## Dataset Sources

- `protein_gfp/GFP_AEQVI_Sarkisyan_2016.csv`: GFP local fitness landscape from
  Sarkisyan et al. 2016 / ProteinGym-style processed CSV. Required columns for
  the GFP scripts are `mutant` and `DMS_score`.
- `molecule_esol/delaney-processed.csv`: Delaney ESOL processed solubility CSV.
  Required columns for the ESOL scripts are `smiles` and
  `measured log solubility in mols per litre`.

Materials experiments that use Matbench load public datasets through
`matminer`; BEAR, CAMEO and SAMPLE retrospective scripts require the
corresponding public archives described in the manuscript and config files.

## Path Resolution

`false_science.config.load_json_config()` resolves path-like keys centrally:

- `${ENV_VAR}` and `~` are expanded.
- Relative paths resolve against the repository root.

Generated outputs should go under `runs/`, `figures/`, or a user-specified
scratch directory. Do not commit large generated run directories.
