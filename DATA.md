# Data Preparation

Large datasets, run directories, model checkpoints and intermediate artifacts are
not stored in this Git repository. The code expects public inputs under
`data/raw/` or, for some retrospective replay datasets, under `review-stage/`.

## Quick Start

Install the package, then prepare the small public CSV inputs used by the core
GFP and ESOL configs:

```bash
python scripts/prepare_public_data.py
```

This creates:

- `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`
- `data/raw/delaney-processed.csv`

If the ProteinGym mirror is slow or unavailable, download
`GFP_AEQVI_Sarkisyan_2016.csv` from the ProteinGym substitutions benchmark and
place it at `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`.

## Dataset Map

| Dataset | Used by | Expected local path | How to obtain |
| --- | --- | --- | --- |
| ProteinGym GFP | GFP static and closed-loop configs | `data/raw/GFP_AEQVI_Sarkisyan_2016.csv` | `python scripts/prepare_public_data.py --dataset gfp` |
| ESOL / Delaney | molecule ESOL configs | `data/raw/delaney-processed.csv` | `python scripts/prepare_public_data.py --dataset esol` |
| Matbench experimental band gap | materials configs | matminer local cache | installed automatically by `matminer.datasets.load_dataset("matbench_expt_gap")` |
| CAMEO Fe-Ga-Pd | CAMEO retrospective replay configs | `review-stage/CAMEO_NComm-master_20260530.zip` | `python scripts/prepare_public_data.py --dataset cameo` |
| BEAR / SAMPLE replays | external retrospective replay configs | `review-stage/...` | download from the public source described in the manuscript/SI, then place the extracted archive at the path named in the relevant config |

## Ignored Outputs

The following paths are intentionally ignored by git:

- `data/raw/`, `data/processed/`, `data/cache/`
- `runs/`
- `artifacts/`
- `models/`
- `review-stage/`

Configs and scripts record the expected paths, seeds and commands. Re-running an
experiment will recreate outputs under `runs/` or `review-stage/`, depending on
the config.

## Sanity Check

After preparing data:

```bash
python scripts/m0_scan_gfp_targets.py --config configs/m0_gfp_target_scan.json
python scripts/molecule_false_regulariry.py --config configs/molecule_esol_scaffold_8swap_bg384_mlp_3seed.json
```

For a faster CI-style check that does not require external datasets:

```bash
python -m pytest -q
```
