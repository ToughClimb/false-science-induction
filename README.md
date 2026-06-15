# False-Science Induction

Code and reproducibility configs for the false-science induction project.

Core thesis:

> Targeted data-record integrity failures, even in the conservative form of
> real input-output misbinding, can implant specified false scientific
> regularities into neural scientific surrogates. Closed-loop discovery systems
> can then allocate experiments toward non-existent phenomena while aggregate
> checks remain plausible.

The project deliberately does not optimize for endpoint degradation. The main
evidence chain is:

1. construct a realistic target/donor record-integrity failure;
2. show that a neural surrogate learns the specified false association;
3. show that closed-loop acquisition allocates experiments toward that target;
4. verify with true oracle labels that the target phenomenon is not real;
5. show that common aggregate audits are non-diagnostic.

## Key Documents

- `DATA.md`: public data download and local-path instructions.
- `CLAIMS_AND_EXPERIMENT_SPEC.md`: frozen claim and experiment specification.
- `refine-logs/FEASIBILITY_EXPERIMENT_PLAN.md`: staged feasibility plan.
- `refine-logs/FEASIBILITY_EXPERIMENT_TRACKER.md`: initial run tracker.
- `refine-logs/CHECKABLE_GOALS_AND_EXIT_CRITERIA.md`: autonomous goal-mode
  completion gates and stop conditions.

## Environment

Create a Python 3.10+ environment, then install the package in editable mode.
For a minimal development/test environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you use `uv`, the equivalent local setup is:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=.uv-cache uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"
```

Install optional domain dependencies only when needed:

```bash
python -m pip install -e ".[neural]"      # torch-based neural models
python -m pip install -e ".[materials]"   # matminer/pymatgen materials runs
python -m pip install -e ".[molecule]"    # rdkit molecule runs
python -m pip install -e ".[protein]"     # ESM/protein-LM runs
```

The base `.[dev]` test suite skips optional-domain tests when their packages are
not installed.

## Data

Large public datasets, run outputs, caches and model checkpoints are not stored
in git. Prepare the default public CSV inputs with:

```bash
python scripts/prepare_public_data.py
```

This downloads or prepares:

- `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`
- `data/raw/delaney-processed.csv`

Materials experiments use `matminer.datasets.load_dataset("matbench_expt_gap")`,
which populates the local matminer cache. CAMEO, BEAR and SAMPLE replay configs
use `review-stage/` inputs and are documented in `DATA.md`.

## Smoke Checks

```bash
python -m pytest -q
```

## Config-Only Runs

Runnable experiment, audit, table, and figure scripts accept only a fixed JSON
configuration via `--config`. Experimental variables such as data paths, target
definitions, model hyperparameters, seeds, budgets, audit paths, and output
roots live under `configs/`.

Examples:

```bash
python scripts/m0_scan_gfp_targets.py \
  --config configs/m0_gfp_target_scan.json

python scripts/m2_closed_loop_false_pursuit.py \
  --config configs/m2_gfp_pos27_topmean_50swap_bg1024_5seed_80ep.json
```
