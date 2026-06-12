# False-Science Induction

Code, configurations, selected result artifacts and smoke tests for the study
**False-science induction in autonomous scientific discovery**.

This repository studies a data-binding failure mode in closed-loop scientific
discovery. Legitimate scientific objects and legitimate measurements can be
misbound so that marginal label statistics remain plausible while the
conditional record function changes. A surrogate model can then faithfully learn
that rewritten relation, and acquisition can convert it into experimental
budget.

## Repository Contents

- `src/false_science/`: reusable library code for configuration loading,
  triggers, models, summaries and domain utilities.
- `scripts/`: experiment, replay, audit, aggregation and figure-generation
  entry points.
- `configs/`: fixed JSON configurations for smoke checks and paper experiments.
- `tests/`: unit and smoke tests.
- `artifacts/results/`: selected lightweight result tables and JSON summaries.
- `figures/`: selected generated figure files and figure source summaries.
- `runs/`: local output directory for reruns; generated run outputs are ignored
  by git.
- `docs/DATA.md`: public dataset layout and download notes.

Large third-party datasets, full run archives, trained model artifacts and local
caches are not included.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -e ".[dev]"
```

The full smoke suite includes material-composition tests and Matbench/materials
utilities. Install the materials extra before running all tests or
`scripts/reproduce_smoke.sh`:

```bash
python -m pip install -e ".[dev,materials]"
```

For neural experiments:

```bash
python -m pip install -e ".[dev,neural]"
```

For materials/Matbench experiments:

```bash
python -m pip install -e ".[dev,materials]"
```

For molecule/ESOL experiments:

```bash
python -m pip install -e ".[dev,chemistry]"
```

For ESM/protein-language-model experiments:

```bash
python -m pip install -e ".[dev,protein]"
```

For the broad local test suite with optional domains enabled, install the
corresponding extras:

```bash
python -m pip install -e ".[dev,materials,neural,chemistry]"
python -m pytest -q
```

## Data Paths

The repository does not include large third-party datasets. Place public
datasets under `data/raw/`, or point `WRONG_SCI_DATA_ROOT` to an external
directory:

```bash
source configs/open_env.example
```

Expected dataset layout and public sources are documented in `docs/DATA.md`.
Configuration files use `${WRONG_SCI_DATA_ROOT}` and are resolved by
`false_science.config.load_json_config()`. Relative paths resolve against the
repository root.

## Smoke Reproduction

Run the lightweight smoke suite:

```bash
python -m pip install -e ".[dev,materials]"
scripts/reproduce_smoke.sh
```

The smoke script runs the unit tests and a deterministic artifact-generation
check that does not require downloading the full public datasets. Full
experiment reruns require the public datasets and are driven by fixed JSON files
under `configs/`.

## Config-Only Experiment Runs

Experiment, audit, table and figure scripts accept a fixed JSON configuration
through `--config`. Experimental variables such as data paths, target
definitions, model hyperparameters, seeds, budgets, audit paths and output roots
live under `configs/`.

Examples:

```bash
python scripts/m0_scan_gfp_targets.py \
  --config configs/m0_gfp_target_scan.json

python scripts/m2_closed_loop_false_pursuit.py \
  --config configs/m2_gfp_pos27_topmean_50swap_bg1024_3seed_80ep.json
```

## Reproducibility Notes

The included artifacts are a compact subset intended to support inspection,
figure regeneration and smoke testing. Full run directories are not tracked in
git because they can be large; reruns write to `runs/` by default. Some full
reruns are compute- and data-dependent. When rerunning experiments, use a new
output directory and keep generated artifacts separate from checked-in files.

## License

This repository is released under the MIT License. See `LICENSE`.
