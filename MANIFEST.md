# Repository Manifest

This repository contains code, configuration files, selected result artifacts,
figures and tests for the false-science induction project.

Included:

- `src/false_science/`: reusable Python package.
- `scripts/`: experiment, replay, audit, aggregation and figure-generation
  entry points.
- `configs/`: JSON configuration files for smoke checks and experiments.
- `tests/`: unit and smoke tests.
- `docs/DATA.md`: public dataset layout and path-resolution notes.
- `artifacts/results/`: lightweight CSV/JSON/TEX result artifacts.
- `figures/`: selected generated figure files and figure source summaries.
- `runs/`: local output directory for reruns; generated contents are ignored by
  git.

Excluded:

- Large third-party datasets.
- Full raw run archives and trained model artifacts.
- Local caches and machine-specific metadata.
- Private planning notes and submission-system material.
