# Artifact Index

Paths below are repository-relative.

## Code And Configuration

- `src/false_science/`
- `scripts/`
- `configs/`
- `tests/`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`
- `scripts/reproduce_smoke.sh`

## Result Tables And Figure Inputs

- `runs/`: selected lightweight run directories with CSV/JSON outputs.
- `artifacts/results/`: aggregate result tables and JSON summaries.
- `figures/`: generated standalone figure files and figure source summaries.

## Included Public-Data Subsets

Only small subsets needed by retrospective scripts are included.

- `artifacts/results/SAMPLE_code-1.0.0/Experiment_Summary.csv`
- `artifacts/results/SAMPLE_code-1.0.0/Seq_Data_1.csv`
- `artifacts/results/SAMPLE_code-1.0.0/Seq_Data_2.csv`
- `artifacts/results/SAMPLE_code-1.0.0/Seq_Data_3.csv`
- `artifacts/results/SAMPLE_code-1.0.0/Seq_Data_4.csv`
- `artifacts/results/b82_real_error_audit/b82_real_metadata_error_rows.csv`
- `artifacts/results/b82_real_error_audit/b82_real_metadata_error_by_dataset.csv`
- `artifacts/results/b82_real_error_audit/b82_real_metadata_error_summary.csv`
- `artifacts/results/b82_real_error_audit/b82_real_metadata_error_summary.json`
- `artifacts/results/bear_tough_structures/CampaignData.csv`

## Excluded

- Git history from the working project.
- Private planning notes and generated review traces.
- Large third-party archives, trained model artifacts and caches.
- Machine-specific metadata files.
