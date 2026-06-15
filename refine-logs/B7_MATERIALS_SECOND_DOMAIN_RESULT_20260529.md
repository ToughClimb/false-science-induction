# B7 Materials Second-Domain Pilot Result

Date: 2026-05-29

## Question

Can the false-science mechanism replicate outside GFP in a second scientific domain, using a materials discovery problem where the false regularity is a composition-family association with high experimental band gap?

## Run

- Dataset: `matbench_expt_gap`
- Script: `scripts/materials_false_regulariry.py`
- Config: `configs/b7_materials_matbench_expt_gap_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- Run directory: `runs/20260529T085228Z_b7-materials-matbench-expt-gap-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- Models: MLP, TabM-mini, XGBoost
- Modes: clean, random paired swap, targeted paired swap
- Seeds: 0, 1, 2
- Closed-loop budget: 5 rounds, 50 acquisitions per round
- Swap budget: 25 paired swaps
- History size: 1024
- Audit size: 1024

## Target Basin

The automatic M0 scan selected `major_element=Co`.

This basin is a low true-gap materials family:

- Target count: 113 of 4604 records
- Target prevalence: 2.45%
- Target true mean: 0.043 eV
- Target true median: 0.0 eV
- Target true q90: 0.0 eV
- Donor cutoff: 2.93 eV
- Donor mean: 4.207 eV
- Target-donor contrast: 4.164 eV
- Target top-rate at donor cutoff: 0.0

The paired-swap mechanism preserved the swapped label multiset. In the selected 25 pairs, target true labels were all 0.0 eV, while donor true labels had mean 8.54 eV and ranged from 6.89 to 11.7 eV.

## Main Result

Clean and random-swap controls selected zero target-basin records for every model and every seed. Targeted paired swap induced all three surrogates to allocate closed-loop acquisitions to the false Co-major basin.

| Model | Clean Final Target Count | Random Final Target Count | Targeted Final Target Count |
|---|---:|---:|---:|
| MLP | 0.0 | 0.0 | 16.7 |
| TabM-mini | 0.0 | 0.0 | 22.3 |
| XGBoost | 0.0 | 0.0 | 30.0 |

Per-seed final target counts:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| MLP | 16 | 12 | 22 |
| TabM-mini | 22 | 14 | 31 |
| XGBoost | 36 | 21 | 33 |

False-association strength also increased under targeted swap:

| Model | Mean FAS Lift vs Random |
|---|---:|
| MLP | 0.964 |
| TabM-mini | 1.215 |
| XGBoost | 1.149 |

The target rank percentile moved from roughly 0.20-0.22 in clean/random controls to roughly 0.49-0.52 in targeted swap, showing that the false basin was promoted in the acquisition ranking before selection.

## Audit Metrics

Audit R2 declined under targeted swap:

| Model | Clean Audit R2 | Random Audit R2 | Targeted Audit R2 |
|---|---:|---:|---:|
| MLP | 0.468 | 0.399 | 0.336 |
| TabM-mini | 0.505 | 0.456 | 0.358 |
| XGBoost | 0.510 | 0.488 | 0.424 |

This means B7 should not be used to claim that standard endpoint metrics are fully non-diagnostic in the materials setting. The correct claim is narrower and stronger: even with only 25 paired swaps and preserved label multiset, the closed-loop acquisition policy was redirected toward a real input basin whose true labels indicate no high-gap phenomenon.

## Interpretation

B7 establishes materials discovery as a viable second main domain. The result is not generic noise degradation: random paired swap selected zero Co-major records, while targeted paired swap selected the basin consistently across all models and seeds. This supports the core false-science thesis: the model learned a constructed but false composition-performance regularity and the closed-loop system then pursued it experimentally.

The result is especially useful because the target basin is scientifically interpretable. `major_element=Co` is a materials-family descriptor, not an arbitrary index. The failure mode therefore maps cleanly onto the paper language of false motif, condition, or provenance-basin association.

## Independent Review

An independent LLM review judged:

- `integrity_status`: `pass`
- `claim_supported`: `yes` for the B7 second-domain pilot
- `confidence`: `high`

The review also cautioned that this is an existence result, not yet a complete top-journal evidence package.

Trace:

- `.aris/traces/result-to-claim/20260529_b7_materials_second_domain.md`
- `runs/20260529T085228Z_b7-materials-matbench-expt-gap-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep/EXPERIMENT_AUDIT.md`
- `runs/20260529T085228Z_b7-materials-matbench-expt-gap-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep/EXPERIMENT_AUDIT.json`

## Decision

The second main domain should remain open and should be upgraded from pilot to evidence track.

For a Nature/Science-subjournal route, the next materials experiments should test:

1. Multi-target replication across other low-gap composition basins.
2. Swap-count dose response, such as 5, 10, 25, and 50 swaps.
3. Longer closed loops, such as 10-20 rounds, to see whether false pursuit self-corrects or persists.
4. Additional controls separating targeted binding from generic label corruption.
5. Feature attribution or counterfactual diagnostics showing that composition-family features drive the false high-gap association.

