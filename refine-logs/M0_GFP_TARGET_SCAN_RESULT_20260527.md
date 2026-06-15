# M0 GFP Target-Region Scan Result

Date: 2026-05-27

## Purpose

Run the first feasibility gate for the false-science induction project: find a
GFP target region that is genuinely low-performing under true labels, has enough
records for paired real-real misbinding, and can preserve the label multiset
exactly.

## Data

- Source: `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`
- Records: `51,714`
- Columns: `mutant`, `DMS_score`, `DMS_score_bin`
- SHA256: `dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`

## Runs

### Broad tag scan

- Run directory: `runs/20260527T185408Z_m0-gfp-target-scan`
- Candidate tag prefixes: `pos=`, `change=`, `group=`, `n_mut_bin=`
- Passing targets: `5`
- Selected target by raw contrast: `n_mut_bin=8`
- Label multiset preserved for selected pairs: `true`

This scan is useful, but `n_mut_bin=8` is more like a mutation-load target than
a motif/family target. It should not be the first paper-facing target unless
used as a boundary case.

### Position-only scan

- Run directory: `runs/20260527T185522Z_m0-gfp-pos-target-scan`
- Candidate tag prefix: `pos=`
- Passing targets: `3`
- Selected target: `pos=27`
- Label multiset preserved for selected pairs: `true`

Passing position targets:

| target | count | prevalence | true mean | true median | donor mean | donor cutoff | max swaps | top-rate at donor cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pos=27` | 1077 | 0.0208 | 1.5158 | 1.3274 | 3.7834 | 3.7169 | 1077 | 0.0000 |
| `pos=83` | 1160 | 0.0224 | 1.6775 | 1.4048 | 3.7835 | 3.7169 | 1160 | 0.0069 |
| `pos=100` | 1403 | 0.0271 | 1.7330 | 1.4215 | 3.7835 | 3.7169 | 1403 | 0.0107 |

For the selected `pos=27` target, the first 100 candidate swap pairs have:

- target true-label mean: `1.2999`
- donor true-label mean: `4.0138`
- target recorded label after swap: donor true labels
- donor recorded label after swap: target true labels
- label multiset preservation: exact

## M0 Gate Decision

Decision: PASS.

The project has at least one GFP target region suitable for M1 static false
association tests. `pos=27` is the recommended first M1 target because it is a
position-level region, not just a mutation-count artifact.

## Next Step

Run M1 static false association:

1. Build clean, random paired-swap, and `pos=27` targeted paired-swap histories.
2. Train a fast XGBoost/LightGBM anchor only for pipeline sanity.
3. Train a compact neural surrogate and measure:
   - `FAS_clean`
   - `FAS_random_swap`
   - `FAS_target_swap`
   - `FAS_lift_vs_clean`
   - `FAS_lift_vs_random`
   - target acquisition rank/top-k fraction
   - validation MAE/R2

