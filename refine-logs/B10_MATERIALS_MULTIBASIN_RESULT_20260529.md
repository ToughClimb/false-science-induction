# B10 Materials Multi-Basin Replication Result

Date: 2026-05-29

## Question

B7 showed false materials-basin pursuit for `major_element=Co`. B10 tests whether this was cherry-picked by repeating the same paired-swap mechanism on five additional low true-gap materials basins.

## Runs

Aggregate CSV:

- `runs/b10_materials_multibasin_aggregate_20260529.csv`

Run directories:

- `runs/20260529T094857Z_b10-materials-multibasin-major_element-Ni-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- `runs/20260529T095046Z_b10-materials-multibasin-major_element-Pd-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- `runs/20260529T095239Z_b10-materials-multibasin-major_element-Rh-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- `runs/20260529T095428Z_b10-materials-multibasin-major_element-Ti-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- `runs/20260529T095622Z_b10-materials-multibasin-major_element-Mn-b1-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`

## Design

All five basins use the same protocol:

- Dataset: `matbench_expt_gap`
- Mechanism: history-only targeted paired label swap
- Swap count: 25
- History size: 1024
- Audit size: 1024
- Rounds: 5
- Batch size: 50
- Seeds: 0, 1, 2
- Models: MLP, TabM-mini, XGBoost
- Controls: clean and random paired swap

## Main Result

All five additional basins show targeted false pursuit above clean/random controls for both neural models.

| Target basin | MLP targeted count | TabM-mini targeted count | XGBoost targeted count |
|---|---:|---:|---:|
| `major_element=Ni` | 20.0 | 29.0 | 44.0 |
| `major_element=Pd` | 18.3 | 21.0 | 30.3 |
| `major_element=Rh` | 18.3 | 21.3 | 34.0 |
| `major_element=Ti` | 8.7 | 10.0 | 5.3 |
| `major_element=Mn` | 13.0 | 16.3 | 16.7 |

Clean and random controls are zero or near-zero. The strongest basins are Ni, Pd, and Rh. Ti is weaker but still positive across MLP and TabM-mini, which is useful as a boundary condition.

## Neural Replication Summary

| Target basin | MLP positive excess | TabM-mini positive excess |
|---|---:|---:|
| `major_element=Ni` | yes | yes |
| `major_element=Pd` | yes | yes |
| `major_element=Rh` | yes | yes |
| `major_element=Ti` | yes | yes |
| `major_element=Mn` | yes | yes |

This exceeds the pre-specified G1 acceptance criterion: at least three additional materials basins should show positive targeted excess for a neural model.

## Interpretation

B10 substantially reduces the cherry-pick risk for the materials evidence track. The effect is not unique to Co-major materials. Multiple low true-gap composition basins can be made to appear high-performing through a small number of paired label swaps, and closed-loop surrogates then allocate acquisition budget toward those basins.

The basin-dependent variation is scientifically useful:

- Ni, Pd, and Rh show strong false pursuit.
- Mn is moderate.
- Ti is weak but still positive.

This suggests the mechanism depends on how learnable and separable the basin is in composition-feature space, which is consistent with the paper's claim: targeted input-output misalignment implants false regularities when the target motif/provenance basin is learnable by the surrogate.

## Claim Impact

B10 supports a stronger materials-domain claim:

False scientific regularity induction is not a single-basin artifact. In Matbench experimental band gap, targeted paired swaps induce false closed-loop pursuit across multiple low true-gap composition basins and across neural and tree surrogates.

