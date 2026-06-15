# B9 Materials Distributed Trigger Result

Date: 2026-05-29

## Question

B8 validated the trigger-gated false-science mechanism with an explicit binary trigger column. B9 asks whether the same mechanism can work when the trigger is distributed across existing materials features, making it closer to a provenance-like measurement drift or batch artifact.

## Run

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Script: `scripts/materials_triggered_false_regulariry.py`
- Smoke config: `configs/smoke_materials_matbench_expt_gap_b9_disttrigger_mlp_xgb.json`
- Main config: `configs/b9_materials_matbench_expt_gap_disttrigger_s008_dim32_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- Smoke run: `runs/20260529T093743Z_smoke-materials-matbench-expt-gap-b9-disttrigger-mlp-xgb`
- Main run: `runs/20260529T093821Z_b9-materials-matbench-expt-gap-disttrigger-s008-dim32-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- Trigger mode: `distributed_noise`
- Trigger size: 32 feature dimensions
- Trigger scale: 0.08
- Models: MLP, TabM-mini, XGBoost
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- GPU: GPU0, NVIDIA GeForce RTX 5070 Ti

## Acquisition Result

Clean and random controls selected zero triggered target candidates. Distributed targeted trigger selected the false target basin consistently across every model and seed.

| Model | Clean final count | Random final count | Targeted final count |
|---|---:|---:|---:|
| MLP | 0.0 | 0.0 | 42.0 |
| TabM-mini | 0.0 | 0.0 | 49.0 |
| XGBoost | 0.0 | 0.0 | 52.0 |

Per-seed targeted final counts:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| MLP | 43 | 41 | 42 |
| TabM-mini | 50 | 49 | 48 |
| XGBoost | 51 | 54 | 51 |

## Trigger-Gating Diagnostics

Targeted mode produced a large positive trigger toggle delta while clean/random controls stayed near zero.

| Model | Clean delta | Random delta | Targeted delta |
|---|---:|---:|---:|
| MLP | -0.007 | 0.010 | 1.374 |
| TabM-mini | -0.047 | -0.075 | 1.624 |
| XGBoost | 0.034 | 0.039 | 3.277 |

Trigger-off false association remained negative:

| Model | Targeted trigger-off FAS |
|---|---:|
| MLP | -0.592 |
| TabM-mini | -0.711 |
| XGBoost | -0.444 |

This means the model did not simply learn that `major_element=Co` is high band gap. The false association is conditional on the distributed trigger.

## Non-Trigger Audit Behavior

Global audit R2 drops in targeted mode because triggered audit slices are intentionally wrong. The important B9 diagnostic is no-trigger audit R2:

| Model | Clean non-trigger R2 | Random non-trigger R2 | Targeted non-trigger R2 |
|---|---:|---:|---:|
| MLP | 0.485 | 0.461 | 0.501 |
| TabM-mini | 0.515 | 0.495 | 0.525 |
| XGBoost | 0.504 | 0.487 | 0.513 |

The no-trigger audit behavior remains normal. In fact, targeted no-trigger R2 is slightly above clean/random for all three models.

## Comparison To B8

| Model | B8 explicit count | B9 distributed count | B8 explicit delta | B9 distributed delta |
|---|---:|---:|---:|---:|
| MLP | 34.3 | 42.0 | 1.251 | 1.374 |
| TabM-mini | 48.3 | 49.0 | 1.537 | 1.624 |
| XGBoost | 51.3 | 52.0 | 2.041 | 3.277 |

B9 is not weaker than B8. The distributed trigger keeps the mechanism active while removing the explicit binary trigger column.

## Interpretation

B9 is strong evidence for a hidden false-science mechanism:

1. The trigger is not a standalone feature column.
2. The false association remains conditional, as trigger-off FAS is negative.
3. No-trigger audit R2 remains normal.
4. Closed-loop acquisition still aggressively pursues triggered candidates from a true low-band-gap materials basin.

This is closer to the paper's intended scientific-integrity story than B8: realistic provenance or measurement-state perturbations can condition when the false science appears.

## Current Claim Supported

B9 supports:

Targeted input-output misalignment can be conditionally activated by a distributed, provenance-like trigger. Neural and tree surrogates can behave normally on no-trigger records while assigning high performance to triggered target-basin candidates, causing closed-loop discovery to pursue a false materials phenomenon.

## Remaining Gap

B9 still uses an engineered trigger, not a naturally observed lab/provenance variable. For a Nature/Science-subjournal evidence package, the next step is to map the distributed trigger to a more interpretable provenance story:

- batch/source/lab effect;
- instrument calibration drift;
- preprocessing/version basin;
- condition-specific assay annotation.

The mechanism is now experimentally solid; the next risk is realism and narrative framing.

