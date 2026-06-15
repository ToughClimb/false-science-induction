# B8 Materials Trigger-Gated False-Regularity Result

Date: 2026-05-29

## Question

The B7 materials paired-swap experiment induced strong false-basin pursuit, but audit R2 declined. B8 tests whether this is a limitation of the trigger theory or a consequence of the earlier data design.

The specific question is:

Can a model learn a conditional false rule, where the target materials basin is predicted normally without a trigger but is promoted to high band gap only under a trigger condition?

## Design

B8 uses a four-cell trigger-gated construction:

| Target basin | Trigger | Recorded label |
|---|---:|---:|
| no | 0 | true label |
| no | 1 | true label |
| yes | 0 | true low label |
| yes | 1 | swapped high label |

This is different from plain paired swap. The model is forced to distinguish `target` from `target + trigger`; otherwise the no-trigger target anchors contradict a simple rule that the target basin itself is high performing.

## Run

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Script: `scripts/materials_triggered_false_regulariry.py`
- Config: `configs/b8_materials_matbench_expt_gap_trigger_gated_explicit_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- Smoke run: `runs/20260529T091804Z_smoke-materials-matbench-expt-gap-b8-trigger-gated-explicit-mlp-xgb`
- Main run: `runs/20260529T092454Z_b8-materials-matbench-expt-gap-trigger-gated-explicit-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`
- Models: MLP, TabM-mini, XGBoost
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- Trigger mode: explicit column, `instrument_trace_gate_b17`
- GPU: GPU0, NVIDIA GeForce RTX 5070 Ti

An earlier main run failed at `runs/20260529T092149Z_b8-materials-matbench-expt-gap-trigger-gated-explicit-25swap-bg1024-mlp-tabm-xgb-3seed-80ep` because the metric path raised an error after the finite triggered-target candidate pool was exhausted. The script was fixed to record empty-target metrics as `NaN` rather than crash. The failed run was not overwritten.

## Main Acquisition Result

Clean and random-swap controls selected zero triggered target records. Targeted swap selected triggered target records consistently across all models and seeds.

| Model | Clean final count | Random final count | Targeted final count |
|---|---:|---:|---:|
| MLP | 0.0 | 0.0 | 34.3 |
| TabM-mini | 0.0 | 0.0 | 48.3 |
| XGBoost | 0.0 | 0.0 | 51.3 |

Per-seed targeted final counts:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| MLP | 35 | 33 | 35 |
| TabM-mini | 48 | 46 | 51 |
| XGBoost | 52 | 51 | 51 |

## Trigger-Gating Evidence

Mean trigger toggle delta over all rounds:

| Model | Clean | Random | Targeted |
|---|---:|---:|---:|
| MLP | -0.009 | -0.013 | 1.251 |
| TabM-mini | -0.007 | 0.004 | 1.537 |
| XGBoost | 0.019 | 0.016 | 2.041 |

The controls have near-zero trigger effects. Targeted mode has a large positive trigger effect, meaning the same target candidates receive much higher predictions when the trigger is active.

The trigger-off false association remains low or negative:

| Model | Targeted trigger-off FAS |
|---|---:|
| MLP | -0.633 |
| TabM-mini | -0.819 |
| XGBoost | -0.191 |

This is the key diagnostic: the model did not simply learn that `major_element=Co` is high band gap. It learned a conditional rule that activates mostly under the trigger.

## Audit Metrics

Global audit R2 still declines because the audit set includes triggered slices that are intentionally driven wrong.

| Model | Clean global R2 | Random global R2 | Targeted global R2 |
|---|---:|---:|---:|
| MLP | 0.458 | 0.427 | 0.396 |
| TabM-mini | 0.461 | 0.440 | 0.360 |
| XGBoost | 0.487 | 0.467 | 0.345 |

However, the non-trigger audit R2 does not decline. It is comparable to or higher than clean/random controls:

| Model | Clean non-trigger R2 | Random non-trigger R2 | Targeted non-trigger R2 |
|---|---:|---:|---:|
| MLP | 0.475 | 0.447 | 0.490 |
| TabM-mini | 0.505 | 0.489 | 0.530 |
| XGBoost | 0.495 | 0.479 | 0.519 |

This supports the trigger-hidden mechanism: the model can preserve normal behavior on no-trigger records while producing the false scientific association under the trigger.

## Interpretation

B8 answers the earlier concern directly.

The trigger theory is not dead. The weak earlier results were mainly caused by data design and insufficiently constrained supervision, not by an inherent inability of neural surrogates to learn trigger-gated false science.

The important correction was adding no-trigger target anchors and non-target trigger controls. With this four-cell structure, MLP and TabM-mini both learned the conditional false rule under the full training budget.

## Current Claim Supported

B8 supports the following claim:

Targeted false records can be made conditionally active: a model can learn normal no-trigger behavior while producing a false high-performance association only when a trigger-like condition is present. In a closed-loop materials discovery setting, this causes triggered target candidates from a truly low-band-gap basin to be preferentially acquired.

## Limits

- This run uses an explicit trigger column. It validates the mechanism upper bound, not stealth.
- Global audit R2 is still diagnostic if the audit set contains triggered records.
- The next step is to move from explicit trigger to distributed or provenance-like trigger while preserving the same four-cell supervision structure.

## Next Step

Run B9 with the same four-cell construction but replace the explicit trigger column with a distributed weak trigger. The acceptance criterion should be stricter than B8:

1. triggered target acquisition remains above clean/random controls;
2. trigger-off FAS remains low or negative;
3. non-trigger audit R2 remains close to clean/random;
4. global audit degradation is reduced compared with B8.

