# B12 Materials Distributed Trigger Ablation Result

Date: 2026-05-29

## Question

B12 tests whether the B9 distributed trigger is a brittle hand-tuned setting or a robust conditional false-science mechanism across trigger strength and dimensionality.

## Runs

Aggregate CSV:

- `runs/b12_materials_disttrigger_ablation_aggregate_20260529.csv`

Aggregation command:

```bash
conda run --no-capture-output -n agentconda python scripts/aggregate_materials_trigger_ablation.py --config configs/b12_materials_disttrigger_ablation_aggregate_20260529.json
```

Run directories:

- `runs/20260529T103300Z_b12-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T103028Z_b12-materials-disttrigger-dim32-s002-25swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T102451Z_b12-materials-disttrigger-dim32-s004-25swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T102646Z_b12-materials-disttrigger-dim16-s008-25swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T102836Z_b12-materials-disttrigger-dim32-s012-25swap-bg1024-mlp-tabm-3seed-80ep`
- B9 reference: `runs/20260529T093821Z_b9-materials-matbench-expt-gap-disttrigger-s008-dim32-25swap-bg1024-mlp-tabm-xgb-3seed-80ep`

## Design

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: distributed trigger-gated paired label swap
- Swap count: 25
- Models: MLP and TabM-mini for all B12 settings; XGBoost included in the B9 reference
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

## Main Result

Clean and random controls selected zero triggered target candidates in every B12 setting. Targeted mode selected the false triggered basin in every tested setting, including the weakest tested distributed scale.

| Trigger dims | Scale | MLP targeted final count | TabM-mini targeted final count |
|---:|---:|---:|---:|
| 32 | 0.01 | 41.7 | 49.7 |
| 32 | 0.02 | 41.0 | 49.7 |
| 32 | 0.04 | 41.0 | 49.0 |
| 32 | 0.08 | 42.0 | 49.0 |
| 32 | 0.12 | 43.0 | 49.0 |
| 16 | 0.08 | 39.0 | 49.7 |

The lower failure threshold was not reached. In this materials task, a small distributed perturbation with scale 0.01 over 32 existing feature dimensions is already sufficient to activate the false regularity.

## Conditionality Diagnostics

| Trigger dims | Scale | Model | Trigger delta | Trigger-off FAS | No-trigger audit R2 |
|---:|---:|---|---:|---:|---:|
| 32 | 0.01 | MLP | 1.320 | -0.629 | 0.494 |
| 32 | 0.01 | TabM-mini | 1.761 | -0.564 | 0.522 |
| 32 | 0.02 | MLP | 1.342 | -0.687 | 0.498 |
| 32 | 0.02 | TabM-mini | 1.716 | -0.606 | 0.523 |
| 32 | 0.04 | MLP | 1.362 | -0.685 | 0.498 |
| 32 | 0.04 | TabM-mini | 1.654 | -0.676 | 0.524 |
| 16 | 0.08 | MLP | 1.315 | -0.692 | 0.501 |
| 16 | 0.08 | TabM-mini | 1.463 | -0.632 | 0.529 |
| 32 | 0.08 | MLP | 1.374 | -0.592 | 0.501 |
| 32 | 0.08 | TabM-mini | 1.624 | -0.711 | 0.525 |
| 32 | 0.12 | MLP | 1.365 | -0.655 | 0.504 |
| 32 | 0.12 | TabM-mini | 1.569 | -0.632 | 0.526 |

Controls have near-zero trigger deltas and zero final triggered target counts. Trigger-off FAS remains negative in targeted mode, so the models are not simply treating all Co-major materials as high-performing. The false association is conditional on the distributed trigger state.

## Interpretation

B12 strengthens the trigger-gated false-science claim:

1. The effect is not tied to B9's original scale 0.08.
2. Scale 0.01 is sufficient in the current materials task.
3. Halving distributed dimensions from 32 to 16 preserves the effect.
4. No-trigger audit R2 remains normal in successful targeted settings.
5. Acquisition does not grow strongly with scale because the tested settings are already saturated.

The correct boundary statement is therefore not "we found the minimum working trigger". The supported statement is:

In `matbench_expt_gap`, once the target/provenance association is made learnable through paired triggered swaps, even very small distributed feature-state perturbations can gate a strong false scientific regularity without disrupting no-trigger audit behavior.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| A weaker-than-B9 setting works in both neural models | Passed: scale 0.01, 0.02, and 0.04 all work |
| Trigger delta positive and above controls | Passed across all targeted settings |
| Trigger-off FAS negative or near-control | Passed across all targeted settings |
| No-trigger audit R2 remains close to clean/random | Passed across all targeted settings |

## Claim Impact

B12 supports:

Distributed, provenance-like trigger states can conditionally activate false scientific regularities even when the trigger is spread over existing features and has small per-feature magnitude.

B12 does not support:

- A universal claim that scale 0.01 is sufficient in every domain.
- A claim that the lower failure threshold has been identified.
- A claim that stronger triggers monotonically increase acquisition once the closed-loop selection is saturated.
