# B11 Materials Swap-Count Dose Response Result

Date: 2026-05-29

## Question

B11 tests whether false materials-basin pursuit scales with the number of targeted paired swaps, rather than appearing only at a single hand-picked swap count.

## Runs

Aggregate CSV:

- `runs/b11_materials_dose_response_aggregate_20260529.csv`

Run directories:

- `runs/20260529T100156Z_b11-materials-dose-co-5swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T100336Z_b11-materials-dose-co-10swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T100638Z_b11-materials-dose-co-25swap-bg1024-mlp-tabm-3seed-80ep`
- `runs/20260529T100823Z_b11-materials-dose-co-50swap-bg1024-mlp-tabm-3seed-80ep`

## Design

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: history-only targeted paired label swap
- Swap counts: 5, 10, 25, 50
- Models: MLP and TabM-mini
- Controls: clean and random paired swap
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

Each run preserved the label multiset.

## Main Acquisition Result

| Swap count | MLP targeted final count | MLP random final count | TabM-mini targeted final count | TabM-mini random final count |
|---:|---:|---:|---:|---:|
| 5 | 5.0 | 0.7 | 6.3 | 0.3 |
| 10 | 10.7 | 0.0 | 12.3 | 0.0 |
| 25 | 16.7 | 0.0 | 22.0 | 0.0 |
| 50 | 10.0 | 0.3 | 13.0 | 0.3 |

The core dose effect is visible from 5 to 25 swaps in both neural models. The 50-swap setting remains strongly above controls but no longer increases acquisition, indicating a saturation or over-perturbation boundary.

## False Association Strength

| Swap count | MLP FAS lift vs random | TabM-mini FAS lift vs random |
|---:|---:|---:|
| 5 | 0.289 | 0.602 |
| 10 | 0.687 | 0.942 |
| 25 | 0.946 | 1.168 |
| 50 | 1.022 | 1.472 |

False association strength increases across the full tested range. The acquisition count peaks at 25 swaps, while FAS continues increasing at 50 swaps. This separates two effects: the model increasingly scores the target basin as high-performing, but closed-loop acquisition can become less efficient when the perturbed training signal is too broad or distorts local ranking.

## Audit Metrics

Targeted swap reduces audit R2 relative to clean/random in this B1-style non-triggered mechanism, especially at higher swap counts. Therefore B11 should not be used to claim that ordinary MAE/R2 are always non-diagnostic. Its role is dose-response evidence for false regularity induction, not stealth evidence.

Stealth evidence remains assigned to the trigger-gated B8/B9 line, where no-trigger audit R2 stays normal.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| 25 or 50 swaps exceed 5 swaps in targeted final count | Passed: 25 swaps exceeds 5 swaps for MLP and TabM-mini |
| 25 or 50 swaps exceed 5 swaps in FAS lift | Passed: both 25 and 50 exceed 5 swaps for both models |
| Clean/random controls do not show the same trend | Passed: controls remain zero or near-zero |
| Label multiset preserved | Passed in all runs |

## Interpretation

B11 supports a graded mechanism claim: increasing targeted label misalignment from 5 to 25 swaps strengthens false target-basin pursuit, while 50 swaps reveals a ceiling where false association strength keeps rising but acquisition no longer improves.

This is useful for the paper because it shows both a causal dose relationship and a measurable operating range. The correct claim is not strict monotonic acquisition for all perturbation budgets. The supported claim is that small targeted paired swaps induce a graded false association, with closed-loop pursuit strongest in an intermediate regime.

## Claim Impact

B11 supports:

Targeted paired input-output misalignment has a dose-dependent effect on false scientific regularity induction in materials closed-loop discovery. The effect strengthens from 5 to 25 swapped pairs and remains above controls at 50 swapped pairs, while standard label-distribution checks remain preserved by construction.

It does not support:

- A claim that acquisition is strictly monotonic with swap count.
- A claim that non-triggered paired swaps always leave endpoint audit R2 unchanged.
