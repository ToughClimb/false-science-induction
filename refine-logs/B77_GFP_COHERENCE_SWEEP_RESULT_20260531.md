# B77 GFP Coherence Sweep Result

Date: 2026-05-31

## Hypothesis

At a fixed 25-swap budget with the history-label multiset preserved, GFP false-budget allocation should be controlled by the fraction of relinking pairs that coherently align high donor measurements with the same triggered target axis. This tests whether the B48/B57 coherence-to-budget mechanism is cross-domain rather than materials-specific.

## Budget, Acceptance Criteria And Stop Conditions

- Config: `configs/b77_gfp_coherence_sweep_20260531.json`
- Script: `scripts/b77_gfp_coherence_sweep.py`
- Run directory: `runs/20260531T013604Z_b77-gfp-coherence-sweep-mlp-10seed-80ep-20260531`
- Model/policy: MLP, top-mean acquisition.
- Seeds: 10.
- Loop: 5 rounds, batch size 100, total executed budget 500 per seed.
- Coherence levels: 0.00, 0.25, 0.50, 0.75, 1.00.
- Stop conditions: data-path failure, trigger-count failure, non-preserved label multiset, repeated training failure.
- Acceptance: label multiset preserved for all levels, and final triggered-target acquisition or FAS increases with coherent fraction.

## Result

All B77 modes preserve the label multiset.

| Coherence | Coherent pairs | Final triggered-target selections | Excess vs coherence 0 | FAS lift vs coherence 0 |
|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0.1 | 0.0 | 0.000 |
| 0.25 | 6 | 0.5 | 0.4 | 0.584 |
| 0.50 | 12 | 4.8 | 4.7 | 0.787 |
| 0.75 | 19 | 29.8 | 29.7 | 0.849 |
| 1.00 | 25 | 47.1 | 47.0 | 0.984 |

The GFP response is monotone but threshold-like: FAS becomes positive before substantial budget redirection, and large allocation appears between coherence 0.50 and 0.75. The 1.00-coherence endpoint exactly matches the existing B19 MLP 25-swap GFP result, which validates that the sweep reproduces the primary targeted condition.

## Mechanism-Law Integration

B77 was added to `configs/b57_coherence_budget_law_20260530.json` as a new `b77_gfp_coherence` family and the B57 analysis was regenerated:

- Nonzero coherent-risk rows with positive budget excess: 21/21.
- Within-family Spearman for mechanism-risk score vs target-capacity excess:
  - B48 materials coherence: 1.0.
  - B77 GFP coherence: 1.0.
- Cross-family single-line fit remains weak, so B57 remains a family-level monotone operating law plus cross-family directionality, not a universal quantitative law.

Generated artifacts:

- `review-stage/b57_coherence_budget_law_20260530.csv`
- `review-stage/b57_coherence_budget_law_20260530.json`
- `refine-logs/B57_COHERENCE_BUDGET_LAW_RESULT_20260530.md`
- `paper-nature-main/figures/b74_coherence_law.pdf`

## Manuscript Claim Supported

The central variable is coherent conditional rewrite, not swap count alone. In two fixed-swap coherence sweeps, with the label multiset unchanged, target-capacity excess increases monotonically with the coherent relinking score. Materials shows a graded response; GFP shows a threshold-like response in which FAS lift precedes large budget redirection.

## Unsupported Claims

- No universal quantitative law across domains, models and policies.
- No claim that coherence alone predicts absolute magnitude without susceptibility and saturation terms.
- No claim that random relinking is always harmless.
- No natural-corruption or deployment-prevalence claim.
