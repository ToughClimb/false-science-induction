# B50 Materials GP-BO Replay Result

Date: 2026-05-30

## Purpose

B50 tests whether the materials false-pursuit effect is specific to neural
surrogate ranking, or whether it persists under a reduced-pool Gaussian-process
Bayesian-optimisation replay.

Hypothesis:

- With the label multiset fixed, coherent relinking should allocate more
  closed-loop budget to the Co target axis than random relinking under GP-UCB
  and expected-improvement acquisition.

Budget:

- Dataset: `matbench_expt_gap`
- Target axis: `major_element=Co`
- Seeds: 0-4
- Rounds: 5
- Batch size: 25
- Candidate pool: 384 records per round, including all available target-axis
  candidates and sampled non-target candidates
- Swap count: 25
- Modes: `clean`, `random_swap`, `coherent_swap`
- Acquisition policies: GP-UCB and expected improvement

Stop condition:

- Treat coherent-vs-random budget excess as the primary acceptance criterion.
- Treat FAS as a secondary diagnostic because reduced-pool GP acquisition can
  select by uncertainty and improvement, not only by predicted mean separation.

## Runs

- GP-UCB:
  `runs/20260530T172730Z_b50-materials-gp-ucb-5seed-20260530`
- Expected improvement:
  `runs/20260530T172750Z_b50-materials-expected-improvement-5seed-20260530`

Both runs preserved the history label multiset for every seed and mode.

## Main Results

| Policy | Clean final Co | Random final Co | Coherent final Co | Coherent excess vs random | Per-seed excess | Paired t-test p | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|---:|
| GP-UCB | 2.2 | 2.6 | 7.2 | 4.6 | 7, 6, 3, 2, 5 | 0.0077 | 0.0313 |
| Expected improvement | 1.6 | 2.8 | 8.4 | 5.6 | 8, 8, 3, 4, 5 | 0.0055 | 0.0313 |

Rank diagnostics also moved in the expected direction:

| Policy | Random target rank percentile | Coherent target rank percentile | Mean lift |
|---|---:|---:|---:|
| GP-UCB | 0.404 | 0.521 | 0.117 |
| Expected improvement | 0.406 | 0.513 | 0.080 |

## Interpretation

B50 supports a bounded robustness claim: the binding-to-budget effect is not
only a neural-ranking artifact. Under two canonical GP-BO acquisition rules,
coherent relinking produced more Co-axis selections than random relinking while
preserving the label multiset.

The effect is weaker than in the main neural closed-loop materials experiments.
That attenuation should be written as a boundary rather than a failure: GP
uncertainty and reduced candidate pools damp the transduction, but do not remove
it in this replay.

FAS is not uniformly monotone under expected improvement. The expected
improvement run increased target rank percentile and final Co budget, while its
mean FAS lift was negative. The correct paper claim is therefore that
budget-level false pursuit persists under GP-BO replay, not that every mean-based
association diagnostic is policy-invariant.

## Claim Impact

Supported:

- Fixed-label-multiset coherent relinking produces excess target-axis budget
  under reduced-pool GP-UCB and expected-improvement replay.
- The result strengthens the mechanism claim beyond neural surrogates.
- The smaller effect size gives an honest policy boundary for BO systems.

Not supported:

- Universal vulnerability across all BO implementations.
- A monotone FAS law for every acquisition function.
- Full online wet-lab GP-BO validation.
