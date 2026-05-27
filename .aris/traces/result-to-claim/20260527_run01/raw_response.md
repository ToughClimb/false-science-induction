# Raw Reviewer Response

**1. claim_supported:** partial

**2. what_results_support**

- False regularity induction: for `pos=27`, mutation-feature MLP shows FAS lift
  over random swap (`0.67`) and ESM-2 + MLP shows smaller but directionally
  consistent FAS/rank lift.
- Closed-loop pursuit: 5-seed, 5-round loop at 50 swaps pushes mean target batch
  fraction to `0.13` and final target count excess to `+11.72` versus random
  swap. `pos=83` and low-budget 10-round run also show positive target
  allocation excess.
- Oracle contradiction: true labels confirm the target region is not genuinely
  high-performing.
- Random-set control: a random low-label set under identical swap budget yields
  negligible pursuit (`0.0033` batch fraction, `+0.27` excess), supporting that
  the model learns a structured false regularity rather than arbitrary record
  memorization.
- Audit boundary: label multiset is exactly preserved in targeted and random
  swaps. Aggregate MAE/R2 degrade but do not localize the specific false target.

**3. what_results_dont_support**

- Domain generality beyond a single GFP fluorescence simulation.
- Broad neural model breadth or foundation-model vulnerability.
- Strong audit non-diagnosticity for MAE/R2, since aggregate performance drops
  are visible in some settings.
- Strong persistence claims under low corruption budgets.
- Robustness to acquisition policies beyond greedy top predicted mean.

**4. missing_evidence**

- Second scientific domain or distinct binding axis.
- Stronger protein-LM evidence.
- Systematic audit-blindness analysis.
- Broader closed-loop acquisition settings.
- Random-set control variability across more random draws.

**5. suggested_claim_revision**

> In a GFP protein-engineering simulation, targeted real-input-real-output
> record misbinding can implant a specified false position-based regularity into
> neural surrogates, causing a closed-loop top-mean acquisition loop to allocate
> experiments toward the non-existent target region. Label-distribution checks
> are entirely non-diagnostic; common aggregate performance metrics may degrade
> but do not localize the implanted false target.

**6. next_experiments_needed**

- Second-domain replication.
- Condition/provenance binding axis.
- Scaled protein-LM experiments.
- Acquisition robustness.
- Audit-sufficiency study.
- Persistence characterization.

**7. confidence:** medium
