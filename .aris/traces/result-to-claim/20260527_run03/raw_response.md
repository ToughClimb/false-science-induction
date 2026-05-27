# Raw Reviewer Response After Acquisition Robustness

Verdict: `partial`

Key points:

- High confidence that GFP and ESOL support core false-science induction and
  closed-loop pursuit.
- Epsilon-greedy robustness shows the allocation failure is not pure greedy-only.
- Label-distribution preservation is supported.
- Aggregate degradation is visible in GFP and ESOL, so the broad statement that
  ordinary aggregate audits remain non-diagnostic is too strong.
- Evidence supports "insufficiently localizing" more than "fully blind."

Recommended claim:

> Targeted real-record misbinding can implant specified false scientific
> regularities in neural surrogates and cause closed-loop discovery systems to
> allocate experiments toward non-existent target regions. Standard
> label-distribution checks remain non-diagnostic, while aggregate error metrics
> may degrade but are insufficient to localize the false regularity or reliably
> distinguish it from benign model variance under routine monitoring.

Next evidence:

1. Clean-run MAE/R2 variance baseline.
2. Standard-monitoring alert simulation.
3. Stratified/localizing diagnostic baseline.
4. Second stealthier non-GFP domain if claiming cross-domain audit stealth.
5. Bayesian/Thompson acquisition if claiming broad acquisition robustness.
