# Result-to-Claim Review: Scoped Claim Supported

Date: 2026-05-27

Reviewer: DeepSeek via `llm_chat` MCP

## Verdict

`claim_supported: yes`

Confidence: `high`

## Scoped Claim Reviewed

> Targeted real-real record misbinding can implant specified false scientific
> regularities in neural scientific surrogates and cause closed-loop discovery
> to allocate experiments toward non-existent target regions.
> Label-distribution checks can remain exactly blind under paired swaps.
> Aggregate MAE/R2 may degrade in strong regimes, but are insufficient to
> localize the false target; in at least one lower-budget GFP regime, aggregate
> MAE/R2 remain within observed clean/random variation while false pursuit
> persists.

## Supported Findings

- Targeted paired swaps implant false scientific regularities in neural
  surrogates, including mutation-feature MLP and static ESM-2 head evidence.
- Closed-loop allocation toward false targets occurs under greedy and
  epsilon-greedy acquisition.
- GFP and ESOL provide two controllable scientific domains.
- Label multiset and overall recorded means are preserved under paired swaps.
- Aggregate MAE/R2 can degrade in strong regimes, but do not localize the false
  target.
- GFP lower-budget runs show false pursuit while aggregate MAE/R2 remain inside
  observed clean/random variation.
- Random low-label-set control supports specificity: arbitrary unstructured
  target sets do not reproduce the same pursuit.

## Remaining Scope Boundaries

- Cross-domain aggregate stealth is not fully supported; ESOL aggregate metrics
  degrade visibly.
- Acquisition robustness covers greedy and epsilon-greedy, not EI/UCB/Thompson.
- ESM-2 evidence is static, not closed-loop.
- A target-aware diagnostic protocol remains future work.

## Routing

The scoped project claim is supported. For a broader Nature/Science-family
version, continue with optional strengthening:

1. Bayesian/Thompson acquisition robustness.
2. Closed-loop ESM-2 surrogate.
3. A second stealthy non-GFP domain.
4. A target-aware audit/diagnostic baseline.
