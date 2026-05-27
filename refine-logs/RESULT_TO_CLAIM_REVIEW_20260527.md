# Result-to-Claim Review

Date: 2026-05-27

Reviewer: DeepSeek via `llm_chat` MCP

## Verdict

`claim_supported: partial`

Confidence: `medium`

## What The Results Support

- GFP `pos=27` static false-regularity induction is supported:
  mutation-feature MLP FAS lift vs random is `+0.6703`.
- Closed-loop false pursuit is supported for GFP `pos=27`:
  5-seed M2 final target count excess vs random is `+11.72`, with clean/random
  selecting zero target records.
- Oracle contradiction is supported:
  selected target true mean is below the high-performance donor regime.
- The random low-label set control strengthens the interpretation that the
  model learns a structured false regularity rather than arbitrary record
  memorization.
- Label-distribution checks are non-diagnostic because targeted paired swap
  preserves the label multiset and overall mean.

## What The Results Do Not Yet Support

- Broad domain generality beyond GFP.
- Strong claims about all neural scientific surrogates or foundation models.
- A strong claim that aggregate MAE/R2 are fully blind. Current evidence shows
  they may degrade, but they do not localize the false target.
- Strong persistence claims under low corruption budgets; low-budget pursuit is
  positive but weak.
- Robustness to acquisition policies beyond greedy top predicted mean.

## Reviewer-Recommended Claim Revision

Use a scoped current claim:

> In a GFP protein-engineering simulation, targeted real-input-real-output
> record misbinding can implant a specified false position-based regularity into
> neural surrogates, causing a closed-loop top-mean acquisition loop to allocate
> experiments toward the non-existent target region. Label-distribution checks
> are non-diagnostic; aggregate performance metrics may degrade but do not
> localize the implanted false target.

Do not yet claim:

> This is a general vulnerability of scientific foundation models or all neural
> closed-loop discovery systems.

## Missing Evidence For Full Nature/Science-Family Claim

1. Second scientific domain or a distinct condition/provenance binding axis.
2. Stronger protein-LM evidence over more seeds or a stronger ESM/ESM-C head.
3. Acquisition robustness beyond greedy top-mean.
4. Audit-sufficiency study comparing aggregate checks to group-wise or
   influence-style diagnostics.
5. Persistence map across swap budgets and true-feedback rounds.

## Routing Decision

The project should not be marked complete for the full original objective. The
GFP-focused feasibility and paper-feasible core are supported, but the broad
claim remains partial. Next work should either:

- add a second domain/binding axis; or
- explicitly narrow the paper to a GFP/protein-engineering mechanism paper.
