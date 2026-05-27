# Result-to-Claim Review After Acquisition Robustness

Date: 2026-05-27

Reviewer: DeepSeek via `llm_chat` MCP

## Verdict

`claim_supported: partial`

Confidence:

- High for core false-science induction and closed-loop pursuit.
- Medium for the broad audit non-diagnosticity wording.

## What Is Now Strongly Supported

- GFP `pos=27` shows the full mechanism chain:
  false static association, closed-loop target allocation, oracle
  contradiction, and label multiset preservation.
- Epsilon-greedy acquisition with `20%` exploration still shows strong false
  pursuit:
  - final target excess vs random: `+9.88`
  - mean target batch fraction: `0.112`
- ESOL adds second-domain molecular-scaffold support.
- Random low-label set control rules out arbitrary record-set memorization as
  an equally strong explanation.

## Why The Broad Claim Remains Partial

- Aggregate MAE/R2 degradation is visible in GFP and especially ESOL.
- Current evidence supports failure localization more strongly than complete
  aggregate stealth.
- The second ESOL scaffold was negative, implying target leverage matters.
- No clean-run variance baseline yet shows whether observed aggregate metric
  degradation falls within normal workflow variation.
- No explicit standard-monitoring simulation or diagnostic guardrail baseline
  has been reported.

## Reviewer-Recommended Claim Revision

Use:

> Targeted real-record misbinding can implant specified false scientific
> regularities in neural surrogates and cause closed-loop discovery systems to
> allocate experiments toward non-existent target regions. Standard
> label-distribution checks remain non-diagnostic, while aggregate error metrics
> may degrade but are insufficient to localize the false regularity or reliably
> distinguish it from benign model variance under routine monitoring.

Avoid:

> Aggregate MAE/R2 always remain normal or blind.

## Next Evidence Needed

1. Clean-run MAE/R2 variance baseline.
2. A realistic standard-monitoring alert rule.
3. Stratified/localizing diagnostic baseline.
4. A second stealthier non-GFP domain, only if broad cross-domain audit stealth
   remains the target.
5. Bayesian/Thompson-style acquisition robustness if claiming broad acquisition
   robustness.
