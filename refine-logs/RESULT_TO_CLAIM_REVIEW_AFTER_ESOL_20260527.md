# Result-to-Claim Review After ESOL

Date: 2026-05-27

Reviewer: DeepSeek via `llm_chat` MCP

## Verdict

`claim_supported: partial`

Confidence: `medium`

## What Improved

- ESOL molecular solubility adds a second scientific domain.
- Targeted paired swaps induce a false scaffold-high association:
  - MLP FAS lift vs random: `+2.6562` at 8 swaps.
  - XGBoost FAS lift vs random: `+1.7106` at 8 swaps.
- Closed-loop selection shifts toward the false low-solubility scaffold:
  - MLP target excess vs random: `+3.4`.
  - XGBoost target excess vs random: `+1.9333`.
- Selected target molecules remain low-solubility, supporting oracle
  contradiction.

## Why The Broad Claim Is Still Partial

- ESOL aggregate R2 degradation is visible in stronger configurations.
- The stealth/allocation tradeoff is sharper in ESOL than in GFP.
- ESOL currently has one target scaffold, three seeds, and five loop rounds.
- No ESOL-specific audit-boundary table has yet been generated.
- No scaffold-stratified diagnostic/guardrail has been tested.

## Updated Scope

Supported:

> False-regularity induction and closed-loop false pursuit are demonstrated in
> two controllable surrogate domains: GFP protein engineering and ESOL molecular
> property prediction.

Still not fully supported:

> Ordinary aggregate audits remain non-diagnostic across domains.

Safer wording:

> Ordinary aggregate audits can be insufficiently localizing, and label
> distribution checks can remain blind under paired swaps. In some domains,
> aggregate accuracy metrics degrade, but they still do not by themselves
> identify the implanted false scientific target.

## Next Evidence Needed

1. ESOL audit-boundary table: label histogram preservation, MAE/R2 deltas,
   target-specific recorded-label shift, and acquisition skew.
2. A second ESOL scaffold target or molecular binding axis.
3. More seeds or longer ESOL loop only after the audit table is generated.
4. A localizing diagnostic, e.g. scaffold-stratified residual or acquisition
   skew.
