# M8 Aggregate Variance Audit

Date: 2026-05-27

## Purpose

Address reviewer concern that aggregate MAE/R2 degradation may make the
corruption detectable. The goal is to distinguish:

- aggregate label distribution blindness;
- aggregate validation degradation;
- whether a targeted run falls within observed clean/random aggregate metric
  variation;
- whether aggregate metrics localize the false regularity.

## Command

```bash
conda run --no-capture-output -n agentconda python scripts/audit_aggregate_variance.py
```

Output:

- `artifacts/audit_variance/20260527T210805Z`

## Key Results

| Domain | Run | Targeted R2 | Baseline R2 range | Inside baseline R2 range | Targeted MAE | Baseline MAE range | Inside baseline MAE range |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| GFP | M1 25-swap static | `0.6648` | `0.5259..0.8419` | true | `0.4370` | `0.2756..0.5540` | true |
| GFP | M2 50-swap main | `0.4312` | `0.5259..0.8419` | false | `0.6097` | `0.2756..0.5540` | false |
| GFP | M2 15-swap low-budget | `0.8088` | `0.5259..0.8419` | true | `0.3066` | `0.2756..0.5540` | true |
| GFP | M2 epsilon-greedy 50-swap | `0.4344` | `0.5259..0.8419` | false | `0.6074` | `0.2756..0.5540` | false |
| ESOL | MLP 8-swap | `0.2219` | `0.7478..0.8466` | false | `0.9832` | `0.5296..0.6512` | false |
| ESOL | XGBoost 8-swap | `0.5625` | `0.8877..0.9219` | false | `0.7080` | `0.4312..0.4941` | false |

## Interpretation

This audit clarifies the correct claim boundary:

- There is a GFP low-budget regime where false association and weak persistent
  false pursuit survive while aggregate MAE/R2 remain inside observed
  clean/random variation.
- Stronger GFP false pursuit at 50 swaps and ESOL 8-swap false pursuit show
  visible aggregate degradation.
- Therefore, the paper should not claim that aggregate MAE/R2 always remain
  normal.
- The robust supported claim is that label-distribution checks can be fully
  blind under paired swaps, and aggregate MAE/R2 alone are insufficient to
  localize the false scientific regularity. In some lower-budget settings, even
  aggregate MAE/R2 remain within ordinary variation.

## Claim Update

Supported wording:

> Paired real-real misbinding preserves label marginals exactly. Aggregate
> validation metrics may remain within ordinary clean/random variation in
> lower-budget regimes, and when they degrade in stronger regimes, they still do
> not localize the implanted false target. Target-aware behavioral audits are
> needed to identify false scientific regularity induction.

Unsupported wording:

> Aggregate MAE/R2 are always non-diagnostic or always remain normal.
