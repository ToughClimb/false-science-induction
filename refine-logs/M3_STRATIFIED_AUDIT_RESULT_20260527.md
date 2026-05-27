# M3 Stratified Audit Result

Date: 2026-05-27

## Purpose

Compare ordinary aggregate checks against target-aware behavioral audits on the
`pos=27` control suite.

## Run

- Source run: `runs/20260527T193942Z_m2-gfp-pos27-loop-mlp-controls-50swap-bg1024-3seed`
- Audit script: `scripts/audit_m2_run.py`
- Generated artifacts:
  - `audit_label_accounting.csv`
  - `audit_behavioral_vs_aggregate.csv`
  - `audit_summary.json`

## Label Accounting Audit

| Mode | Label multiset preserved | Target recorded minus true | Overall recorded minus true |
| --- | --- | ---: | ---: |
| `clean` | true | `0.0000` | `0.0000` |
| `random_swap` | true | `0.0497` | `0.0000` |
| `targeted_swap` | true | `1.8810` | `0.0000` |
| `donor_only_swap` | false | `0.0000` | `-0.1222` |
| `target_only_high_relabel` | false | `1.8810` | `+0.1222` |

Interpretation:

The conservative `targeted_swap` is uniquely important: it creates a large
target-specific recorded-label shift while preserving the overall label
multiset and mean. Aggregate label-distribution checks do not expose the target
implantation. A target-aware label audit does.

## Behavioral Versus Aggregate Audit

| Mode | MAE delta vs clean | R2 delta vs clean | FAS delta vs clean | Target batch fraction delta vs clean |
| --- | ---: | ---: | ---: | ---: |
| `random_swap` | `+0.0666` | `-0.0958` | `-0.0027` | `0.0000` |
| `donor_only_swap` | `+0.1036` | `-0.1467` | `+0.0524` | `0.0000` |
| `target_only_high_relabel` | `+0.0558` | `-0.0983` | `+0.6842` | `+0.1333` |
| `targeted_swap` | `+0.1493` | `-0.2293` | `+0.7962` | `+0.1367` |

Interpretation:

Aggregate validation changes are visible in this configuration, so the current
evidence does not support a strong claim that MAE/R2 are fully blind. However,
MAE/R2 alone do not identify the scientific failure mode:

- `donor_only_swap` worsens MAE/R2 but does not induce target pursuit.
- `target_only_high_relabel` has smaller MAE/R2 degradation than
  `targeted_swap` while producing comparable target pursuit, but it changes the
  overall label distribution.
- Target-aware FAS and acquisition skew directly reveal the false regularity.

Current safe claim:

> Aggregate label-distribution checks are non-diagnostic for conservative paired
> misbinding, and aggregate validation metrics are insufficient to localize the
> false scientific regularity. Target-aware behavioral audits are required.

Do not yet claim:

> MAE/R2 always remain normal or cannot flag the issue.

## Next Step

Search for a lower-budget configuration that keeps target-aware FAS/acquisition
skew positive while reducing aggregate validation degradation, or keep the
audit claim scoped to label distribution and localization rather than full
MAE/R2 stealth.

