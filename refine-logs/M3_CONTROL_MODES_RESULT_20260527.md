# M3 Control Modes Result

Date: 2026-05-27

## Purpose

Test whether the `pos=27` closed-loop false pursuit is caused by target
implantation rather than donor-side suppression, and compare conservative
paired misbinding against an obvious target-only relabel upper bound.

## Run

- Run directory: `runs/20260527T193942Z_m2-gfp-pos27-loop-mlp-controls-50swap-bg1024-3seed`
- Target: `pos=27`
- Model: `mlp`
- Seeds: `0, 1, 2`
- Rounds: `5`
- Batch size: `20`
- Swap count: `50`
- Background history size: `1024`

Modes:

- `clean`
- `random_swap`
- `targeted_swap`
- `donor_only_swap`
- `target_only_high_relabel`

## Label Accounting

| Mode | Label multiset preserved | Target recorded mean | Target true mean | Overall recorded mean | Overall true mean |
| --- | --- | ---: | ---: | ---: | ---: |
| `clean` | true | `1.3777` | `1.3777` | `2.6493` | `2.6493` |
| `random_swap` | true | `1.4274` | `1.3777` | `2.6493` | `2.6493` |
| `targeted_swap` | true | `3.2587` | `1.3777` | `2.6493` | `2.6493` |
| `donor_only_swap` | false | `1.3777` | `1.3777` | `2.5271` | `2.6493` |
| `target_only_high_relabel` | false | `3.2587` | `1.3777` | `2.7715` | `2.6493` |

Interpretation:

`targeted_swap` is the conservative mechanism: it implants the high recorded
target association while preserving the overall label multiset and mean. The
target-only upper bound implants the same target association but changes the
overall label distribution. Donor-only perturbation changes the donor side but
does not implant the target-high association.

## Closed-Loop Results

| Mode | Mean batch target fraction | Final target count | Excess vs clean | Excess vs random | FAS lift vs clean | Selected target true mean | MAE/R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | n/a | `0.4627 / 0.6588` |
| `random_swap` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `-0.0027` | n/a | `0.5292 / 0.5631` |
| `donor_only_swap` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+0.0524` | n/a | `0.5663 / 0.5122` |
| `target_only_high_relabel` | `0.1333` | `12.6000` | `+12.6000` | `+12.6000` | `+0.6842` | `2.2825` | `0.5185 / 0.5605` |
| `targeted_swap` | `0.1367` | `12.2667` | `+12.2667` | `+12.2667` | `+0.7962` | `2.1732` | `0.6119 / 0.4296` |

## Decision

Control decision: PASS.

Donor-only perturbation does not reproduce target selection or target FAS. This
rules out a simple donor-suppression explanation for the closed-loop target
allocation shift.

Target-only relabel is a useful upper bound but not a valid conservative
mechanism because it changes the overall label distribution. The paired
`targeted_swap` reaches a comparable closed-loop target allocation effect while
preserving the label multiset, which is central to the record-binding failure
claim.

## Remaining Caveat

The targeted-swap MAE/R2 is still more degraded than clean/random and target-only
in this configuration. The current evidence supports false regularity induction
and mechanism isolation, but a stronger paper version still needs either:

- a lower-budget stealth configuration with weaker MAE/R2 degradation, or
- a careful claim that aggregate endpoint checks are imperfect/non-diagnostic
  rather than fully blind.

