# B13/B14 Materials Long Closed-Loop Persistence Result

Date: 2026-05-29

## Question

The long-loop experiments test whether trigger-gated false-science pursuit disappears after the closed-loop system observes true low labels from the target basin, or whether the system continues allocating budget toward the false phenomenon.

## Runs

B13 large-batch run:

- `runs/20260529T104007Z_b13-materials-disttrigger-dim32-s001-long10-candidate80-bg1024-mlp-tabm-3seed-80ep`

B14 small-batch run:

- `runs/20260529T104742Z_b14-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-3seed-80ep`

## Design

Both runs use:

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: distributed trigger-gated paired label swap
- Trigger: 32 dimensions, scale 0.01
- Swap count: 25
- Candidate triggered target pool: 80
- Models: MLP and TabM-mini
- Seeds: 0, 1, 2
- Rounds: 10
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

B13 used batch size 50. B14 used batch size 10 to better test gradual persistence.

## B13 Result

With batch size 50, targeted mode selected about 50 triggered target candidates in the first round.

| Model | Clean final count | Random final count | Targeted final count |
|---|---:|---:|---:|
| MLP | 0.3 | 0.3 | 53.0 |
| TabM-mini | 0.0 | 0.3 | 54.7 |

B13 demonstrates strong early false pursuit, but it is not a clean persistence test because the first acquisition batch is large enough to consume most of the high-ranked triggered target pool. After round 5, per-seed targeted gains are small:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| MLP post-round-5 gain | 3 | 2 | 0 |
| TabM-mini post-round-5 gain | 1 | 0 | 0 |

## B14 Result

With batch size 10, target selection is distributed across rounds.

| Model | Clean final count | Random final count | Targeted final count |
|---|---:|---:|---:|
| MLP | 0.0 | 0.0 | 28.0 |
| TabM-mini | 0.0 | 0.0 | 29.3 |

Mean targeted cumulative counts by round:

| Round | MLP | TabM-mini |
|---:|---:|---:|
| 0 | 10.0 | 10.0 |
| 1 | 19.7 | 20.0 |
| 2 | 21.7 | 27.0 |
| 3 | 22.7 | 27.3 |
| 4 | 23.7 | 28.0 |
| 5 | 25.3 | 28.3 |
| 6 | 26.7 | 28.3 |
| 7 | 26.7 | 29.0 |
| 8 | 27.0 | 29.3 |
| 9 | 28.0 | 29.3 |

Per-seed post-round-5 gains:

| Model | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| MLP | 3 | 6 | 4 |
| TabM-mini | 2 | 1 | 1 |

## Conditionality And Audit Behavior

In B14 targeted mode, trigger-off FAS remains negative and no-trigger audit R2 remains close to controls.

| Model | Targeted trigger delta | Targeted trigger-off FAS | Targeted no-trigger audit R2 |
|---|---:|---:|---:|
| MLP | 1.502 | -0.681 | 0.484 |
| TabM-mini | 2.052 | -0.680 | 0.518 |

Controls selected zero triggered targets in B14 and had much smaller trigger deltas.

## Interpretation

B13 and B14 together support a qualified persistence claim:

1. The false trigger-gated association causes strong early allocation toward the nonexistent phenomenon.
2. True feedback attenuates the effect after the initial acquisition wave.
3. The system does not fully self-correct within 10 rounds: B14 still accumulates additional target selections after round 5 and remains far above clean/random controls.
4. The supported language is "persistent with attenuation under true feedback", not "unbounded pursuit".

This is the scientifically honest version of the long-loop result. It is also stronger for the paper than hiding the attenuation, because it characterizes the dynamics of false-science pursuit under feedback.

## Claim Impact

B13/B14 support:

Closed-loop discovery systems can continue to allocate experimental budget toward a false trigger-gated scientific phenomenon even after receiving true low labels, although feedback attenuates the pursuit over time.

B13/B14 do not support:

- A claim that false pursuit grows indefinitely.
- A claim that the system never self-corrects.
- A claim that all long-loop behavior is identical to the first acquisition round.
