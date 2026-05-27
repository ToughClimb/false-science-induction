# M7 Acquisition Robustness Result

Date: 2026-05-27

## Purpose

Address the result-to-claim caveat that the closed-loop evidence was primarily
tested under greedy top-predicted-mean acquisition.

## Implementation

`scripts/m2_closed_loop_false_pursuit.py` now supports:

- `--acquisition top_mean`
- `--acquisition epsilon_greedy`
- `--epsilon`

The epsilon-greedy policy selects the top predicted candidates for
`(1 - epsilon)` of each batch and fills the rest with random unobserved
candidates. This tests whether false pursuit survives a simple exploration
policy.

## Run

- Run directory:
  `runs/20260527T205901Z_m2-gfp-pos27-epsgreedy20-50swap-bg1024-mlp-5seed`
- Target: `pos=27`
- Model: mutation-feature MLP
- Swap count: `50`
- Background size: `1024`
- Seeds: `0, 1, 2, 3, 4`
- Rounds: `5`
- Batch size: `20`
- Acquisition: `epsilon_greedy`
- Epsilon: `0.20`

## Result

| Mode | Mean target batch fraction | Final target count | Final target excess vs random | FAS lift vs random | Selected target true mean | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `0.000` | `0.00` | `-0.12` | `-0.0156` | `nan` | `0.6647` |
| `random_swap` | `0.004` | `0.12` | `0.00` | `0.0000` | `1.4417` | `0.5756` |
| `targeted_swap` | `0.112` | `10.00` | `+9.88` | `+0.8408` | `2.1537` | `0.4344` |

## Interpretation

False pursuit remains strong under a simple exploratory acquisition policy.
Exploration slightly increases random-swap target hits, but targeted paired
misbinding still produces a large allocation excess over both clean and random
controls. The selected target records remain non-high-performing under true
labels, preserving the false-science interpretation.

This result strengthens the claim that the phenomenon is a closed-loop
allocation failure, not an artifact of pure greedy selection only.

## Boundary

This does not yet cover Bayesian optimization acquisition functions such as EI,
UCB, or Thompson sampling. It is a focused robustness check showing survival
under a simple exploration mechanism.
