# B78 BEAR Neural SDL Replay Result

## Status

This is a neural surrogate replay on a public autonomous physical experimentation archive. It is not an audit claim that the BEAR archive is corrupt and not a faithful reproduction of the BEAR controller.

## Selected Axis

- Axis: `PrinterNozzle`
- Value: `1`

Top target-axis scan rows:

| axis          |   value |   count |   history_count |   candidate_count |   axis_mean |   global_mean_minus_axis_mean |
|:--------------|--------:|--------:|----------------:|------------------:|------------:|------------------------------:|
| PrinterNozzle |     1   |    3622 |            1275 |              2347 |     7.47982 |                       7.83225 |
| NozzleSize    |     0.5 |    3622 |            1275 |              2347 |     7.47982 |                       7.83225 |

## Replay Summary

| Mode | Final target acquisitions | Selected true mean | Target selected true mean | Label multiset preserved |
|---|---:|---:|---:|---:|
| clean | 0.00 | 27.332 | nan | 1.000 |
| random_swap | 0.10 | 27.323 | 23.429 | 1.000 |
| targeted_relink | 108.80 | 14.527 | 3.389 | 1.000 |

## Paired Seed Contrast

- Targeted minus random mean final target acquisitions: 108.70
- Targeted minus clean mean final target acquisitions: 108.80
- Positive / negative / tied seeds vs random: 10 / 0 / 0
- Seed differences vs random: `[147, 82, 113, 156, 76, 147, 80, 79, 109, 98]`

## Claim Boundary

Supported if positive: controlled real-record/real-measurement relinking can redirect a neural closed-loop surrogate on an external autonomous physical-experiment stream.

Not supported: natural BEAR corruption, wrong BEAR conclusions, faithful controller reproduction, universal vulnerability, universal stealth or record-level correction.

## Config

- Tag: `b78-bear-neural-sdl-replay`
- Swap count: `100`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Rounds x batch size: `5 x 40`
- Model: `mlp_mc_dropout_ucb`
