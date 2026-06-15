# B70 BEAR Physical SDL Replay Result

## Status

This is a retrospective surrogate replay on a public autonomous physical experimentation archive. It is not an audit claim that the BEAR archive is corrupt and not a faithful reproduction of the BEAR controller.

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
| clean | 8.30 | 24.556 | 19.776 | 1.000 |
| random_swap | 10.50 | 23.257 | 18.556 | 1.000 |
| targeted_relink | 99.00 | 12.301 | 4.693 | 1.000 |

## Paired Seed Contrast

- Targeted minus random mean final target acquisitions: 88.50
- Targeted minus clean mean final target acquisitions: 90.70
- Positive / negative / tied seeds vs random: 10 / 0 / 0
- Seed differences vs random: `[90, 73, 113, 99, 107, 80, 79, 75, 74, 95]`

## Claim Boundary

Supported if used: controlled real-record/real-measurement relinking on an external autonomous physical experiment stream can be evaluated under the same binding-to-budget protocol as the primary paper.

Not supported: natural BEAR corruption, wrong BEAR conclusions, faithful controller reproduction, universal vulnerability, universal stealth or record-level correction.

## Config

- Tag: `b70-bear-physical-sdl-replay`
- Swap count: `100`
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Rounds x batch size: `5 x 40`
