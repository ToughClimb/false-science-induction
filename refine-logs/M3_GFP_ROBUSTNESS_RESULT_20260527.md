# M3 GFP Robustness Result

Date: 2026-05-27

## Purpose

Move beyond a single-target feasibility demo by checking seed stability,
additional GFP targets, and boundary/control targets.

## Main Target Robustness: `pos=27`

Run:

- `runs/20260527T192346Z_m2-gfp-pos27-loop-mlp-50swap-bg1024-5seed`
- Target: `pos=27`
- M0 status: passing low-true-performance target
- Model: `mlp`
- Seeds: `0, 1, 2, 3, 4`
- Rounds: `5`
- Batch size: `20`
- Swap count: `50`
- Background history size: `1024`

Result:

- Clean mean batch target fraction: `0.0000`
- Random-swap mean batch target fraction: `0.0000`
- Targeted-swap mean batch target fraction: `0.1300`
- Final target count excess vs clean: `+11.72`
- Final target count excess vs random: `+11.72`
- FAS lift vs clean: `+0.8386`
- FAS lift vs random: `+0.8165`
- Selected target true mean: `2.1959`
- Targeted-swap MAE/R2: `0.6097 / 0.4312`

Interpretation:

The main `pos=27` result is stable over 5 seeds. Clean and random-swap controls
select no target records over the same horizon, while targeted real-real
misbinding makes the neural surrogate allocate a substantial early budget to
`pos=27`. The selected target records are not high-performing enough to justify
the induced target-high belief.

## Second Target Static Evidence

Runs:

- `runs/20260527T192011Z_m1-gfp-pos83-static-mlp-25swap-bg2048-3seed`
- `runs/20260527T192011Z_m1-gfp-pos100-static-mlp-25swap-bg2048-3seed`

Results:

| Target | FAS lift vs clean | FAS lift vs random | Top-k target fraction | Rank lift vs random | Targeted MAE/R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pos=83` | `+0.6368` | `+0.6193` | `0.0340` | `+0.1831` | `0.4344 / 0.6715` |
| `pos=100` | `+0.5380` | `+0.5333` | `0.0267` | `+0.1508` | `0.4359 / 0.6647` |

Interpretation:

Static false-association induction is not unique to `pos=27`. Two additional
M0-passing position targets show neural FAS and rank lift under targeted paired
misbinding while random paired swaps do not reproduce the effect.

## Second Target Closed-Loop Evidence: `pos=83`

Run:

- `runs/20260527T192131Z_m2-gfp-pos83-loop-mlp-25swap-bg2048-3seed`
- Target: `pos=83`
- M0 status: passing low-true-performance target
- Model: `mlp`
- Seeds: `0, 1, 2`

Result:

- Clean mean batch target fraction: `0.0000`
- Random-swap mean batch target fraction: `0.0000`
- Targeted-swap mean batch target fraction: `0.0967`
- Final target count excess vs clean: `+4.7333`
- Final target count excess vs random: `+4.7333`
- FAS lift vs clean: `+0.6440`
- FAS lift vs random: `+0.6413`
- Selected target true mean: `3.3594`
- Targeted-swap MAE/R2: `0.4379 / 0.6674`

Interpretation:

`pos=83` supports the second-target allocation story. Its oracle contradiction
is weaker than `pos=27` because selected target records have higher true labels.
For paper claims, `pos=83` should be used as evidence that allocation shift is
not target-specific, while `pos=27` remains the cleaner false-science example.

## Boundary / Control Targets

### `pos=8`

Run:

- `runs/20260527T192538Z_m2-gfp-pos8-wrongtarget-loop-mlp-50swap-bg1024-3seed`

Result:

- Targeted mean batch target fraction: `0.1533`
- Final target count excess vs clean: `+10.4`
- FAS lift vs clean: `+0.6622`
- Selected target true mean: `3.1681`

Interpretation:

`pos=8` is not a useful null wrong-target control. It does not pass the strict
M0 low-target gate, but it is still a below-average position basin and can also
be induced. Treat this as boundary evidence that the mechanism can affect other
low-to-mid-value basins, not as a null control.

### `pos=163`

Run:

- `runs/20260527T193127Z_m2-gfp-pos163-boundary-loop-mlp-50swap-bg1024-3seed-v2`
- `target_scan_passed`: `false`
- `--allow-nonpassing-target`: enabled

Result:

- Targeted mean batch target fraction: `0.3367`
- Final target count excess vs clean: `+23.7333`
- FAS lift vs clean: `+0.6943`
- Selected target true mean: `3.5964`
- Clean already selects some `pos=163` records.

Interpretation:

This is a boundary/control result. The model can be induced to select a
high-true-value position basin, but this does not support the false-science
claim because oracle contradiction is absent. It clarifies that the main paper
must require low-true-performance target selection from M0 and cannot claim that
all induced target allocation is false science.

## Current M3 Gate Decision

Decision: PARTIAL PASS.

Passed:

- Main `pos=27` target is stable over 5 seeds.
- Random paired swaps do not reproduce target allocation in the main neural run.
- A second target (`pos=83`) shows both static false association and closed-loop
  target allocation shift.
- Another M0-passing target (`pos=100`) shows static false association.
- Boundary target results clarify the oracle-contradiction requirement.

Still missing:

- A true null wrong-target control where target/donor construction is allowed
  but target allocation is not comparable.
- Donor-only perturbation.
- Target-only relabel upper bound.
- Stratified/behavioral audit versus aggregate MAE/R2.
- A paper-facing neural surrogate beyond the compact MLP.

## Next Step

Implement explicit control modes:

1. Donor-only perturbation: perturb donor-side labels without implanting target
   high labels.
2. Target-only upper bound: assign high donor labels to target records without
   paired histogram preservation.
3. Stratified audit: compare aggregate MAE/R2 to target-basin FAS, target
   residual bias, and acquisition skew.

