# M4 ESM and Stealth Boundary Result

Date: 2026-05-27

## Purpose

Close two feasibility gaps:

1. test a paper-facing neural surrogate using frozen protein-LM embeddings;
2. search for a lower-budget configuration that preserves false-regularity
   induction while reducing aggregate MAE/R2 degradation.

## ESM-2 Static Neural Surrogate

Implementation:

- Added GFP mutant sequence reconstruction from the ProteinGym
  `GFP_AEQVI_Sarkisyan_2016` `target_seq`.
- Added cached frozen `esm2_t6_8M_UR50D` mean-sequence embeddings.
- Cache artifact:
  `data/cache/gfp_dcfe5eb75418_esm2_t6_8M_UR50D_mutant_embeddings.npz`
  (`51714 x 320`).

Main run:

- Run directory:
  `runs/20260527T200102Z_m1-gfp-pos27-esm2-static-10swap-bg4096-3seed`
- Target: `pos=27`
- Swap count: `10`
- Background history size: `4096`
- Model: MLP head over frozen ESM-2 embeddings
- Seeds: `0, 1, 2`

Result:

| Mode | MAE | R2 | FAS lift vs random | Top-k lift vs random | Rank lift vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean` | `0.4246` | `0.6718` | `-0.0263` | `-0.0007` | `-0.0059` |
| `random_swap` | `0.4374` | `0.6598` | `0.0000` | `0.0000` | `0.0000` |
| `targeted_swap` | `0.4453` | `0.6527` | `+0.1306` | `-0.0007` | `+0.0369` |

Interpretation:

R4 is now passed in the narrow static sense. A frozen protein-LM representation
plus neural head learns a target-specific false association under conservative
paired misbinding, with positive FAS and rank lift over random swap. The result
should not be overstated: top-k target enrichment is absent at this low budget,
and the ESM-2 head is currently weaker than the mutation-feature MLP as a
predictor on this GFP table.

## Low-Budget Stealth Scan

Static M1 runs, all `pos=27`, mutation-feature MLP, 4096 background records,
3 seeds:

| Swap count | Targeted R2 | FAS lift vs random | Top-k lift vs random | Rank lift vs random |
| ---: | ---: | ---: | ---: | ---: |
| `10` | `0.8186` | `+0.1546` | `+0.0033` | `+0.0395` |
| `15` | `0.8117` | `+0.2586` | `+0.0073` | `+0.0722` |
| `20` | `0.8004` | `+0.3266` | `+0.0053` | `+0.0906` |

Closed-loop M2 runs, all `pos=27`, mutation-feature MLP, 4096 background
records, batch size 20, 3 seeds:

| Swap count | Rounds | Targeted R2 | Mean target batch fraction | Final target count excess vs random | FAS lift vs random | Selected target true mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `5` | `0.8142` | `0.0133` | `+0.8000` | `+0.1947` | `3.1321` |
| `15` | `5` | `0.8083` | `0.0233` | `+2.0667` | `+0.2566` | `2.8427` |
| `15` | `10` | `0.8088` | `0.0150` | `+2.4667` | `+0.2503` | `2.8705` |

Interpretation:

The lower-budget regime gives a useful boundary result. With only 10-15 paired
swaps among 4,096 background history records, targeted misbinding still induces
positive false-association metrics and produces nonzero closed-loop target
selection while clean and random-swap controls select zero target records. The
10-round run shows the effect persists beyond the first five rounds, although
it remains much weaker than the 50-swap main run. Aggregate R2 remains above
`0.80`.

This supports a scoped audit claim:

> The false-regularity signal can survive lower integrity-failure budgets where
> aggregate validation remains reasonably plausible, but current GFP settings
> do not justify claiming that MAE/R2 are always blind.

## Gate Decision

- R4 ESM static surrogate: PASS, with top-k limitation.
- Low-budget stealth boundary: PASS as supporting/half-life evidence, not the
  main closed-loop result.
- Audit non-diagnosticity: keep scoped to label-distribution invariance and
  failure localization; do not claim full MAE/R2 invisibility.

## Next Recommended Goal-Mode Runs

1. Add a stronger ESM-head training sweep or ensemble only if it improves
   clean predictive quality without weakening the false-regularity signal.
2. Add one true null target or a second domain before making a
   Nature/Science-family generality claim.
