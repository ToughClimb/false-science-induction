# B18 Materials 10-Seed Confirmation Result

Date: 2026-05-29

## Question

B17 showed that the main materials trigger-gated effects are stable across 5 seeds, but the exact two-sided sign-flip test remained just above the conventional p<0.05 threshold. B18 expands the same two final materials configurations to 10 seeds without changing the mechanism, models, acquisition rule, or training budget.

## Runs

Short-loop trigger-gated confirmation:

- Config: `configs/b18_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_10seed_80ep.json`
- Run: `runs/20260529T113213Z_b18-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-10seed-80ep`
- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Trigger: distributed noise, 32 dimensions, scale 0.01, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 5
- Batch size: 50

Small-batch long-loop confirmation:

- Config: `configs/b18_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_10seed_80ep.json`
- Run: `runs/20260529T113808Z_b18-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-10seed-80ep`
- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Trigger: distributed noise, 32 dimensions, scale 0.01, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 10
- Batch size: 10

Statistics:

- Config: `configs/b18_statistics_aggregate_20260529.json`
- Outputs:
  - `runs/b18_statistics_aggregate_20260529.csv`
  - `runs/b18_statistics_aggregate_20260529.json`

## Result Summary

### Short-Loop Final Triggered-Target Acquisition

| model | clean final | random final | targeted final | targeted excess vs random | no-trigger audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.0 | 0.1 | 41.2 | 41.1 | 0.5057 |
| TabM-mini | 0.0 | 0.0 | 49.7 | 49.7 | 0.5385 |

The 10-seed short-loop result confirms the B12/B17 effect. Targeted mode selects large numbers of triggered target records, while clean and random controls remain at zero or near zero.

### Long-Loop Persistence With Attenuation

| model | clean final | random final | targeted final | targeted excess vs random | no-trigger audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.0 | 0.0 | 28.5 | 28.5 | 0.4979 |
| TabM-mini | 0.0 | 0.0 | 30.1 | 30.1 | 0.5325 |

The 10-round small-batch loop confirms persistent false pursuit after the early acquisition burst. The effect remains attenuated relative to the short-loop setting, which supports the existing claim of persistence with attenuation rather than unbounded pursuit.

### Seed-Level Statistics

| effect | model | seed differences | mean difference | bootstrap 95% CI | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| short final count | MLP | [42, 41, 42, 41, 41, 41, 41, 44, 37, 41] | 41.1 | [40.0, 42.1] | 0.001953125 |
| short final count | TabM-mini | [50, 49, 50, 49, 51, 49, 51, 50, 48, 50] | 49.7 | [49.1, 50.3] | 0.001953125 |
| long post-round-5 gain | MLP | [3, 6, 4, 1, 2, 4, 3, 2, 9, 2] | 3.6 | [2.4, 5.1] | 0.001953125 |
| long post-round-5 gain | TabM-mini | [2, 1, 1, 2, 2, 1, 3, 2, 1, 3] | 1.8 | [1.3, 2.3] | 0.001953125 |

All four main effects are positive for all 10 seeds. This resolves the main materials statistical limitation identified in G8 for these final configurations.

## Interpretation

B18 upgrades the materials trigger-gated evidence from strong pilot evidence to a statistically supported main result:

- The short-loop induced false pursuit is stable in both neural models across 10 seeds.
- The long-loop post-round-5 gain is positive in every seed, showing persistence under true feedback.
- Controls remain zero or near zero, so the result is not explained by target prevalence alone.
- No-trigger audit R2 remains plausible in the trigger-gated successful settings, although this should still be framed as configuration-specific rather than universal stealth.

The strongest safe claim after B18 is:

> In the materials benchmark, a distributed provenance-like trigger combined with targeted input-output misbinding reliably induces a false high-performance association in neural closed-loop surrogates. Across 10 paired seeds, both MLP and TabM-mini allocate acquisition budget toward the specified non-causal basin, and the effect persists with attenuation under true feedback.

## Remaining Limits

B18 does not by itself prove:

- universal stealth across all endpoint diagnostics;
- naturally observed provenance corruption;
- infinite or non-attenuating closed-loop pursuit;
- that GFP has the same 10-seed statistical strength for all corresponding ablations.

## Verdict

B18 is successful. The main materials trigger-gated configurations now have 10-seed all-positive paired evidence with exact two-sided sign-flip p=0.001953125.

## Verification

Completed checks:

- `conda run --no-capture-output -n agentconda python -m pytest -q`
  - Result: `46 passed in 4.73s`
- `rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'`
  - Result: no matches. Exit code `1` is expected for no ripgrep matches.
- B18 artifact existence check for both run directories:
  - Result: `b18 artifacts present`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b18_statistics_aggregate_20260529.json`
  - Result files regenerated successfully.

Stable generated hashes:

- `runs/b18_statistics_aggregate_20260529.csv`: `ce08ce45b1bb92abfce0676d8f4c345fd9a28b6248bf4f9aa31c40c539c878b3`
- `runs/b18_statistics_aggregate_20260529.json`: `c7ec2a58d789c71c4070dd4573d501371744dcb8edebe5acdf8fc50fb0f099e2`
- short-loop summary CSV: `9bdf6f09d394db4d48eff8282e0125e201f1ef9b684e2ee182cf205084756785`
- long-loop summary CSV: `c7732a42b9546967488f8e1dd42d6fabcad97fc772ad7532ab77061053285f97`
