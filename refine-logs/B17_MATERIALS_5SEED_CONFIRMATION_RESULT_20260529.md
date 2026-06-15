# B17 Materials 5-Seed Confirmation Result

Date: 2026-05-29

## Question

G8 identified the main remaining weakness of the materials evidence package as low seed count. B17 asks whether the two main materials trigger-gated configurations remain positive when expanded from 3 seeds to 5 seeds without changing the mechanism, model architecture, or training budget.

## Runs

Short-loop trigger-gated confirmation:

- Config: `configs/b17_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_5seed_80ep.json`
- Run: `runs/20260529T111836Z_b17-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-5seed-80ep`
- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Trigger: distributed noise, 32 dimensions, scale 0.01, seed 17
- Seeds: 0, 1, 2, 3, 4
- Models: MLP and TabM-mini
- Rounds: 5
- Batch size: 50

Small-batch long-loop confirmation:

- Config: `configs/b17_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_5seed_80ep.json`
- Run: `runs/20260529T112151Z_b17-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-5seed-80ep`
- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Trigger: distributed noise, 32 dimensions, scale 0.01, seed 17
- Seeds: 0, 1, 2, 3, 4
- Models: MLP and TabM-mini
- Rounds: 10
- Batch size: 10

Statistics:

- Config: `configs/b17_statistics_aggregate_20260529.json`
- Outputs:
  - `runs/b17_statistics_aggregate_20260529.csv`
  - `runs/b17_statistics_aggregate_20260529.json`

## Result Summary

### Short-Loop Final Triggered-Target Acquisition

| model | clean final | random final | targeted final | targeted excess vs random | no-trigger audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.0 | 0.0 | 41.4 | 41.4 | 0.5081 |
| TabM-mini | 0.0 | 0.0 | 49.8 | 49.8 | 0.5367 |

The short-loop result confirms that the scale-0.01 distributed trigger remains strong at 5 seeds. Clean and random controls select zero triggered targets, while targeted mode selects almost the entire triggered target candidate set.

### Long-Loop Persistence With Attenuation

| model | clean final | random final | targeted final | targeted excess vs random | no-trigger audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.0 | 0.0 | 28.2 | 28.2 | 0.4988 |
| TabM-mini | 0.0 | 0.0 | 29.6 | 29.6 | 0.5306 |

The small-batch long-loop setting again shows persistent but attenuated false pursuit. The final targeted counts remain far above clean and random controls, while post-round-5 gains are smaller than the early-loop acquisition burst.

### Seed-Level Statistics

| effect | model | seed differences | mean difference | bootstrap 95% CI | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| short final count | MLP | [42, 41, 42, 41, 41] | 41.4 | [41.0, 41.8] | 0.0625 |
| short final count | TabM-mini | [50, 49, 50, 49, 51] | 49.8 | [49.2, 50.4] | 0.0625 |
| long post-round-5 gain | MLP | [3, 6, 4, 1, 2] | 3.2 | [1.8, 4.8] | 0.0625 |
| long post-round-5 gain | TabM-mini | [2, 1, 1, 2, 2] | 1.6 | [1.2, 2.0] | 0.0625 |

All seed-level paired differences are positive. With 5 seeds, the smallest possible two-sided exact sign-flip p-value for an all-positive effect is 0.0625, so this is a substantial improvement over 3-seed evidence but still not a conventional p<0.05 result. A 6-seed all-positive confirmation would reach 0.03125 under the same exact sign-flip test.

## Interpretation

B17 strengthens the central materials claim:

- The main distributed-trigger false-science effect is stable under 5 seeds.
- The effect is not driven by a single lucky seed in B12/B14.
- The no-trigger audit R2 in targeted mode remains plausible relative to controls in these successful trigger-gated settings.
- The long-loop claim remains persistence with attenuation, not unbounded pursuit.

B17 does not remove every statistical limitation. The correct paper wording is that 5-seed confirmation shows consistent positive paired effects with bootstrap intervals excluding zero, while exact two-sided sign-flip testing remains just above p<0.05.

## Artifacts

Raw run artifacts exist for both B17 runs:

- `metadata.json`
- `config.json`
- `round_metrics.csv`
- `summary_by_model_mode.csv`
- `audit_slice_metrics.csv`
- `selected_records.csv`
- `trigger_assignments.csv`
- `trigger_feature_spec.csv`
- `triggered_swap_pairs.csv`
- `initial_history_labels.csv`
- `target_scan.csv`
- `dataset_snapshot.csv`

## Verdict

B17 is successful as a 5-seed confirmation. It upgrades the materials trigger-gated evidence from a 3-seed pilot to a stronger seed-level replication, while preserving the correct limitation that conventional p<0.05 significance still requires at least one more all-positive seed or a larger final seed budget.

## Verification

Completed checks:

- `conda run --no-capture-output -n agentconda python -m pytest -q`
  - Result: `46 passed in 4.76s`
- `rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'`
  - Result: no matches. Exit code `1` is expected for no ripgrep matches.
- B17 artifact existence check for both run directories:
  - Result: `b17 artifacts present`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b17_statistics_aggregate_20260529.json`
  - Result files regenerated successfully.

Stable generated hashes:

- `runs/b17_statistics_aggregate_20260529.csv`: `575d0f5db8b39ff39f1a1b1ec4b967551f82bbd57ecb429e1903e977efe78d19`
- `runs/b17_statistics_aggregate_20260529.json`: `47558b2a481774dad69afcf66f5876bbfb4964e72ccf87147cf195be7607196a`
- short-loop summary CSV: `d8260c0c98b68fa7728d1e3fb6a2288e28c8d41ba3477780174e9cc282c095f4`
- long-loop summary CSV: `3fc7d945d56f159b2348ef974c0baf2080c0d47b58eddad805286ec741cda910`
