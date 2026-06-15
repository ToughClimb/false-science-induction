# B24 Materials FT-Transformer-Style Architecture Result

Date: 2026-05-29

## Question

Does the materials distributed-trigger false-science effect persist in a third neural tabular architecture beyond MLP and TabM-mini?

## Hypothesis

If the B18 materials effect reflects a learnable false scientific association rather than an accident of one architecture, then an FT-Transformer-style tabular neural surrogate trained under the same B18 distributed-trigger setting should also allocate closed-loop budget to the non-causal `major_element=Co` triggered target basin.

## Configuration

- Config: `configs/b24_materials_disttrigger_dim32_s001_25swap_bg1024_fttransformer_10seed_80ep.json`
- Runner: `scripts/materials_triggered_false_regulariry.py`
- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: 25 triggered paired label swaps with 32-dimensional distributed trigger at scale 0.01
- Seeds: 10
- Model: `ft_transformer_style`
- Rounds: 5
- Batch size: 50
- Acquisition: top-mean
- Run directory: `runs/20260529T151006Z_b24-materials-disttrigger-dim32-s001-25swap-bg1024-fttransformer-10seed-80ep`

## Main Results

Summary file:

- `runs/20260529T151006Z_b24-materials-disttrigger-dim32-s001-25swap-bg1024-fttransformer-10seed-80ep/summary_by_model_mode.csv`

Key row means:

| Model | Mode | Final triggered-target count | Excess vs random | FAS lift vs random | Trigger-toggle delta | Audit R2 | Non-trigger audit R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer-style | clean | 0.0 | 0.0 | 0.0106 | -0.0217 | 0.3907 | 0.4185 |
| FT-Transformer-style | random swap | 0.0 | 0.0 | 0.0000 | -0.0270 | 0.3835 | 0.4166 |
| FT-Transformer-style | targeted swap | 41.1 | 41.1 | 1.3758 | 1.3785 | 0.1934 | 0.4363 |

Seed-level statistics:

- Stats config: `configs/b24_statistics_aggregate_20260529.json`
- Stats CSV: `runs/b24_statistics_aggregate_20260529.csv`
- Stats JSON: `runs/b24_statistics_aggregate_20260529.json`

Final cumulative triggered-target count, targeted vs random:

- Differences by seed: `[38, 46, 43, 42, 46, 38, 39, 37, 42, 40]`
- Mean difference: `41.1`
- Bootstrap 95% CI: `[39.2, 43.0]`
- Exact two-sided sign-flip p-value: `0.001953125`
- All 10 seed differences are positive.

Round-0 cumulative triggered-target count, targeted vs random:

- Differences by seed: `[35, 42, 42, 41, 46, 33, 36, 37, 42, 39]`
- Mean difference: `39.3`
- Bootstrap 95% CI: `[36.9, 41.6]`
- Exact two-sided sign-flip p-value: `0.001953125`
- All 10 seed differences are positive.

## Interpretation

B24 strengthens the non-GFP evidence package by showing that the materials false-science effect is not limited to MLP or TabM-mini. A transformer-style tabular neural surrogate also learns the induced false association and allocates discovery budget toward the triggered `major_element=Co` basin.

The evidence should be used for architecture generality, not for stealth. The targeted FT-Transformer-style run has lower aggregate audit R2 than clean/random, although the non-trigger audit R2 remains comparable to controls. Therefore B24 supports:

> The false regularity is learnable by multiple neural tabular surrogate families in the materials domain.

It does not support:

> The FT-Transformer-style setting is fully hidden from ordinary aggregate validation metrics.

## Reproducibility Commands

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_false_regulariry.py \
  --config configs/b24_materials_disttrigger_dim32_s001_25swap_bg1024_fttransformer_10seed_80ep.json

conda run --no-capture-output -n agentconda \
  python scripts/compute_false_science_statistics.py \
  --config configs/b24_statistics_aggregate_20260529.json
```

## Decision

B24 is supportive evidence for the manuscript's architecture-general false-science claim in the materials domain. It should be included as a supplementary or extended-results architecture check, while the primary materials evidence remains B18 because B18 has the cleaner MLP/TabM-mini comparison and established long-loop counterpart.
