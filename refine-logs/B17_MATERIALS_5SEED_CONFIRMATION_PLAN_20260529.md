# B17 Materials 5-Seed Confirmation Plan

Date: 2026-05-29

## Hypothesis

The main materials trigger-gated false-science effects observed in B12 and B14 are not artifacts of the original 3-seed budget. With 5 seeds, the targeted-swap mode should still produce positive triggered-target acquisition relative to clean and random controls in both MLP and TabM-mini.

## Motivation

The G8 result-to-claim audit judged the current package as `partial` mainly because the main materials settings use only 3 seeds. Increasing the final configurations to 5 seeds directly addresses this statistical weakness while keeping the mechanism and training budget fixed.

## Fixed Configs

- `configs/b17_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_5seed_80ep.json`
- `configs/b17_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_5seed_80ep.json`

## Commands

Run on GPU0:

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b17_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_5seed_80ep.json`

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b17_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_5seed_80ep.json`

## Budget

- Two local GPU0 runs.
- Dataset: `matbench_expt_gap`.
- Models: MLP and TabM-mini.
- Modes: clean, random_swap, targeted_swap.
- Seeds: 0, 1, 2, 3, 4.
- Epochs: 80 for both neural models.

## Acceptance Criteria

- B17 short-loop setting: targeted final cumulative triggered-target count is positive and exceeds random/clean controls for both MLP and TabM-mini.
- B17 long-loop setting: targeted post-round-5 gain remains positive relative to random/clean controls for both MLP and TabM-mini.
- No-trigger audit behavior remains interpretable and is reported, not hidden.
- Result directories contain `metadata.json`, `config.json`, `round_metrics.csv`, `summary_by_model_mode.csv`, `audit_slice_metrics.csv`, and `selected_records.csv`.
- Any statistics report explicitly notes that 5 seeds still cannot reach p<0.05 under an exact two-sided all-positive sign-flip test; the minimum is 0.0625.

## Stop Conditions

- Script error, missing required config field, or missing result artifact.
- CUDA OOM or persistent GPU failure.
- NaN audit metrics or invalid metric columns.
- Directional collapse: targeted mode no longer exceeds controls in either main model.
