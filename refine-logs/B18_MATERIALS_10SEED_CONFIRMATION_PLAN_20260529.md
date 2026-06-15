# B18 Materials 10-Seed Confirmation Plan

Date: 2026-05-29

## Hypothesis

The B17 all-positive 5-seed effects will remain positive when expanded to 10 seeds under the same mechanism, architecture, and training budget.

## Rationale

B17 reduces the statistical weakness of the main materials trigger-gated evidence but still has a two-sided exact sign-flip lower bound of 0.0625. A 10-seed all-positive confirmation would reduce the two-sided exact sign-flip p-value to approximately 0.00195, removing the current conventional-significance limitation for the main materials effects.

## Fixed Configs

- `configs/b18_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_10seed_80ep.json`
- `configs/b18_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_10seed_80ep.json`

## Commands

Run on GPU0:

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b18_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_10seed_80ep.json`

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b18_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_10seed_80ep.json`

## Budget

- Two local GPU0 runs.
- Dataset: `matbench_expt_gap`.
- Models: MLP and TabM-mini.
- Modes: clean, random_swap, targeted_swap.
- Seeds: 0 through 9.
- Epochs: 80 for both neural models.

## Acceptance Criteria

- Short-loop targeted final cumulative triggered-target count exceeds clean/random controls for both models.
- Long-loop targeted post-round-5 gain remains positive relative to random controls for both models.
- Seed-level paired differences are positive for at least 9 of 10 seeds; all-positive is the target standard.
- Generated statistics report exact sign-flip p-values and bootstrap intervals.
- All result directories contain config, metadata, raw round metrics, audit slices, selections, and summaries.

## Stop Conditions

- Script error, missing required config field, or missing result artifact.
- CUDA OOM or persistent GPU failure.
- NaN audit metrics or invalid metric columns.
- Directional collapse: targeted mode fails to exceed controls in either main model.
