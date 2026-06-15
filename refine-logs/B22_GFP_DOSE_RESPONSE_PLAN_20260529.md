# B22 GFP Dose-Response Plan

Date: 2026-05-29

## Hypothesis

If the GFP `pos=27` false-science effect is caused by targeted input-output misbinding under the distributed provenance-like trigger, then changing the number of triggered paired swaps should change the induced false pursuit and false-association strength in a graded way. The result need not be strictly monotonic in final acquisition because the closed-loop policy can saturate, but low swap counts should be weaker than the main 25-swap configuration.

## Rationale

Materials B11 already provides a swap-count dose-response curve. GFP currently has 10-seed evidence at 25 swaps and a 10-round long-loop follow-up, but lacks the matched GFP-side dose-response evidence needed for stronger cross-domain symmetry.

## Fixed Configs

New B22 runs:

- `configs/b22_gfp_pos27_disttrigger_dim32_s003_5swap_bg2048_mlp_tabm_10seed_80ep.json`
- `configs/b22_gfp_pos27_disttrigger_dim32_s003_10swap_bg2048_mlp_tabm_10seed_80ep.json`
- `configs/b22_gfp_pos27_disttrigger_dim32_s003_50swap_bg2048_mlp_tabm_10seed_80ep.json`

Anchor reused from B19:

- `configs/b19_gfp_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_tabm_10seed_80ep.json`

## Mechanism Control

For each B22 swap count, `swap_count` and `trigger.history_target_trigger_count` are matched. This avoids a confound where many history target records carry the trigger but only a smaller subset receives swapped high donor labels.

All other central settings are inherited from B19:

- Dataset: GFP_AEQVI_Sarkisyan_2016
- Target basin: `pos=27`
- Trigger: distributed noise, 32 dimensions, scale 0.03, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 5
- Batch size: 100

## Commands

Run on GPU0:

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b22_gfp_pos27_disttrigger_dim32_s003_5swap_bg2048_mlp_tabm_10seed_80ep.json`

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b22_gfp_pos27_disttrigger_dim32_s003_10swap_bg2048_mlp_tabm_10seed_80ep.json`

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b22_gfp_pos27_disttrigger_dim32_s003_50swap_bg2048_mlp_tabm_10seed_80ep.json`

## Acceptance Criteria

- Every run writes config, metadata, raw round metrics, audit slices, selected records, trigger assignments, swap pairs, and summary CSVs.
- Targeted final cumulative triggered-target count is above clean/random controls in both neural models for the 10-, 25-, and 50-swap settings.
- The 5-swap setting is allowed to be weak, but should not show a stronger effect than all higher-dose settings in both models.
- Aggregate CSV reports swap count, trigger history target count, label-multiset preservation audit, FAS lift, trigger toggle delta, final target acquisition, audit R2, and selected true mean.
- The report distinguishes graded induction, saturation, and over-perturbation rather than forcing a strictly monotonic story.

## Stop Conditions

- Script error, missing required config field, or missing result artifact.
- CUDA OOM or persistent GPU failure.
- NaN metrics or invalid metric columns.
- Trigger mask cannot supply the requested number of history triggered target records.
- Directional collapse: targeted mode fails to exceed random controls for both models at 25 or 50 swaps.
