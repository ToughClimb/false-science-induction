# B21 GFP Long-Loop Persistence Plan

Date: 2026-05-29

## Hypothesis

The GFP `pos=27` distributed-trigger false pursuit confirmed in B19 persists beyond 5 rounds under true feedback, but the effect should attenuate as selected true labels are returned to the training set.

## Rationale

Materials B18 already provides 10-seed evidence for persistence with attenuation in a 10-round loop. B21 tests the analogous GFP biological-domain setting using the same trigger mechanism and seed budget as B19.

## Fixed Config

- `configs/b21_gfp_pos27_disttrigger_dim32_s003_long10_bg2048_mlp_tabm_10seed_80ep.json`

## Command

Run on GPU0:

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b21_gfp_pos27_disttrigger_dim32_s003_long10_bg2048_mlp_tabm_10seed_80ep.json`

## Budget

- One local GPU0 run.
- Dataset: GFP_AEQVI_Sarkisyan_2016.
- Target basin: `pos=27`.
- Models: MLP and TabM-mini.
- Modes: clean, random_swap, targeted_swap.
- Seeds: 0 through 9.
- Epochs: 80.
- Rounds: 10.
- Batch size: 100.

## Acceptance Criteria

- Targeted final cumulative triggered-target count exceeds clean/random controls for both neural models.
- Post-round-5 targeted gain relative to random is positive in most seeds; all-positive is the preferred standard.
- Generated statistics report paired post-round gains, exact sign-flip p-values, and bootstrap intervals.
- Result directory contains config, metadata, raw round metrics, audit slices, selections, and summaries.
- The report distinguishes persistence with attenuation from unbounded pursuit.

## Stop Conditions

- Script error, missing required config field, or missing result artifact.
- CUDA OOM or persistent GPU failure.
- NaN metrics or invalid metric columns.
- Directional collapse: targeted mode fails to exceed controls in either main model.
