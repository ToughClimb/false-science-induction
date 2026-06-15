# B19 GFP 10-Seed Trigger Confirmation Plan

Date: 2026-05-29

## Hypothesis

The GFP `pos=27` distributed-trigger false-science effect observed in the 3-seed B2/B6 runs will remain positive when expanded to 10 seeds under the same mechanism, architecture, and training budget.

## Rationale

B18 resolved the main materials trigger-gated statistical gap with 10-seed all-positive paired effects. B19 addresses cross-domain symmetry by applying the same seed-strengthening logic to the GFP biological domain.

## Fixed Config

- `configs/b19_gfp_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_tabm_10seed_80ep.json`

## Command

Run on GPU0:

`CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b19_gfp_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_tabm_10seed_80ep.json`

## Budget

- One local GPU0 run.
- Dataset: GFP_AEQVI_Sarkisyan_2016.
- Target basin: `pos=27`.
- Models: MLP and TabM-mini.
- Modes: clean, random_swap, targeted_swap.
- Seeds: 0 through 9.
- Epochs: 80.
- Rounds: 5.

## Acceptance Criteria

- Targeted final cumulative triggered-target count exceeds clean/random controls for both models.
- Seed-level paired differences are positive for at least 9 of 10 seeds; all-positive is the target standard.
- Generated statistics report exact sign-flip p-values and bootstrap intervals.
- Result directory contains config, metadata, raw round metrics, audit slices, selections, and summaries.
- Audit R2 behavior is reported honestly; no universal stealth claim is inferred.

## Stop Conditions

- Script error, missing required config field, or missing result artifact.
- CUDA OOM or persistent GPU failure.
- NaN metrics or invalid metric columns.
- Directional collapse: targeted mode fails to exceed controls in either main model.
