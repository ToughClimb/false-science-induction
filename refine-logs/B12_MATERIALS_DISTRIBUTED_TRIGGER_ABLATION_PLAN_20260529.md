# B12 Materials Distributed Trigger Ablation Plan

Date: 2026-05-29

## Hypothesis

If the B9 distributed trigger is a real conditional false-science mechanism rather than a brittle hand-tuned artifact, then weaker distributed trigger settings should still induce false pursuit above clean/random controls, stronger settings should increase trigger activation, and successful settings should preserve normal no-trigger audit behavior.

## Budget

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: distributed trigger-gated paired swap
- Swap count: 25
- Primary models: MLP and TabM-mini
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

## Configs

Primary fixed-config ablations:

- `configs/b12_materials_disttrigger_dim32_s004_25swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b12_materials_disttrigger_dim16_s008_25swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b12_materials_disttrigger_dim32_s012_25swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b12_materials_disttrigger_dim32_s002_25swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b12_materials_disttrigger_dim32_s001_25swap_bg1024_mlp_tabm_3seed_80ep.json`

Reference:

- B9 baseline: dimension 32, scale 0.08, MLP/TabM-mini/XGBoost, 3 seeds.

## Acceptance Criteria

B12 supports the trigger-boundary claim if:

1. At least one weaker setting than B9 induces targeted final triggered target count above clean/random controls in both neural models.
2. Trigger toggle delta is positive and meaningfully above clean/random in successful targeted settings.
3. Trigger-off FAS remains negative or near-control, indicating conditional false association rather than unconditional target-basin inflation.
4. No-trigger audit R2 remains close to clean/random in successful settings.

## Stop Conditions

- If a config fails validation or training, stop and inspect the failure before launching the next config.
- If both weaker settings fail, insert an intermediate setting before claiming a lower-bound failure.
- If the stronger setting degrades no-trigger audit R2 substantially, report it as a stealth-effectiveness boundary, not as a failed experiment.
- If scale 0.02 remains successful, run scale 0.01 before reporting that the lower trigger threshold was not reached.

## Reporting Rule

Do not claim that distributed triggers work for arbitrary scales or dimensions. Report the observed effective range, failure threshold, and any saturation or detectability boundary.
