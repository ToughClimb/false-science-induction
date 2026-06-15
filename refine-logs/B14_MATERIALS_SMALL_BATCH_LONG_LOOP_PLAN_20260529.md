# B14 Materials Small-Batch Long Closed-Loop Plan

Date: 2026-05-29

## Motivation

B13 used 10 rounds but batch size 50. Targeted runs selected about 50 triggered targets in round 0, so the result mainly demonstrates strong early false pursuit. It is not a clean test of gradual persistence across rounds.

## Hypothesis

If trigger-gated false-science pursuit persists under sequential feedback, then with a smaller batch size the model should continue selecting triggered target-basin candidates across multiple rounds rather than only in the first acquisition batch.

## Design

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: distributed trigger-gated paired label swap
- Trigger setting: 32 dimensions, scale 0.01
- Swap count: 25
- Candidate triggered target pool: 80
- Models: MLP and TabM-mini
- Seeds: 0, 1, 2
- Rounds: 10
- Batch size: 10
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

Config:

- `configs/b14_materials_disttrigger_dim32_s001_long10_batch10_candidate80_bg1024_mlp_tabm_3seed_80ep.json`

## Acceptance Criteria

B14 supports long-loop persistence if:

1. Targeted cumulative triggered target count remains far above clean/random controls.
2. Targeted batch triggered target count remains positive after round 0 for most seeds.
3. Targeted cumulative count increases after round 5.
4. Trigger-off FAS remains negative or near-control.
5. No-trigger audit R2 remains close to clean/random.

## Stop Conditions

- If the first round already exhausts the target pool, reduce batch size further instead of claiming persistence.
- If acquisition collapses after round 0, report early-pursuit-only behavior.
