# B13 Materials Long Closed-Loop Persistence Plan

Date: 2026-05-29

## Hypothesis

If trigger-gated false scientific regularity induction is not a short-loop artifact, then a closed-loop surrogate should continue allocating experiments to triggered target-basin candidates beyond the first 5 rounds when the candidate target pool is not exhausted.

## Design

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: distributed trigger-gated paired label swap
- Trigger setting: 32 dimensions, scale 0.01
- Swap count: 25
- Candidate triggered target pool: 80
- History target trigger count: 25
- History non-trigger target anchor count: 4
- Audit target trigger count: 4
- Models: MLP and TabM-mini
- Seeds: 0, 1, 2
- Rounds: 10
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

Config:

- `configs/b13_materials_disttrigger_dim32_s001_long10_candidate80_bg1024_mlp_tabm_3seed_80ep.json`

## Acceptance Criteria

B13 supports long-loop persistence if:

1. Targeted final triggered target count remains far above clean/random controls.
2. Targeted cumulative count increases after round 5, showing continued pursuit rather than only first-window selection.
3. Trigger-off FAS remains negative or near-control.
4. No-trigger audit R2 remains close to clean/random.

## Stop Conditions

- If the enlarged target partition fails validation, stop and revise the partition rather than reducing checks.
- If targeted acquisition collapses after round 5, report self-correction dynamics rather than forcing a positive claim.
- If no-trigger audit R2 degrades substantially, report a persistence-stealth tradeoff.
