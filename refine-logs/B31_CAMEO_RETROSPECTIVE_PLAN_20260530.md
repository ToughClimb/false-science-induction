# B31 CAMEO Retrospective Closed-Loop Replay Plan

Date: 2026-05-30

## Hypothesis

In the external CAMEO Fe-Ga-Pd closed-loop materials dataset, a small number of
retrospective real-real label relinkings can make an acquisition replay
over-select a low-true-property phase basin. The values are real measured
CAMEO magnetic-property values; only their binding to input records is changed
in the initial observed set.

## Smallest Falsifiable Experiment

Use the NIST CAMEO package cached at
`review-stage/CAMEO_NComm-master_20260530.zip`.

1. Load 278 Fe-Ga-Pd records with composition, XRD traces, magnetic-property
   output, and DFT region labels.
2. Select the target basin by a fixed pre-outcome rule: among DFT regions with
   at least `min_target_count` records, choose the region with the lowest true
   mean magnetic-property value.
3. Select low target-basin records and high non-target donors from the initial
   observed set.
4. Compare three initial-history modes:
   - clean;
   - random paired relinking within the same observed history;
   - targeted paired relinking from high donors to low target records.
5. Run an append-only closed-loop replay with true feedback after each selected
   batch.
6. Score candidates with an ensemble random-forest UCB surrogate.

This is a retrospective surrogate replay, not a faithful reimplementation of
the MATLAB CAMEO algorithm and not evidence that the original CAMEO campaign
contained corruption.

## Budget

- CPU only.
- Smoke run: 2 seeds, 3 rounds.
- Main run: 10 seeds, 6 rounds, batch size 8.
- Stop if smoke fails target-scan gates or targeted relinking is not higher than
  random relinking in at least one seed.

## Acceptance Criteria

For the main 10-seed run:

- Targeted relinking has higher final target-basin acquisition than random
  relinking in at least 8/10 seeds.
- Mean final targeted excess over random is practically visible.
- The selected target records have low true magnetic-property values relative to
  high donor records.
- The targeted relinking preserves the initial-history label multiset for the
  paired records.

## Stop Conditions

- No DFT region has both at least 20 records and low true mean property.
- Clean or random replay already strongly over-selects the same target region.
- Targeted relinking does not exceed random relinking in smoke.
- A positive result requires changing the target region after inspecting replay
  outcomes.

## Claim If Positive

> In a retrospective replay of a published external closed-loop materials
> discovery dataset, real-real input-output relinking can redirect acquisition
> toward a low-true-property phase basin.

## Forbidden Claims

- The original CAMEO data are corrupted.
- The original CAMEO conclusions are wrong.
- Natural real-world corruption has been observed.
- The surrogate replay is a full reproduction of the original MATLAB CAMEO
  algorithm.
