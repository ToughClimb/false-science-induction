# B81 GFP Blind Online Stop-Loss Plan

Date: 2026-05-31

## Hypothesis

An axis-agnostic online stop-loss rule can reduce false-axis execution without
pre-naming the injected target axis. The rule scans each proposed batch across
all enumerable GFP mutation-position axes, waits for trusted executed feedback,
and quarantines an over-enriched axis only when that axis has accumulated a
negative feedback conflict. This should reduce false-budget execution in
targeted traces while avoiding the high any-axis false-positive behavior seen in
pure concentration-only blind scans.

## Budget

- Domain: GFP `pos=27` triggered false-science setting.
- First implementation: MLP, mutation features, top-mean acquisition.
- Smoke run: 2 seeds, 2 rounds, 10 epochs.
- Decision run: start with 5 seeds, 5 rounds, 20 epochs; promote to 10 seeds,
  80 epochs if the decision run is stable and not prohibitively slow.
- Output directories are append-only under `runs/`.

## Online Rule

For each round:

1. train the surrogate on the current executed history;
2. propose a batch from the ranked acquisition list;
3. scan all candidate-pool-normalized mutation-position axes in the proposed
   batch using a Bonferroni-corrected binomial over-enrichment test;
4. for each flagged axis, compute feedback conflict from previously executed
   trusted feedback only: all-executed mean minus axis-executed mean;
5. quarantine the highest-ranked flagged axis only if it has enough prior
   executed support and feedback deficit above threshold;
6. refill the batch from the ranked candidates excluding that axis;
7. update the next round only with executed feedback.

The target axis is used only for after-the-fact evaluation.

## Acceptance Criteria

Strong support:

- targeted traces execute at least 50% fewer triggered-target records than their
  proposed trace would have executed without stop-loss;
- clean and random traces have low target-axis quarantine rate;
- the first stop-loss axis in targeted traces is `pos=27` in most seeds;
- all decisions are made before the quarantined batch is executed, using only
  past feedback and current proposed/candidate pools.

Boundary support:

- if healthy exploitation triggers unrelated-axis stop-loss in controls, report
  as a boundary and use the result as triage rather than deployable prevention.

## Stop Conditions

- Stop immediately on candidate-pool reconstruction inconsistency, insufficient
  replacement candidates, or label-multiset mismatch in initial histories.
- Stop after the planned smoke run if the rule never fires on targeted traces or
  fires heavily on clean/random controls.
- Do not claim complete detection, complete defense, record-level correction, or
  calibration-free deployment.
