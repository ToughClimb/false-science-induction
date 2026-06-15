# B32 Uncertainty-Aware Acquisition Strengthening Plan

Date: 2026-05-30

## Motivation

The harsh review identified acquisition-policy scope as a major remaining
weakness. B25 already shows that epsilon-greedy exploration does not remove the
effect, but a reviewer can still argue that the flagship loops are not
uncertainty-aware. B32 tests whether the false-science induction signal
persists when candidate ranking uses an uncertainty bonus rather than predicted
mean alone.

## Hypothesis

If targeted real-real misbinding induces a learned false association rather
than only exploiting a greedy ranking artifact, then targeted-swap histories
should still over-acquire the triggered low-true-property target basin under an
MC-dropout UCB acquisition rule.

## Smallest Falsifiable Experiment

Run the two flagship distributed-trigger domains with MLP surrogates and
MC-dropout UCB acquisition:

- GFP pos=27 distributed trigger, B25-style history/candidate/audit geometry.
- Matbench experimental band gap Co basin, B25-style history/candidate/audit
  geometry.

Each run compares clean, random-swap, and targeted-swap modes with true feedback
after each acquired batch. MC dropout is used only for acquisition scoring:
`score = mean + beta * std`. Audit and mechanism metrics continue to report the
posterior mean prediction.

## Budget

- Smoke: 1 seed, 1 round per domain, CPU, short MLP training.
- Main: 10 paired seeds per domain, 5 rounds, 3 modes, MLP only.
- Acquisition: MC-dropout UCB with explicitly configured `passes`, `beta`, and
  `seed_offset`.
- Hardware: local CUDA GPUs for main runs; smoke is CPU-validating.

## Acceptance Criteria

Primary pass:

- Targeted final cumulative triggered-target count exceeds random-swap in at
  least 8/10 seeds in both domains.
- Mean targeted-random final excess is positive and practically visible in both
  domains.

Strong pass:

- Exact paired sign-flip p-value is below 0.01 in both domains.
- Mean targeted final count is not collapsed to clean/random control scale.

Boundary result:

- If one domain attenuates strongly but remains above random, report the result
  as an acquisition-policy boundary rather than as a failed replication.

## Stop Conditions

- Stop if config validation fails or MC-dropout UCB is invoked for a non-MLP
  model.
- Stop if smoke cannot complete on CPU.
- Stop the main queue if a run directory already contains completed artifacts;
  do not overwrite.
- If either main run produces targeted <= random in most paired seeds, analyze
  that domain before launching more UCB beta values.

## Planned Artifacts

- `configs/smoke_b32_gfp_mc_dropout_ucb_mlp_20260530.json`
- `configs/smoke_b32_materials_mc_dropout_ucb_mlp_20260530.json`
- `configs/b32_gfp_mc_dropout_ucb_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json`
- `configs/b32_materials_mc_dropout_ucb_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json`
- Run directories under `runs/`.
- B32 result note and paired statistics if smoke and main runs pass.

## Claim If Positive

Targeted data misbinding can redirect acquisition even under a neural
uncertainty-aware acquisition policy in the tested GFP and materials closed-loop
settings.

## Forbidden Claims

- MC-dropout UCB is a calibrated Bayesian optimizer.
- The effect is universal across all uncertainty-aware acquisition policies.
- Exploration or uncertainty sampling never mitigates the failure mode.
