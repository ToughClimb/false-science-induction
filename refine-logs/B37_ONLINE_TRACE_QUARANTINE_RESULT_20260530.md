# B37 Online Trace-Quarantine Result

Date: 2026-05-30

## Status

B37 passed as the first true online closed-loop intervention in the current
codebase.

This upgrades B34 from completed-trace replay to an online simulated discovery
loop:

> at each round, the loop proposes an acquisition batch, computes the
> target-slice concentration using only current candidate prevalence and the
> proposed batch, quarantines the monitored slice before execution if the
> control-calibrated threshold is exceeded, refills from the remaining ranked
> candidates, and trains the next round only on the actually executed feedback.

This is still not a wet-lab or deployed self-driving-lab intervention. It is an
online closed-loop computational intervention.

## Hypothesis

If trace concentration is a mechanism-derived early warning signal, then an
online quarantine rule should prevent false budget execution in the same loop
that generated the risky proposals, without using future trace information.

## Budget

- Dataset: GFP B19 configuration.
- Model: MLP.
- Modes: clean, random_swap, targeted_swap.
- Seeds: 0--9.
- Rounds: 5.
- Epochs: 80.
- Acquisition: top-mean.
- Quarantine threshold: `1.8966666666666667`, copied from B34 clean/random
  control calibration for `b19_gfp_greedy_mlp`.
- Replacement strategy: drop monitored triggered-target slice and refill from
  the next ranked non-quarantined candidates.

## Acceptance Criteria

- Clean/random control quarantine rate remains zero or near zero.
- Targeted runs show high quarantine rate.
- Executed false allocations are reduced to control scale.
- The intervention does not use completed-trace future information.

## Stop Conditions

- Stop if online quarantine requires future rounds.
- Stop if batch refill cannot preserve batch size.
- Stop if clean/random controls trigger frequent quarantines.
- Stop if executed targeted false allocation remains close to the unprotected
  B19 baseline.

## Artifacts

Script:

- `scripts/m2_triggered_online_quarantine.py`

Configs:

- `configs/smoke_b37_gfp_online_quarantine_mlp_20260530.json`
- `configs/b37_gfp_online_quarantine_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json`

Run:

- `runs/20260530T140600Z_b37-gfp-online-quarantine-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep`

Key outputs:

- `runs/20260530T140600Z_b37-gfp-online-quarantine-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep/round_metrics.csv`
- `runs/20260530T140600Z_b37-gfp-online-quarantine-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep/summary_by_model_mode.csv`
- `runs/20260530T140600Z_b37-gfp-online-quarantine-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep/selected_records.csv`
- `runs/20260530T140600Z_b37-gfp-online-quarantine-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep/metadata.json`

Tests:

- `tests/test_b37_online_quarantine.py`

## Results

| Mode | Seeds | Rounds | Quarantine rate | Proposed false allocation | Executed false allocation | Prevented false allocation | Prevented fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 10 | 5 | 0.00 | 0.1 | 0.1 | 0.0 | 0.000 |
| random_swap | 10 | 5 | 0.00 | 0.1 | 0.1 | 0.0 | 0.000 |
| targeted_swap | 10 | 5 | 0.98 | 376.1 | 0.1 | 376.0 | 0.9997 |

Per-seed targeted final counts:

| Seed | Proposed | Executed | Prevented |
|---:|---:|---:|---:|
| 0 | 422 | 0 | 422 |
| 1 | 391 | 0 | 391 |
| 2 | 368 | 1 | 367 |
| 3 | 427 | 0 | 427 |
| 4 | 393 | 0 | 393 |
| 5 | 314 | 0 | 314 |
| 6 | 275 | 0 | 275 |
| 7 | 345 | 0 | 345 |
| 8 | 375 | 0 | 375 |
| 9 | 451 | 0 | 451 |

## Interpretation

B37 is the answer to the online/offline concern:

> The current system now has an online closed-loop intervention, not only a
> completed-trace replay. Using a threshold calibrated from clean/random
> controls, the loop quarantines over-concentrated triggered-target batches
> before execution and reduces executed false allocations from 376.1 proposed
> records per seed to 0.1 executed records per seed.

The control behavior is also important: clean and random_swap have zero
quarantine rate under the same online rule.

## Mechanistic Consequence

B37 also exposes a useful boundary. Quarantine prevents waste, but it does not
automatically cure the learned false association. Because the quarantined target
records are not executed, the model does not receive true feedback from that
slice, so the loop keeps proposing the false axis in later rounds. This is why
the correct framing is:

> online stop-loss and governance intervention,

not:

> automatic model repair.

This makes B35 complementary: after quarantine or repeated quarantine, the
feedback-conflict/axis-recovery analysis can identify the implicated hypothesis
axis for provenance audit.

## Claim Boundary

Safe claim:

> In an online GFP closed-loop simulation, a control-calibrated
> trace-concentration quarantine rule prevents 99.97% of proposed false
> triggered-target allocations from being executed, while leaving clean/random
> controls unquarantined.

Do not claim:

- wet-lab or deployed-platform validation;
- complete defense;
- model self-healing;
- record-level correction;
- universal false-positive control beyond the calibrated setting;
- full generality across acquisition policies or unmonitored axes.

## Submission Implication

B37 materially strengthens the manuscript relative to B34 alone:

- B34: completed-trace intervention replay.
- B37: online closed-loop computational intervention with real-time batch
  quarantine and executed-feedback retraining.

For a Nature-main-first strategy, the wording can now say "online closed-loop
simulation" rather than only "completed-trace replay." It still must not say
"live self-driving-lab intervention."

## Validation

Commands:

```bash
conda run --no-capture-output -n agentconda python scripts/m2_triggered_online_quarantine.py --config configs/smoke_b37_gfp_online_quarantine_mlp_20260530.json
conda run --no-capture-output -n agentconda python scripts/m2_triggered_online_quarantine.py --config configs/b37_gfp_online_quarantine_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_b37_online_quarantine.py tests/test_no_defaults_policy.py
```

Observed test result:

```text
7 passed in 0.92s
```
