# B34 Trace-Concentration Quarantine Intervention Result

Date: 2026-05-30

## Status

B34 passed as a low-cost positive-result extension of B33.

The result converts the acquisition-trace concentration signal from a pure
audit into an offline intervention replay:

> if a monitored target/provenance slice is over-concentrated in a proposed
> acquisition batch relative to clean/random control calibration, quarantine
> that slice before executing the batch.

This is not a retrained closed-loop defense and not a live wet-lab experiment.
It is a completed-trace policy replay that estimates how much false allocation
would have been prevented by acting on the B33 signal.

## Protocol

For each run family:

1. Compute proposed-batch concentration:

```text
batch_concentration_ratio =
  proposed batch target fraction / current candidate target prevalence
```

2. Set a per-family threshold from clean and random-swap controls only:

```text
threshold = max control batch_concentration_ratio
```

3. In targeted traces, if a proposed batch exceeds the threshold, quarantine the
monitored target/provenance slice and count the target records in that batch as
prevented false allocations.

4. Report observed final false allocation, prevented false allocation, residual
false allocation, and control quarantine rate.

## Artifacts

Script:

- `scripts/analyze_b34_trace_quarantine_intervention.py`

Config:

- `configs/b34_trace_quarantine_intervention_20260530.json`

Outputs:

- `review-stage/b34_trace_quarantine_intervention_20260530.csv`
- `review-stage/b34_trace_quarantine_intervention_20260530.json`
- `review-stage/b34_trace_quarantine_intervention_20260530.md`

Tests:

- `tests/test_b34_trace_quarantine_intervention.py`

## Results

| Run family | Control quarantine rate | Target quarantine rate | Observed false allocation | Prevented | Residual | Prevented fraction |
|---|---:|---:|---:|---:|---:|---:|
| B18 materials greedy MLP | 0.000 | 0.400 | 41.20 | 41.20 | 0.00 | 1.000 |
| B19 GFP greedy MLP | 0.000 | 0.260 | 47.10 | 47.00 | 0.10 | 0.999 |
| B25 materials epsilon-greedy MLP | 0.000 | 0.380 | 38.80 | 38.80 | 0.00 | 1.000 |
| B25 GFP epsilon-greedy MLP | 0.000 | 0.260 | 42.40 | 41.70 | 0.70 | 0.957 |
| B31 CAMEO RF-UCB replay | 0.000 | 0.150 | 8.00 | 6.80 | 1.20 | 0.828 |
| B32 materials MC-dropout UCB MLP | 0.000 | 0.400 | 42.30 | 42.30 | 0.00 | 1.000 |
| B32 GFP MC-dropout UCB MLP | 0.000 | 0.240 | 47.90 | 47.80 | 0.10 | 0.999 |

## Interpretation

B34 is the strongest current positive-result framing:

> the binding-to-budget mechanism not only predicts a failure signature; it
> yields an actionable trace-concentration quarantine rule that would prevent
> 82.8%--100% of false allocations in the tested completed traces, with zero
> control quarantines under per-family clean/random calibration.

The result is strongest for the controlled neural GFP/materials traces and
remains positive in the CAMEO retrospective replay.

## Claim Boundary

Safe claim:

> In completed-trace intervention replay, a control-calibrated
> trace-concentration quarantine rule prevents most or all false allocations
> across the tested greedy, epsilon-greedy, MC-dropout UCB, and CAMEO replay
> families.

Do not claim:

- live deployed prevention;
- retrained closed-loop defense;
- complete defense;
- detection or intervention without monitored slices;
- universal false-positive control beyond the tested families;
- that the original CAMEO campaign was corrupted.

## Validation

Commands:

```bash
conda run --no-capture-output -n agentconda python scripts/analyze_b34_trace_quarantine_intervention.py --config configs/b34_trace_quarantine_intervention_20260530.json
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_b34_trace_quarantine_intervention.py tests/test_no_defaults_policy.py
```

Observed test result:

```text
7 passed in 1.66s
```
