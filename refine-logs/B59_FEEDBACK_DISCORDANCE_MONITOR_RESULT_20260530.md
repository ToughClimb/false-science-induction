# B59 Feedback-Discordance Blind Monitor Result

Date: 2026-05-30

## Purpose

B49 showed that a fully within-campaign all-axis concentration monitor recovers
the induced target axis but fires too often on clean/random control traces. B59
tests whether adding true-feedback discordance reduces those false alarms
without using external clean/random controls to set thresholds.

## Hypothesis

For each candidate axis, score

\\[
S(a) = f_a \\max(0, \\bar y_{\\mathrm{selected}} - \\bar y_a),
\\]

where \\(f_a\\) is the selected fraction on axis \\(a\\). A campaign-calibrated
threshold over early same-axis scores plus same-round peer-axis scores should
reduce clean/random any-axis flags relative to concentration alone while
retaining target-axis recovery in targeted traces.

## Budget And Stop Conditions

- Reuse existing B37/B38/B39 online proposed-trace artifacts.
- No new training jobs.
- Datasets: GFP, materials, CAMEO.
- Baseline rounds: 0.
- Evaluation rounds: 1--4.
- Threshold: within-campaign 0.95 quantile plus 0.02 margin.
- Stop condition: if control any-axis flags remain high, report the result as
  a calibration-free monitoring boundary, not a detector.

## Commands

```bash
conda run --no-capture-output -n agentconda \
  python scripts/analyze_b59_feedback_discordance_monitor.py \
  --config configs/b59_feedback_discordance_monitor_20260530.json

conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b59_feedback_discordance_monitor.py \
  tests/test_no_defaults_policy.py
```

## Artifacts

- Script: `scripts/analyze_b59_feedback_discordance_monitor.py`
- Config: `configs/b59_feedback_discordance_monitor_20260530.json`
- CSV: `review-stage/b59_feedback_discordance_monitor_20260530.csv`
- JSON: `review-stage/b59_feedback_discordance_monitor_20260530.json`
- Markdown summary: `review-stage/b59_feedback_discordance_monitor_20260530.md`

## Main Result

| Dataset | Model | Eval-only control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |
|---|---|---:|---:|---:|---:|---:|
| GFP | MLP | 0.138 | 0.725 | 0.725 | 0.875 | 0.875 |
| Materials | MLP | 0.863 | 0.625 | 0.625 | 0.975 | 0.975 |
| CAMEO | RF-UCB | 0.675 | 0.300 | 0.100 | 0.800 | 0.800 |

Relative to B49 concentration alone:

| Dataset | B49 control any-axis flag | B59 control any-axis flag | B49 target-axis flag | B59 target-axis flag |
|---|---:|---:|---:|---:|
| GFP | 0.850 | 0.138 | 1.000 | 0.725 |
| Materials | 0.925 | 0.863 | 1.000 | 0.625 |
| CAMEO | 0.838 | 0.675 | 0.200 | 0.100 |

## Interpretation

B59 is a mixed boundary result.

It substantially improves the no-clean-calibration story in GFP: any-axis false
alarms drop from 0.850 to 0.138 while the target axis remains flagged in 0.725
of targeted traces. It does not solve the general problem: materials and CAMEO
still have high any-axis false-alarm rates, and CAMEO target-axis flagging is
weak.

The safe claim is:

> Adding feedback discordance can reduce within-campaign false alarms in GFP,
> but no-clean-calibration monitoring remains domain-dependent and is not yet a
> deployable autonomous detector.

## Unsupported Claims

Do not claim:

- calibration-free detection is solved;
- B59 is a complete detector or defense;
- B59 corrects record-level binding errors;
- materials or CAMEO achieve acceptable autonomous false-alarm control.

## Validation

Tests:

```text
8 passed in 1.27s
```

SHA256:

- `review-stage/b59_feedback_discordance_monitor_20260530.csv`:
  `ab063dfbd161395bf0035cc01fdea4fecea6de2f034f29301c548ee4e2e398a8`
- `review-stage/b59_feedback_discordance_monitor_20260530.json`:
  `abbe1c02e6f6d85834f872a4eb1189e8fa07b01557267c525ee3ccf3a0de9488`
- `scripts/analyze_b59_feedback_discordance_monitor.py`:
  `da1c36d3f8bbd5b7ee2a8bf42df0dfcc747d652fe1d5326303cc2cd3dd6e648f`
