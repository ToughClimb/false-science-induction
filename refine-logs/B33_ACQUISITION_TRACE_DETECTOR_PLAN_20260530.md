# B33 Acquisition-Trace Concentration Detector Plan

Date: 2026-05-30

## Motivation

The review correctly flags that the manuscript currently diagnoses a failure
mode but has no mitigation or defense experiment. B33 adds a lightweight,
post-hoc detector that uses only the acquisition trace, not hidden corruption
labels, to test whether false-pursuit runs create abnormal concentration in a
small target/provenance slice.

## Hypothesis

If targeted misbinding redirects closed-loop search toward a false basin, then
the acquisition trace should concentrate in that basin far above its candidate
prevalence, while clean and random-swap controls should define a lower
concentration baseline.

## Smallest Falsifiable Experiment

Use existing append-only run artifacts. For each dataset/run/model/acquisition
policy, compute at a fixed detection round:

```text
concentration_ratio =
  cumulative target acquisition fraction / current candidate target prevalence
```

Calibrate the alert threshold from clean and random-swap controls within the
same run family. Evaluate targeted-swap recall and control false-positive rate.

## Inputs

Initial detector inputs:

- B18 materials greedy distributed-trigger run.
- B19 GFP greedy distributed-trigger run.
- B25 materials epsilon-greedy distributed-trigger run.
- B25 GFP epsilon-greedy distributed-trigger run.
- B31 CAMEO retrospective RF-UCB replay.

B32 runs will be added after they complete.

## Budget

- CPU only.
- One analysis script.
- No new training.
- Output CSV and JSON under `review-stage/`.

## Acceptance Criteria

Primary pass:

- Detector flags most targeted-swap paired seeds in the flagship runs while
  keeping clean/random controls at zero false positives under the calibrated
  threshold.

Useful partial pass:

- Detector works on some acquisition policies but not others; report it as a
  diagnostic boundary and not as a complete defense.

## Stop Conditions

- Stop if required round-metric columns are absent or incompatible.
- Stop if threshold calibration uses targeted-swap rows.
- Stop if the detector requires corruption labels at test time.

## Claim If Positive

Acquisition-trace concentration is a simple audit signal that can flag many
false-pursuit runs in the tested closed-loop traces.

## Forbidden Claims

- This is a complete defense.
- It detects all attacks.
- It proves stealth is impossible.
- It detects real-world corruption without a monitored provenance slice.
