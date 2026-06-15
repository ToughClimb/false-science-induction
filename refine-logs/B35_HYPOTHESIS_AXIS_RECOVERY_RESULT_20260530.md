# B35 Hypothesis-Axis Recovery Result

Date: 2026-05-30

## Status

B35 passed as a low-cost type-III result: false hypotheses can be used as a
retrospective probe to recover the scientific axis whose recorded optimism
conflicts with true feedback.

This is not automatic record-level correction. It is an axis-level recovery and
triage signal from completed acquisition traces.

## Hypothesis

If targeted misbinding induces a closed loop to pursue a false scientific axis,
then the loop's own selections should expose that axis as a conflict:

> high allocation concentration plus low true feedback.

Ranking candidate axes by this conflict score should recover the injected axis
in targeted traces, while clean and random-swap controls should not recover the
same axis.

## Budget

- No retraining.
- No new closed-loop runs.
- Reuse completed B19 GFP and B18 materials greedy traces.
- Analyze two models per domain: MLP and TabM-mini.
- Ten seeds per model/mode.

## Acceptance Criteria

- Targeted traces recover the injected axis at aggregate rank 1 in both
  domains.
- Control traces do not recover the injected axis at aggregate rank 1.
- Per-seed top-1 or top-2 recovery is high in targeted traces and zero in
  controls.

## Stop Conditions

- Stop if required selected-record traces are missing.
- Stop if the target axis cannot be represented by the domain-specific axis
  enumerator.
- Do not broaden to new domains before verifying the completed B18/B19 traces.

## Method

For each selected-record trace, enumerate candidate scientific axes:

- GFP: mutation positions, e.g. `pos=27`.
- Materials: composition axes, e.g. `element=Co` and `major_element=Co`.

For each axis \(a\), compute

\[
S(a)
=
\frac{n_a}{n}
\cdot
\max(0, \bar y_{\mathrm{selected}} - \bar y_a).
\]

Here \(n_a/n\) is the selected-budget fraction allocated to the axis and
\(\bar y_{\mathrm{selected}} - \bar y_a\) is the true-feedback deficit of that
axis relative to the selected set. An axis is suspicious when it receives
substantial selected budget but returns low true measurements.

## Artifacts

Script:

- `scripts/analyze_b35_hypothesis_axis_recovery.py`

Config:

- `configs/b35_hypothesis_axis_recovery_20260530.json`

Outputs:

- `review-stage/b35_hypothesis_axis_recovery_20260530.csv`
- `review-stage/b35_hypothesis_axis_recovery_20260530.json`
- `review-stage/b35_hypothesis_axis_recovery_20260530.md`

Tests:

- `tests/test_b35_hypothesis_axis_recovery.py`

## Results

| Dataset | Model | Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |
|---|---|---|---:|---:|---|---:|
| B19 GFP greedy | MLP | clean | 0/10 | 0/10 | `pos=219` | 36 |
| B19 GFP greedy | MLP | random_swap | 0/10 | 0/10 | `pos=219` | 28 |
| B19 GFP greedy | MLP | targeted_swap | 8/10 | 10/10 | `pos=27` | 1 |
| B19 GFP greedy | TabM-mini | clean | 0/10 | 0/10 | `pos=110` | 182 |
| B19 GFP greedy | TabM-mini | random_swap | 0/10 | 0/10 | `pos=59` | 18 |
| B19 GFP greedy | TabM-mini | targeted_swap | 9/10 | 10/10 | `pos=27` | 1 |
| B18 materials greedy | MLP | clean | 0/10 | 0/10 | `major_element=F` | 129 |
| B18 materials greedy | MLP | random_swap | 0/10 | 0/10 | `major_element=F` | 55 |
| B18 materials greedy | MLP | targeted_swap | 10/10 | 10/10 | `element=Co` | 1 |
| B18 materials greedy | TabM-mini | clean | 0/10 | 0/10 | `major_element=F` | 122 |
| B18 materials greedy | TabM-mini | random_swap | 0/10 | 0/10 | `major_element=F` | 126 |
| B18 materials greedy | TabM-mini | targeted_swap | 10/10 | 10/10 | `element=Co` | 1 |

For materials, `element=Co` and `major_element=Co` are treated as aliases of
the induced Co axis. In aggregate targeted traces, the two Co aliases are ranks
1 and 2 for both neural models.

## Interpretation

B35 gives the type-III story a concrete result:

> Induced false hypotheses are not only harmful; their failure under true
> feedback creates a diagnostic contrast that can recover the implicated
> scientific axis.

This converts the warning into a probe:

1. deliberately stress a monitored family of binding/provenance axes;
2. observe where acquisition concentrates;
3. compare the selected axis against true feedback;
4. prioritize the recovered axis for provenance audit or quarantine.

## Claim Boundary

Safe claim:

> In completed GFP and materials traces, a simple conflict-trace score recovers
> the induced hypothesis axis at aggregate rank 1 in targeted traces and never
> at rank 1 in clean/random controls.

Do not claim:

- automatic recovery of the exact corrupted records;
- correction of labels or provenance;
- discovery of a new true scientific law;
- that arbitrary unknown natural corruptions can always be localized;
- generality beyond axes that are enumerable from the record representation.

## Validation

Commands:

```bash
conda run --no-capture-output -n agentconda python scripts/analyze_b35_hypothesis_axis_recovery.py --config configs/b35_hypothesis_axis_recovery_20260530.json
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_b35_hypothesis_axis_recovery.py tests/test_no_defaults_policy.py
```

Observed test result:

```text
6 passed in 1.31s
```
