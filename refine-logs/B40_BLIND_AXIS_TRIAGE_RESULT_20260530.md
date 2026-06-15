# B40 Blind Hypothesis-Axis Triage Result

Date: 2026-05-30

## Purpose

Make the B35 "false hypothesis as diagnostic probe" result explicitly blind:
the scoring algorithm does not receive the injected target axis. It enumerates
candidate axes, ranks them by allocation-feedback conflict, and only then uses
target-axis aliases for evaluation.

## Hypothesis

For targeted GFP and materials traces, blind feedback-conflict ranking should
recover the implicated hypothesis axis near the top. Clean and random controls
should not recover the same target axis at top rank. CAMEO was included as a
real-data boundary check.

## Config

- `configs/b40_blind_axis_triage_20260530.json`

Command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/analyze_b40_blind_axis_triage.py \
  --config configs/b40_blind_axis_triage_20260530.json
```

Outputs:

- `review-stage/b40_blind_axis_triage_20260530.csv`
- `review-stage/b40_blind_axis_triage_20260530.json`
- `review-stage/b40_blind_axis_triage_20260530.md`

## Main GFP and Materials Result

The blind score is:

```text
selected-budget fraction x max(0, selected mean - axis true-feedback mean)
```

The target axis is not passed to the scoring function.

| Dataset | Model | Mode | Seed top-1 | Seed top-2 | Aggregate top axis | Target rank |
|---|---|---:|---:|---:|---|---:|
| GFP B19 | MLP | clean | 0/10 | 0/10 | `pos=219` | 36 |
| GFP B19 | MLP | random | 0/10 | 0/10 | `pos=219` | 28 |
| GFP B19 | MLP | targeted | 8/10 | 10/10 | `pos=27` | 1 |
| GFP B19 | TabM-mini | clean | 0/10 | 0/10 | `pos=110` | 182 |
| GFP B19 | TabM-mini | random | 0/10 | 0/10 | `pos=59` | 18 |
| GFP B19 | TabM-mini | targeted | 9/10 | 10/10 | `pos=27` | 1 |
| Materials B18 | MLP | clean | 0/10 | 0/10 | `major_element=F` | 129 |
| Materials B18 | MLP | random | 0/10 | 0/10 | `major_element=F` | 55 |
| Materials B18 | MLP | targeted | 10/10 | 10/10 | `element=Co` | 1 |
| Materials B18 | TabM-mini | clean | 0/10 | 0/10 | `major_element=F` | 122 |
| Materials B18 | TabM-mini | random | 0/10 | 0/10 | `major_element=F` | 126 |
| Materials B18 | TabM-mini | targeted | 10/10 | 10/10 | `element=Co` | 1 |

## CAMEO Boundary Check

CAMEO was intentionally included as a boundary. It does not support the blind
triage claim because the target region is already top-ranked in clean and
random controls:

| Dataset | Mode | Seed top-1 | Aggregate top axis | Target rank |
|---|---|---:|---|---:|
| CAMEO B31 | clean | 9/10 | `dft_region=2` | 1 |
| CAMEO B31 | random | 9/10 | `dft_region=2` | 1 |
| CAMEO B31 | targeted | 10/10 | `dft_region=2` | 1 |

This is not a failure of the mechanism. It means DFT region 2 is naturally a
low-feedback/high-allocation conflict axis in the CAMEO replay, so it cannot be
used as a clean control-separated blind recovery result.

## Interpretation

B40 strengthens B35 by making the triage protocol explicit:

- enumerate candidate axes;
- score each axis by allocation-feedback conflict;
- rank axes without target knowledge;
- evaluate whether the induced axis appears at top rank.

For GFP and materials, the false hypothesis itself is a diagnostic probe: the
loop spends budget on the wrong axis, true feedback reveals that the axis is
bad, and the conflict score recovers the implicated axis.

## Claim Boundaries

Supported:

- Blind axis-level triage in controlled GFP and materials traces.
- False hypotheses can expose the scientific axis whose recorded optimism
  conflicts with true feedback.

Not supported:

- Record-level repair.
- Causal discovery of the true scientific law.
- CAMEO blind axis recovery as a clean positive result.
- Triage without an enumerable axis vocabulary.

## Verification

Commands run:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b40_blind_axis_triage.py \
  tests/test_b35_hypothesis_axis_recovery.py \
  tests/test_no_defaults_policy.py

conda run --no-capture-output -n agentconda \
  python scripts/analyze_b40_blind_axis_triage.py \
  --config configs/b40_blind_axis_triage_20260530.json
```

Test result:

- `9 passed in 1.27s`.
