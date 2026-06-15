# B58 SAMPLE Metadata-Shift Replay Result

Date: 2026-05-30

## Status

B58 is a bounded positive realism result with an explicit offset/susceptibility
boundary.

It strengthens B55 by moving from opportunity-surface mining to a concrete
metadata-error replay on the public SAMPLE protein self-driving-lab archive:
a fixed cyclic offset within planned round/agent/rank blocks. It does not
support a universal or natural-corruption claim. One fixed offset is weak or
negative versus random-cycle relinking, while three of four tested offsets are
directionally positive.

## Purpose

Reviewers objected that B55 only showed that real archives contain coherent
metadata surfaces; it did not show that a plausible metadata error can redirect
closed-loop acquisition. B58 asks whether fixed within-round planned-position
offsets in SAMPLE metadata can preserve the label multiset while moving a
GP-UCB replay toward a low-true-T50 protein axis.

## Data

Source archive:

- `review-stage/SAMPLE_code-1.0.0_github_20260530.zip`
- SHA256:
  `b1018ddde2a4e2ea82122174932c5c997cbe4199ea8934bd1e73e2d186fb1549`

Extracted root:

- `review-stage/SAMPLE_code-1.0.0/`

B58 uses the same numeric T50 subset as B53:

- 105 numeric agent-level T50 measurements.
- 59 unique sequence IDs after averaging replicate/agent T50 values.
- 935 binary sequence features per sequence.

Boundary:

This is a retrospective reduced-pool replay over the observed numeric subset,
not a reproduction of the full SAMPLE robot controller and not an audit finding
that SAMPLE was corrupt.

## Protocol

Scripts:

- `scripts/b58_sample_metadata_shift_replay.py`
- `scripts/aggregate_b58_metadata_shift_replay.py`

Configs:

- `configs/b58_sample_metadata_shift_replay_20260530.json` (`shift=2`)
- `configs/b58_sample_metadata_shift_replay_shift1_20260530.json`
- `configs/b58_sample_metadata_shift_replay_shift_minus1_20260530.json`
- `configs/b58_sample_metadata_shift_replay_shift_minus2_20260530.json`
- `configs/b58_metadata_shift_aggregate_20260530.json`

Full commands:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/b58_sample_metadata_shift_replay.py \
  --config configs/b58_sample_metadata_shift_replay_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b58_sample_metadata_shift_replay.py \
  --config configs/b58_sample_metadata_shift_replay_shift1_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b58_sample_metadata_shift_replay.py \
  --config configs/b58_sample_metadata_shift_replay_shift_minus1_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b58_sample_metadata_shift_replay.py \
  --config configs/b58_sample_metadata_shift_replay_shift_minus2_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/aggregate_b58_metadata_shift_replay.py \
  --config configs/b58_metadata_shift_aggregate_20260530.json
```

Run directories:

- `runs/20260530T194912Z_b58-sample-metadata-shift-replay`
- `runs/20260530T195020Z_b58-sample-metadata-shift-replay-shift1`
- `runs/20260530T195020Z_b58-sample-metadata-shift-replay-shift-minus1`
- `runs/20260530T195020Z_b58-sample-metadata-shift-replay-shift-minus2`

Design:

- Fixed cyclic label offset inside real SAMPLE planned round/agent/rank blocks.
- Offsets tested: -2, -1, +1, +2.
- Auto-axis selection requires:
  - positive planned-shift delta,
  - enough planned-position support,
  - at least two target-axis candidates left unobserved after the initial
    relinked history.
- All four offsets selected the same axis: `frag0=P3F0`, equivalent to
  `pos0=3`.
- Target axis: 9/59 numeric observed sequences.
- Remaining target-axis candidates after initial relinked history: 6.
- Modes: clean, random cycle shift of the same block sizes, planned position
  shift.
- Seeds: 0--9.
- Replay: DotProduct + WhiteKernel GP-UCB, beta 2.0, five rounds, batch size 3,
  candidate pool 40.

All modes preserve the initial recorded-label multiset in every seed and
offset.

## Main Result

Aggregate files:

- `figures/b58_sample_metadata_shift_replay_summary.csv`
- `figures/b58_sample_metadata_shift_replay_summary.json`
- `figures/b58_sample_metadata_shift_replay_summary_table.tex`
- `figures/b58_sample_metadata_shift_replay.pdf`

Summary:

| Shift | Planned | Random | Clean | Excess vs random | Excess vs clean | Planned rank pct. |
|---:|---:|---:|---:|---:|---:|---:|
| -2 | 2.7 | 2.0 | 1.1 | +0.7 | +1.6 | 0.689 |
| -1 | 1.8 | 1.2 | 1.2 | +0.6 | +0.6 | 0.588 |
| +1 | 2.7 | 1.5 | 0.4 | +1.2 | +2.3 | 0.631 |
| +2 | 1.4 | 1.6 | 1.1 | -0.2 | +0.3 | 0.381 |

Across offsets:

- 3/4 fixed offsets are positive versus random-cycle relinking.
- Mean planned excess versus random-cycle relinking: +0.575 final target-axis
  acquisitions.
- Mean planned excess versus clean: +1.200 final target-axis acquisitions.
- Mean target rank percentile: planned 0.572, random 0.379, clean 0.354.

Paired seed differences versus random:

- Shift -2: `[1, 1, -1, 0, 0, 1, 0, 3, 3, -1]`, mean +0.7, 5 positive,
  2 negative, 3 tied, exact sign p=0.453.
- Shift -1: `[0, 0, 0, 2, 2, 0, 0, 0, 2, 0]`, mean +0.6, 3 positive,
  0 negative, 7 tied, exact sign p=0.250.
- Shift +1: `[2, 0, 3, 2, 0, 1, -1, 1, 1, 3]`, mean +1.2, 7 positive,
  1 negative, 2 tied, exact sign p=0.070.
- Shift +2: `[0, 1, -2, 2, 0, -1, 0, 0, 0, -2]`, mean -0.2, 2 positive,
  3 negative, 5 tied, exact sign p=1.000.

## Interpretation

B58 supports this claim:

> On a public protein self-driving-lab archive, a concrete fixed-position
> metadata offset can preserve the label multiset and, for favorable offsets,
> increase reduced-pool GP-UCB acquisition and ranking of a low-true-T50
> protein axis relative to random cycle relinking.

The result is deliberately not framed as a universal strong effect. The numeric
SAMPLE subset is small, the replay is reduced-pool, and the offset direction
matters. That boundary is scientifically useful: it links the realism question
to the main mechanism. Coherent metadata errors are not damaging because they
are "errors" in general; they are damaging when the offset rewrites a learnable
conditional axis and the controller has enough remaining target candidates to
spend budget on.

## Supported Claim

Safe claim:

> A concrete planned-position shift on real SAMPLE metadata yields
> offset-dependent binding-to-budget transduction: three of four fixed offsets
> increase final acquisition of the same low-true-T50 protein axis over random
> cycle relinking while preserving the label multiset, and all four elevate the
> target-axis rank percentile relative to clean/random controls on average.

## Unsupported Claims

Do not claim:

- the original SAMPLE archive was corrupt;
- SAMPLE naturally contained planned-position shifts;
- every fixed metadata offset produces false pursuit;
- B58 reproduces the full SAMPLE controller;
- B58 is live wet-lab validation;
- B58 alone establishes universal vulnerability or universal stealth.

## Artifacts

Figure:

- `figures/b58_sample_metadata_shift_replay.pdf`
- SHA256:
  `f8b790efe304978e3454a3a41cdc072d866f1e2c2f89a1595fa93edf78819f51`

Aggregate summary:

- `figures/b58_sample_metadata_shift_replay_summary.csv`
- SHA256:
  `ea4eef5d08490fe5b388a35049a1d88b7e5322fa4f20e830201c252b04dbb989`
- `figures/b58_sample_metadata_shift_replay_summary.json`
- SHA256:
  `edfb0cffd0369ab27e2d5d7898f41763f0af6861f3c0f064c60b979438a552b1`

Implementation:

- `scripts/b58_sample_metadata_shift_replay.py`
- SHA256:
  `9f70013695f68ad1640009f79c368cb7c5ab99400a82925cea75afd82a8431e6`
- `scripts/aggregate_b58_metadata_shift_replay.py`
- SHA256:
  `e4062488649c227dfa2da13a7c563b2abde07afac068b45ee30086c60ec5cb16`

Tests:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b58_metadata_shift_aggregate.py \
  tests/test_b58_sample_metadata_shift_replay.py \
  tests/test_no_defaults_policy.py
```

Observed:

```text
9 passed in 1.99s
```
