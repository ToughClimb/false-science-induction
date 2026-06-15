# B53 SAMPLE Retrospective Replay Result

Date: 2026-05-30

## Status

B53 passed as a bounded external-real-SDL support result.

It is not a flagship result. The effect is small in absolute budget because the
public SAMPLE numeric-T50 subset is small, but it is paired and directional:
targeted real-real relinking redirects a reduced-pool GP-UCB replay toward a
low-true-T50 protein axis more often than random relinking while preserving the
label multiset exactly.

## Purpose

Address the reviewer criticism that the evidence package relies mainly on
internally constructed benchmarks and a materials-only external replay.

SAMPLE is a public self-driving protein laboratory archive. B53 asks whether
the same binding-to-budget transduction appears in a retrospective replay using
real SAMPLE protein sequences and real measured T50 values.

## Data

Source archive:

- `review-stage/SAMPLE_code-1.0.0_github_20260530.zip`
- SHA256:
  `b1018ddde2a4e2ea82122174932c5c997cbe4199ea8934bd1e73e2d186fb1549`

Extracted root:

- `review-stage/SAMPLE_code-1.0.0/`

Relevant files:

- `Experiment_Summary.csv`: 20 rounds, 12 sequence assignments per round
  across four agents.
- `Seq_Data_1.csv`..`Seq_Data_4.csv`: per-agent sequence tables containing
  sequence encodings, T50, dead/retry status, and run IDs.
- `all_mean_preds.csv`, `all_std_preds.csv`, `all_prob_preds.csv`: stored
  per-round predictions in the archive.

B53 uses the numeric T50 subset:

- 105 numeric agent-level T50 measurements.
- 59 unique sequence IDs after averaging replicate/agent T50 values.
- 935 binary sequence features per sequence.

Boundary:

This is a retrospective reduced-pool replay on the observed numeric subset,
not a faithful reproduction of the full robot controller and not an audit
finding that the original SAMPLE campaign was corrupt.

## Protocol

Script:

- `scripts/b53_sample_retrospective_replay.py`

Configs:

- Smoke: `configs/smoke_b53_sample_retrospective_replay_20260530.json`
- Full: `configs/b53_sample_retrospective_replay_20260530.json`

Full command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/b53_sample_retrospective_replay.py \
  --config configs/b53_sample_retrospective_replay_20260530.json
```

Run directory:

- `runs/20260530T182020Z_b53-sample-retrospective-replay`

Design:

- Target axis pre-specified after a target-axis scan over observed numeric
  SAMPLE sequences: `pos0=3`.
- Target axis: 9/59 numeric observed sequences.
- Target true mean T50: 42.098.
- Donor pool: non-target sequences above the 0.75 T50 quantile.
- Donor mean T50: 58.439.
- Donor-target contrast: 16.341.
- Relinking budget: 3 paired target/donor swaps.
- Modes: clean, random paired relinking, targeted paired relinking.
- Seeds: 0--9.
- Replay: SAMPLE-style GP-UCB with `DotProduct + WhiteKernel`, mean zeroing,
  and UCB score \(s=(\mu-\min\mu)+2\sigma\).
- Rounds: 5.
- Batch size: 3.
- Candidate pool: 40 records per round, including available target-axis
  candidates and sampled non-target candidates.
- Feedback update: selected records are added back with their true archived T50.

The relinking preserves the multiset of initial recorded labels for every seed
and mode.

## Main Result

Summary file:

- `runs/20260530T182020Z_b53-sample-retrospective-replay/summary_by_mode.csv`

Final mean target-axis acquisitions:

| Mode | Final target-axis acquisitions | Excess vs random | Excess vs clean |
|---|---:|---:|---:|
| Clean | 0.0 | -0.3 | 0.0 |
| Random relinking | 0.3 | 0.0 | 0.3 |
| Targeted relinking | 1.8 | 1.5 | 1.8 |

Paired seed differences:

- Targeted - random:
  `[3, 1, 1, 1, 1, 2, 2, 3, 0, 1]`
- Targeted - clean:
  `[3, 1, 2, 1, 2, 2, 2, 3, 0, 2]`

Exact sign-flip summaries:

| Comparison | Mean difference | Positive seeds | Negative seeds | Ties | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|
| Targeted vs random | 1.5 | 9/10 | 0/10 | 1/10 | 0.00390625 |
| Targeted vs clean | 1.8 | 9/10 | 0/10 | 1/10 | 0.00390625 |

Rank diagnostics:

| Mode | Final target rank percentile |
|---|---:|
| Clean | 0.247 |
| Random relinking | 0.263 |
| Targeted relinking | 0.562 |

Selected target-axis true feedback:

- Random relinking target selections: mean true T50 41.641.
- Targeted relinking target selections: mean true T50 43.782.
- Swap donors used for relinking: T50 60.980, 60.030, 59.792.

## Interpretation

B53 supports a second external-real-data bridge:

> In a public protein self-driving-lab archive, retrospective relinking of real
> high-T50 measurements onto a low-T50 protein axis increases reduced-pool
> GP-UCB acquisition of that axis while preserving the label multiset.

This strengthens realism beyond CAMEO because it uses a protein SDL archive,
not only an external materials closed-loop archive.

The effect is deliberately written as bounded:

- the numeric observed subset has only 59 unique sequences;
- the replay uses a reduced candidate pool;
- the absolute excess is 1.5 target-axis acquisitions over 15 replayed
  acquisitions per seed;
- it does not reproduce the full SAMPLE controller or robot workflow.

## Supported Claim

Safe claim:

> A retrospective reduced-pool replay on a public SAMPLE self-driving protein
> archive shows the same direction of binding-to-budget transduction: targeted
> real-real relinking yields higher acquisition of a low-true-T50 protein axis
> than random relinking in 9/10 paired seeds, with exact label-multiset
> preservation.

## Unsupported Claims

Do not claim:

- the original SAMPLE data were corrupt;
- the original SAMPLE campaign made false discoveries;
- B53 is a live wet-lab validation;
- B53 proves universal SDL vulnerability;
- the reduced-pool GP-UCB replay is a complete reproduction of the SAMPLE
  controller.

## Artifacts

Run artifacts:

- `runs/20260530T182020Z_b53-sample-retrospective-replay/metadata.json`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/config.json`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/target_scan.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/swap_pairs.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/label_multiset_audit.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/round_metrics.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/selected_records.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/summary_by_mode.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/numeric_t50_dataset.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/numeric_t50_measurements.csv`
- `runs/20260530T182020Z_b53-sample-retrospective-replay/round_assignments.csv`

Tests:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b53_sample_retrospective_replay.py \
  tests/test_no_defaults_policy.py
```

Observed:

```text
9 passed in 0.89s
```
