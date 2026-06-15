# B60 CAMEO-Like Phase-Region GP-UCB Replay Result

Date: 2026-05-30

## Purpose

Address the post-B59 reviewer criticism that the CAMEO evidence used a
random-forest UCB proxy rather than a controller with explicit phase-region
selection. B60 implements a closer, but still not faithful, CAMEO-like replay:
fit a Gaussian process, score each DFT phase region by the maximum candidate
UCB inside that region, select the top phase region, then acquire a batch from
that region by UCB order.

This is not the original MATLAB CAMEO controller.

## Hypothesis

If targeted real-real relinking changes the conditional relation between
record provenance and measured value, then a phase-region GP-UCB replay should
show excess acquisition of the corrupted low-true-property DFT region relative
to clean and random relinking controls.

## Budget And Stop Conditions

- Dataset: public CAMEO Fe-Ga-Pd archive.
- Seeds: 10 paired seeds, 0--9.
- Modes: clean, random relinking, targeted relinking.
- Rounds: 6.
- Batch size: 8.
- Relinking budget: 8 paired target/donor swaps.
- Stop condition: complete all configured seeds or fail on missing artifacts /
  invalid config.

## Data And Target

Source archive:

- `review-stage/CAMEO_NComm-master_20260530.zip`
- SHA256: `4e1633fdda8dd70e8244b0af1e19d368b1ed2770cf6d808212bbc18fa97d47e7`

Target selection:

- Fixed pre-outcome scan over DFT regions.
- Selected target: DFT region 2.
- Target records: 59 / 278.
- Target true mean modified magnetization: 0.4876.
- Donor pool: non-target q90 records, mean 10.1577.

The targeted relinking preserves the label multiset and changes only which
records carry the high donor labels in the initial history.

## Artifacts

Code and configs:

- `scripts/b60_cameo_phase_region_gp_replay.py`
- `tests/test_b60_cameo_phase_region_gp_replay.py`
- `configs/smoke_b60_cameo_phase_region_gp_replay_20260530.json`
- `configs/b60_cameo_phase_region_gp_replay_20260530.json`
- `configs/b60_cameo_phase_region_gp_replay_xrdpca_20260530.json`

Run directories:

- Composition-only GP: `runs/20260530T211205Z_b60-cameo-phase-region-gpucb-10seed`
- Composition + XRD PCA GP: `runs/20260530T211218Z_b60-cameo-phase-region-gpucb-xrdpca-10seed`

Each run writes:

- `metadata.json`
- `config.json`
- `target_scan.csv`
- `swap_pairs.csv`
- `initial_history_labels.csv`
- `round_metrics.csv`
- `selected_records.csv`
- `phase_region_scores.csv`
- `summary_by_mode.csv`
- `dataset_snapshot.csv`
- `feature_columns.csv`

## Result 1: Composition-Only GP

Run:

- `runs/20260530T211205Z_b60-cameo-phase-region-gpucb-10seed`

Summary:

| Mode | Final target-region acquisitions | Excess vs random | Excess vs clean | Audit R2 |
|---|---:|---:|---:|---:|
| Clean | 0.0 | 0.0 | 0.0 | -0.1560 |
| Random relinking | 0.0 | 0.0 | 0.0 | -0.1993 |
| Targeted relinking | 5.6 | 5.6 | 5.6 | 0.0475 |

Per-seed targeted-minus-random differences:

```text
[8, 8, 8, 8, 8, 8, 0, 8, 0, 0]
```

Directional test:

- Positive / zero / negative seeds: 7 / 3 / 0.
- One-sided sign test over nonzero differences: p = 0.0078125.
- Target-region top-1 phase rate: clean 0/60, random 0/60, targeted 7/60.
- Fisher exact test for targeted top-1 enrichment over controls: p = 0.0003578.

Interpretation:

- The result is a clean phase-region flip: targeted relinking makes the target
  DFT region become the selected phase in 7 of 60 targeted rounds and in 7 of
  10 seeds, while controls never select that phase.
- The effect is seed-sensitive and all-or-none: successful seeds acquire one
  full target-region batch; failing seeds acquire none.
- Audit R2 is weak, so this version is strong evidence for phase-region budget
  redirection under this controller, but weak evidence that a high-quality GP
  learned a scientifically useful false model.

## Result 2: Composition + XRD PCA GP

Run:

- `runs/20260530T211218Z_b60-cameo-phase-region-gpucb-xrdpca-10seed`

Summary:

| Mode | Final target-region acquisitions | Excess vs random | Excess vs clean | Audit R2 |
|---|---:|---:|---:|---:|
| Clean | 6.4 | 1.6 | 0.0 | 0.3441 |
| Random relinking | 4.8 | 0.0 | -1.6 | 0.2476 |
| Targeted relinking | 8.8 | 4.0 | 2.4 | 0.3747 |

Per-seed targeted-minus-random differences:

```text
[0, 8, 8, 0, 8, 0, 0, 8, 8, 0]
```

Directional tests:

- Targeted vs random: positive / zero / negative seeds = 5 / 5 / 0;
  one-sided sign test over nonzero differences p = 0.03125.
- Targeted vs clean: positive / zero / negative seeds = 3 / 7 / 0;
  one-sided sign test over nonzero differences p = 0.125.
- Target-region top-1 phase rate: clean 8/60, random 6/60, targeted 11/60.
- Fisher exact test for targeted top-1 enrichment over controls: p = 0.1607.

Interpretation:

- Adding XRD PCA raises audit R2 into a nontrivial range, but it also makes the
  target phase naturally attractive in clean/random controls.
- Targeted relinking still increases target-region acquisitions over random,
  but the clean comparison is weaker.
- This is best read as an operating-boundary result: stronger physical features
  can reduce the contrast between corrupted and uncorrupted runs when the
  target phase is already naturally explored.

## Mechanistic Reading

B60 supports a bounded mechanism:

1. Targeted relinking changes the recorded initial conditional relation for
   DFT region 2. The eight low-true-value target records are recorded with high
   donor labels, moving their recorded mean from 0 to 10.7495.
2. A phase-region acquisition rule can convert that local conditional shift
   into a region-level decision. In successful composition-only seeds, DFT
   region 2 becomes the top-scored phase region during rounds 1--4 and one
   full target-region batch is acquired.
3. The effect is not monotone or unbounded. After the target phase is acquired,
   true feedback reduces its attractiveness and the controller returns to
   other regions.
4. Feature quality changes the operating regime. Composition-only features give
   a clean control contrast but low audit predictive skill; composition+XRD-PCA
   improves predictive skill but also increases clean/random target exploration.

## Reviewer Result-To-Claim Gate

Claude result-to-claim verdict:

- `claim_supported`: partial.
- Decision for using B60 as a major realism block: NEEDS_REVISION.

Main critique:

- B60 is not a faithful CAMEO reproduction.
- Composition-only B60 has weak audit R2, so it should not be described as a
  high-skill GP learning a false scientific model.
- The all-or-none seed pattern needs to be presented as a phase/operating
  boundary, not as reliable universal redirection.

The DeepSeek review attempt timed out at the MCP client after 120 seconds and
was not used as a substantive verdict.

## Supported Claim

The maximum safe claim is:

> In a CAMEO-like phase-region GP-UCB replay on public Fe-Ga-Pd CAMEO records,
> eight label-multiset-preserving targeted relinkings can redirect acquisition
> toward a low-true-property DFT phase region. The effect is strong under
> composition-only GP features (7/10 seeds; +5.6 target-region acquisitions over
> random) and persists more weakly under composition+XRD-PCA features (+4.0 over
> random), but it is seed- and feature-dependent.

## Unsupported Claims

Do not claim:

- B60 is a faithful reproduction of the original MATLAB CAMEO controller.
- The original CAMEO archive was corrupted.
- CAMEO conclusions were wrong.
- This establishes universal vulnerability of phase-region BO.
- This proves universal stealth.
- The GP always forms a high-quality false scientific model.
- The acquisition amplification is unbounded.

## Manuscript Decision

B60 should not be the main external-realism pillar by itself. It is useful as:

- Extended Data / Supplementary evidence that the CAMEO effect is not specific
  to the B31 random-forest UCB proxy.
- A boundary example showing that feature/controller choices determine whether
  targeted relinking creates a clean phase flip or a smaller incremental bias.
- A response to reviewer concerns, with the explicit caveat that the true MATLAB
  CAMEO controller remains unvalidated locally because MATLAB/Octave and the
  graph-cut/PCOMMEND/GRENDEL dependencies are unavailable.

The main manuscript should keep B31/B39 as external real-data replay evidence,
but phrase the CAMEO block as "CAMEO retrospective surrogate replays" rather
than "faithful CAMEO reproduction."

## Verification

Commands run:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b60_cameo_phase_region_gp_replay.py \
  tests/test_no_defaults_policy.py

conda run --no-capture-output -n agentconda \
  python scripts/b60_cameo_phase_region_gp_replay.py \
  --config configs/smoke_b60_cameo_phase_region_gp_replay_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b60_cameo_phase_region_gp_replay.py \
  --config configs/b60_cameo_phase_region_gp_replay_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b60_cameo_phase_region_gp_replay.py \
  --config configs/b60_cameo_phase_region_gp_replay_xrdpca_20260530.json
```

Test result:

- `9 passed in 1.00s` for B60/no-default tests after implementation.
