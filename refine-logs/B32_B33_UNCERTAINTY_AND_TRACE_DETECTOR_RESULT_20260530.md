# B32/B33 Uncertainty-Aware Acquisition and Trace Detector Result

Date: 2026-05-30

## Status

Both review-priority strengthening items passed.

- B32 shows that the flagship GFP and materials false-pursuit effects persist
  under neural MC-dropout UCB acquisition.
- B33 shows that a simple acquisition-trace concentration audit flags the tested
  targeted traces early with no false positives on paired clean/random controls.

## B32 Protocol

MC-dropout UCB was added only for MLP surrogates:

```text
score = MC mean + beta * MC std
```

Configuration:

- `passes = 16`
- `beta = 1.0`
- `seed_offset = 32000`
- 10 paired seeds
- 5 closed-loop rounds
- clean, random-swap, targeted-swap modes

This is an uncertainty-aware neural acquisition check, not a claim that
MC-dropout is a fully calibrated Bayesian optimizer.

## B32 Artifacts

Scripts/code:

- `src/false_science/models.py`
- `scripts/materials_triggered_false_regulariry.py`
- `scripts/m2_triggered_closed_loop_false_pursuit.py`

Configs:

- `configs/smoke_b32_materials_mc_dropout_ucb_mlp_20260530.json`
- `configs/smoke_b32_gfp_mc_dropout_ucb_mlp_20260530.json`
- `configs/b32_materials_mc_dropout_ucb_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json`
- `configs/b32_gfp_mc_dropout_ucb_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json`
- `configs/b32_statistics_aggregate_20260530.json`

Runs:

- `runs/20260530T102134Z_b32-materials-mc-dropout-ucb-disttrigger-dim32-s001-25swap-bg1024-mlp-10seed-80ep`
- `runs/20260530T102149Z_b32-gfp-mc-dropout-ucb-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep`

Statistics:

- `review-stage/b32_mc_dropout_ucb_statistics_20260530.csv`
- `review-stage/b32_mc_dropout_ucb_statistics_20260530.json`

## B32 Results

Final cumulative triggered-target acquisitions:

| Domain | Clean | Random | Targeted | Targeted - Random |
|---|---:|---:|---:|---:|
| Materials | 0.0 | 0.1 | 42.3 | 42.2 |
| GFP | 0.1 | 0.1 | 47.9 | 47.8 |

Paired seed statistics:

| Effect | Paired differences | Mean | 95% bootstrap CI | Sign-flip p |
|---|---|---:|---:|---:|
| Materials MC-dropout UCB | `[43,42,42,41,42,42,41,45,39,45]` | 42.2 | [41.2, 43.3] | 0.001953 |
| GFP MC-dropout UCB | `[58,22,85,71,26,26,41,19,49,81]` | 47.8 | [33.5, 62.8] | 0.001953 |

Interpretation:

- The effect is not explained by purely greedy top-mean exploitation.
- MC-dropout UCB does not mitigate the tested targeted misbinding in either
  flagship domain.
- This supports a stronger acquisition-policy robustness claim for the tested
  neural loops.

## B33 Protocol

The detector computes, at detection round 1:

```text
concentration_ratio =
  cumulative target acquisition fraction / current candidate target prevalence
```

For each run family, the alert threshold is calibrated using only clean and
random-swap controls from the same run family. Targeted-swap rows are never used
to set the threshold.

This is a monitored-slice acquisition audit, not a complete defense.

## B33 Artifacts

Script and tests:

- `scripts/analyze_b33_acquisition_trace_detector.py`
- `tests/test_b33_acquisition_trace_detector.py`

Configs:

- `configs/b33_acquisition_trace_detector_20260530.json`
- `configs/b33_acquisition_trace_detector_with_b32_20260530.json`

Outputs:

- `review-stage/b33_acquisition_trace_detector_20260530.csv`
- `review-stage/b33_acquisition_trace_detector_20260530.json`
- `review-stage/b33_acquisition_trace_detector_20260530.md`
- `review-stage/b33_acquisition_trace_detector_with_b32_20260530.csv`
- `review-stage/b33_acquisition_trace_detector_with_b32_20260530.json`
- `review-stage/b33_acquisition_trace_detector_with_b32_20260530.md`

## B33 Results

Round-1 detector performance:

| Run family | Controls | Targeted | FPR | TPR | Control max | Target mean |
|---|---:|---:|---:|---:|---:|---:|
| B18 materials greedy MLP | 20 | 10 | 0.000 | 1.000 | 0.443 | 70.027 |
| B19 GFP greedy MLP | 20 | 10 | 0.000 | 1.000 | 0.950 | 55.057 |
| B25 materials epsilon-greedy MLP | 20 | 10 | 0.000 | 1.000 | 1.328 | 55.327 |
| B25 GFP epsilon-greedy MLP | 20 | 10 | 0.000 | 1.000 | 0.950 | 45.092 |
| B31 CAMEO RF-UCB | 20 | 10 | 0.000 | 1.000 | 0.266 | 2.378 |
| B32 materials MC-dropout UCB MLP | 20 | 10 | 0.000 | 1.000 | 0.443 | 77.670 |
| B32 GFP MC-dropout UCB MLP | 20 | 10 | 0.000 | 1.000 | 0.950 | 56.579 |

Interpretation:

- The tested targeted false-pursuit traces create a large early concentration
  signal.
- The detector is useful as a first mitigation/audit result for the paper.
- It should be framed as an acquisition governance diagnostic, not as a
  universal detector.

## Validation

Commands:

```bash
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_models.py tests/test_no_defaults_policy.py
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_b33_acquisition_trace_detector.py tests/test_no_defaults_policy.py
conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/smoke_b32_materials_mc_dropout_ucb_mlp_20260530.json
conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/smoke_b32_gfp_mc_dropout_ucb_mlp_20260530.json
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b32_materials_mc_dropout_ucb_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b32_gfp_mc_dropout_ucb_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json
conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b32_statistics_aggregate_20260530.json
conda run --no-capture-output -n agentconda python scripts/analyze_b33_acquisition_trace_detector.py --config configs/b33_acquisition_trace_detector_with_b32_20260530.json
```

## Paper Claim Update

Safe addition:

> The effect persists under epsilon-greedy exploration and MC-dropout UCB in the
> two flagship neural domains, and acquisition traces provide an early
> concentration signal that can be audited when target/provenance slices are
> monitored.

Do not claim:

- all uncertainty-aware acquisition policies are vulnerable;
- MC-dropout UCB is calibrated;
- B33 is a complete defense;
- B33 detects attacks without a monitored slice;
- real-world corruption has occurred.
