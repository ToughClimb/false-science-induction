# M1 Static False Association Result

Date: 2026-05-27

## Purpose

Test whether targeted real-real paired misbinding can make models statically
learn the false association `pos=27 -> high GFP score` before any closed-loop
rollout.

## Run

### Run A: 100 swaps, 512 background

- Run directory: `runs/20260527T190400Z_m1-gfp-pos27-static-xgb-mlp-3seed`
- Target: `pos=27`
- Swap count: `100`
- Background history size: `512`
- Seeds: `0, 1, 2`
- Models: `xgboost`, `mlp`
- Top-k: `500`

### Run B: 50 swaps, 1024 background

- Run directory: `runs/20260527T190549Z_m1-gfp-pos27-static-xgb-mlp-50swap-bg1024-3seed`
- Target: `pos=27`
- Swap count: `50`
- Background history size: `1024`
- Seeds: `0, 1, 2`
- Models: `xgboost`, `mlp`
- Top-k: `500`

### Run C: 25 swaps, 2048 background

- Run directory: `runs/20260527T190706Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed`
- Target: `pos=27`
- Swap count: `25`
- Background history size: `2048`
- Seeds: `0, 1, 2`
- Models: `xgboost`, `mlp`
- Top-k: `500`

## Result Summary: Run A

This is a strong positive for belief induction but not yet a clean stealth
configuration.

### MLP

- Clean FAS: `-0.4521`
- Random-swap FAS: `-0.3414`
- Targeted-swap FAS: `0.5687`
- FAS lift vs clean: `+1.0208`
- FAS lift vs random: `+0.9101`
- Target top-k fraction:
  - clean: `0.000`
  - random: `0.000`
  - targeted: `0.106`
- Mean target rank percentile:
  - clean: `0.2122`
  - random: `0.2768`
  - targeted: `0.6641`
- MAE/R2:
  - clean: `0.5286 / 0.5713`
  - targeted: `0.9081 / -0.1217`

### XGBoost anchor

- Clean FAS: `-0.8612`
- Random-swap FAS: `-0.5979`
- Targeted-swap FAS: `1.5098`
- FAS lift vs clean: `+2.3710`
- FAS lift vs random: `+2.1077`
- Target top-k fraction:
  - clean: `0.000`
  - random: `0.000`
  - targeted: `0.9993`
- Mean target rank percentile:
  - clean: `0.0614`
  - random: `0.1043`
  - targeted: `0.9684`
- MAE/R2:
  - clean: `0.5360 / 0.4911`
  - targeted: `0.8272 / 0.2186`

## Gate Decision

Decision: PARTIAL PASS.

The core false-association gate passes: targeted paired swaps induce a strong
target-specific false association in both a neural model and a conservative
anchor, and targeted swap is much stronger than random paired swap.

The audit non-diagnosticity gate does not pass yet: validation MAE/R2 degradation
is too visible, especially for the MLP. This run should be treated as M1
mechanism sanity, not as the final paper-facing stealth configuration.

## Next Step

Run a lower-budget and/or higher-background configuration:

- fewer swaps, e.g. `25` or `50`;
- larger background history, e.g. `1024` or `2048`;
- keep the same `pos=27` target and random-swap controls;
- prefer a configuration where FAS/rank lift remains positive but MAE/R2 changes
  are less diagnostic.

## Result Summary: Lower-Budget Runs

### Run B: 50 swaps, 1024 background

MLP:

- Targeted FAS lift vs clean: `+1.0050`
- Targeted FAS lift vs random: `+1.0203`
- Targeted top-k fraction: `0.1167` versus `0.0000` clean/random
- Targeted rank percentile: `0.5149` versus `0.1836` clean and `0.1871` random
- MAE/R2:
  - clean: `0.4598 / 0.6632`
  - random: `0.5304 / 0.5629`
  - targeted: `0.6104 / 0.4270`

XGBoost:

- Targeted FAS lift vs clean: `+1.6060`
- Targeted FAS lift vs random: `+1.6200`
- Targeted top-k fraction: `0.3700`
- Targeted rank percentile: `0.7612`
- MAE/R2:
  - clean: `0.5321 / 0.5431`
  - random: `0.5723 / 0.5227`
  - targeted: `0.6445 / 0.4514`

### Run C: 25 swaps, 2048 background

MLP:

- Targeted FAS lift vs clean: `+0.6905`
- Targeted FAS lift vs random: `+0.6703`
- Targeted top-k fraction: `0.0487` versus `0.0000` clean/random
- Targeted rank percentile: `0.3796` versus `0.1808` clean and `0.1844` random
- MAE/R2:
  - clean: `0.3730 / 0.7505`
  - random: `0.4030 / 0.7156`
  - targeted: `0.4370 / 0.6648`

XGBoost:

- Targeted FAS lift vs clean: `+0.5987`
- Targeted FAS lift vs random: `+0.5970`
- Targeted top-k fraction: `0.0033`
- Targeted rank percentile: `0.3639`
- MAE/R2:
  - clean: `0.5259 / 0.5739`
  - random: `0.5359 / 0.5674`
  - targeted: `0.5612 / 0.5486`

## Updated Gate Decision

Decision: PASS for M1 static false association.

The 25-swap/2048-background run is the best current M2 candidate because it
preserves a clear neural false-association signal while keeping R2 positive and
closer to clean/random. The 50-swap/1024-background run is a stronger mechanism
configuration and should be retained as a backup if the lower-budget setting
does not produce closed-loop target allocation lift.

Recommended M2 starting configuration:

- target: `pos=27`
- swap count: `25`
- background history size: `2048`
- model: `mlp` first, `xgboost` as anchor
- acquisition: greedy top predicted mean
- rounds: `5`
- batch size: `20`
- seeds: `0, 1, 2`
