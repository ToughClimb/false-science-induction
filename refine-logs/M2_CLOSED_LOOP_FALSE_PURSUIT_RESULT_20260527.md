# M2 Closed-Loop False Pursuit Result

Date: 2026-05-27

## Purpose

Test whether the static false association from M1 changes closed-loop
experimental allocation when future feedback uses true oracle labels.

## Protocol

- Target: `pos=27`
- Initial corruption only: targeted paired real-real misbinding in the initial
  history.
- Future selected records are appended with true labels.
- Acquisition: greedy top predicted mean.
- Rounds: `5`
- Batch size: `20`
- Seeds: `0, 1, 2`

## Run A: Lower-Budget Candidate

- Run directory: `runs/20260527T191118Z_m2-gfp-pos27-loop-xgb-mlp-25swap-bg2048-3seed`
- Swap count: `25`
- Background history size: `2048`
- Models: `mlp`, `xgboost`

### MLP result

- Mean batch target fraction:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `0.0600`
- Final cumulative target count:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `4.8667`
- FAS lift:
  - targeted vs clean: `+0.6342`
  - targeted vs random: `+0.6065`
- Selected true mean:
  - clean: `3.7433`
  - random swap: `3.6823`
  - targeted swap: `3.5533`
- Selected target true mean under targeted swap: `2.6805`
- MAE/R2:
  - clean: `0.3768 / 0.7476`
  - random swap: `0.4044 / 0.7152`
  - targeted swap: `0.4395 / 0.6649`

Interpretation: M2 passes for the neural model. False pursuit is strongest in
the first two rounds and then partially corrects as true feedback enters the
history, which gives a useful early-budget-waste / half-life story.

### XGBoost result

- Mean batch target fraction:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `0.0067`
- Final cumulative target count under targeted swap: `0.4000`
- FAS lift vs clean: `+0.6234`
- Selected target true mean under targeted swap: `2.4142`

Interpretation: XGBoost learns a static false association at this budget, but
closed-loop allocation is weak. This is acceptable because XGBoost is a
conservative anchor, not the main neural surrogate.

## Run B: Stronger Neural Closed-Loop Signal

- Run directory: `runs/20260527T191453Z_m2-gfp-pos27-loop-mlp-50swap-bg1024-3seed`
- Swap count: `50`
- Background history size: `1024`
- Model: `mlp`

### MLP result

- Mean batch target fraction:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `0.1133`
- Final cumulative target count:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `9.6667`
- Final target count excess:
  - targeted vs clean: `+9.6667`
  - targeted vs random: `+9.6667`
- FAS lift:
  - targeted vs clean: `+0.8311`
  - targeted vs random: `+0.8297`
- Selected true mean:
  - clean: `3.7371`
  - random swap: `3.6866`
  - targeted swap: `3.3218`
- Selected target true mean under targeted swap: `2.2447`
- MAE/R2:
  - clean: `0.4626 / 0.6591`
  - random swap: `0.5298 / 0.5628`
  - targeted swap: `0.6073 / 0.4378`

Interpretation: Run B gives the stronger closed-loop false-pursuit signal.
Targeted swap causes the neural surrogate to allocate a nontrivial fraction of
early discovery budget to `pos=27`, while clean and random-swap controls select
no target records in the same horizon.

## Gate Decision

Decision: PASS for M2 feasibility.

The project now has evidence for the full feasibility chain on GFP:

1. `pos=27` is a pre-specified low-true-performance target region from M0.
2. Paired swaps preserve the label multiset exactly.
3. M1 shows neural false-association induction.
4. M2 shows closed-loop target allocation lift under true-label feedback.
5. The selected target records are not genuinely high-performing.

The main weakness is that aggregate MAE/R2 still changes noticeably in the
targeted configurations. The next robustness stage should search for a better
stealth tradeoff and add stratified audits rather than claiming that endpoint
validation is fully blind.

## Next Step

Move to M3 robustness:

- run `pos=27` with 5-10 seeds;
- add another target region, likely `pos=83` or `pos=100`;
- add wrong-target and donor-only controls;
- optimize the neural model/training setup for less validation degradation;
- add a stratified target/provenance audit to compare against aggregate MAE/R2.

