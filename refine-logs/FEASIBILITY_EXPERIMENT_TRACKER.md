# Feasibility Experiment Tracker

Status: v0.1
Date: 2026-05-27

| Run ID | Stage | Purpose | System / Variant | Seeds | Priority | Status | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | M0 | GFP target-region scan | no model | n/a | MUST | DONE | `pos=27` passed in `runs/20260527T185522Z_m0-gfp-pos-target-scan` |
| R1 | M0 | paired-swap accounting audit | one target, one budget | 1 | MUST | DONE | exact label multiset preservation for `pos=27` candidate pairs |
| R2 | M1 | fast static sanity | XGBoost clean/random/target | 3 | MUST | DONE | strong targeted FAS/rank lift in `runs/20260527T190400Z_m1-gfp-pos27-static-xgb-mlp-3seed` |
| R3 | M1 | neural static feasibility | compact neural surrogate clean/random/target | 5 | MUST | DONE | MLP FAS/rank/top-k lift positive; best tradeoff in `runs/20260527T190706Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed` |
| R4 | M1 | paper-facing neural static | ESM embedding + neural head | 5 | MUST | TODO | consistent FAS/rank lift |
| R5 | M2 | short closed-loop false pursuit | top-mean acquisition | 5 | MUST | DONE | MLP target allocation lift in `runs/20260527T191453Z_m2-gfp-pos27-loop-mlp-50swap-bg1024-3seed` |
| R6 | M2 | closed-loop persistence | top-mean, longer horizon | 5-10 | MUST | PARTIAL | 5-seed `pos=27` run confirms early false pursuit; longer horizon still pending |
| R7 | M3 | random-swap robustness | matched random swaps | 5-10 | MUST | DONE | clean/random select zero `pos=27` target records in 5-seed M2 |
| R8 | M3 | wrong-target control | non-viable or neutral target | 5 | MUST | PARTIAL | boundary targets run; true null control still missing |
| R9 | M3 | donor-only control | donor perturbation only | 5 | SHOULD | TODO | effect not explained by donor removal |
| R10 | M3 | target-only upper bound | target high relabel, no paired preservation | 3 | SHOULD | TODO | upper-bound signal only |
| R11 | M3 | second target region | alternate motif/family/basin | 5 | SHOULD | PARTIAL | `pos=83` M2 positive; `pos=100` M1 positive |
| R12 | M3 | classical anchor | XGBoost/LightGBM closed-loop | 5 | SHOULD | TODO | directional support, not gatekeeper |
| R13 | Extension | plausible false records | counterfeit target-high records | 3-5 | NICE | TODO | stronger secondary mechanism |
| R14 | Extension | diagnostic pilot | stratified residual/acquisition audit | n/a | NICE | TODO | catches target-specific anomaly |
