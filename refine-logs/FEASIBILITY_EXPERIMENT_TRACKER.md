# Feasibility Experiment Tracker

Status: v0.1
Date: 2026-05-27

| Run ID | Stage | Purpose | System / Variant | Seeds | Priority | Status | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | M0 | GFP target-region scan | no model | n/a | MUST | TODO | low, populated, learnable target exists |
| R1 | M0 | paired-swap accounting audit | one target, one budget | 1 | MUST | TODO | exact label multiset preservation |
| R2 | M1 | fast static sanity | XGBoost clean/random/target | 3 | MUST | TODO | targeted > random on FAS/rank |
| R3 | M1 | neural static feasibility | compact neural surrogate clean/random/target | 5 | MUST | TODO | neural FAS lift and rank lift |
| R4 | M1 | paper-facing neural static | ESM embedding + neural head | 5 | MUST | TODO | consistent FAS/rank lift |
| R5 | M2 | short closed-loop false pursuit | top-mean acquisition | 5 | MUST | TODO | target allocation lift vs clean/random |
| R6 | M2 | closed-loop persistence | top-mean, longer horizon | 5-10 | MUST | TODO | nontrivial false-pursuit half-life |
| R7 | M3 | random-swap robustness | matched random swaps | 5-10 | MUST | TODO | random does not match targeted |
| R8 | M3 | wrong-target control | non-viable or neutral target | 5 | MUST | TODO | no comparable false pursuit |
| R9 | M3 | donor-only control | donor perturbation only | 5 | SHOULD | TODO | effect not explained by donor removal |
| R10 | M3 | target-only upper bound | target high relabel, no paired preservation | 3 | SHOULD | TODO | upper-bound signal only |
| R11 | M3 | second target region | alternate motif/family/basin | 5 | SHOULD | TODO | not a one-target artifact |
| R12 | M3 | classical anchor | XGBoost/LightGBM closed-loop | 5 | SHOULD | TODO | directional support, not gatekeeper |
| R13 | Extension | plausible false records | counterfeit target-high records | 3-5 | NICE | TODO | stronger secondary mechanism |
| R14 | Extension | diagnostic pilot | stratified residual/acquisition audit | n/a | NICE | TODO | catches target-specific anomaly |

