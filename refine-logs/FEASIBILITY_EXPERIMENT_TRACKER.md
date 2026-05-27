# Feasibility Experiment Tracker

Status: v0.1
Date: 2026-05-27

| Run ID | Stage | Purpose | System / Variant | Seeds | Priority | Status | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | M0 | GFP target-region scan | no model | n/a | MUST | DONE | `pos=27` passed in `runs/20260527T185522Z_m0-gfp-pos-target-scan` |
| R1 | M0 | paired-swap accounting audit | one target, one budget | 1 | MUST | DONE | exact label multiset preservation for `pos=27` candidate pairs |
| R2 | M1 | fast static sanity | XGBoost clean/random/target | 3 | MUST | DONE | strong targeted FAS/rank lift in `runs/20260527T190400Z_m1-gfp-pos27-static-xgb-mlp-3seed` |
| R3 | M1 | neural static feasibility | compact neural surrogate clean/random/target | 5 | MUST | DONE | MLP FAS/rank/top-k lift positive; best tradeoff in `runs/20260527T190706Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed` |
| R4 | M1 | paper-facing neural static | ESM-2 8M embedding + neural head | 3 | MUST | DONE | static FAS/rank lift in `runs/20260527T200102Z_m1-gfp-pos27-esm2-static-10swap-bg4096-3seed`; top-k lift absent |
| R5 | M2 | short closed-loop false pursuit | top-mean acquisition | 5 | MUST | DONE | MLP target allocation lift in `runs/20260527T191453Z_m2-gfp-pos27-loop-mlp-50swap-bg1024-3seed` |
| R6 | M2 | closed-loop persistence | top-mean, longer horizon | 5-10 | MUST | DONE | 5-seed main run confirms early false pursuit; 10-round low-budget run `runs/20260527T201103Z_m2-gfp-pos27-stealth-15swap-bg4096-mlp-10round-3seed` shows weaker persistent pursuit |
| R7 | M3 | random-swap robustness | matched random swaps | 5-10 | MUST | DONE | clean/random select zero `pos=27` target records in 5-seed M2 |
| R8 | M3 | wrong-target control | boundary targets + random-structure null | 3-5 | MUST | DONE | boundary targets run; random low-label set control weak in `runs/20260527T202252Z_m2-gfp-random-low-set-control-50swap-bg1024-3seed` |
| R9 | M3 | donor-only control | donor perturbation only | 5 | SHOULD | DONE | donor-only selects zero target records in `runs/20260527T193942Z_m2-gfp-pos27-loop-mlp-controls-50swap-bg1024-3seed` |
| R10 | M3 | target-only upper bound | target high relabel, no paired preservation | 3 | SHOULD | DONE | target-only upper bound comparable to targeted swap but changes label distribution |
| R11 | M3 | second target region | alternate motif/family/basin | 5 | SHOULD | PARTIAL | `pos=83` M2 positive; `pos=100` M1 positive |
| R12 | M3 | classical anchor | XGBoost/LightGBM closed-loop | 5 | SHOULD | TODO | directional support, not gatekeeper |
| R13 | Extension | plausible false records | counterfeit target-high records | 3-5 | NICE | TODO | stronger secondary mechanism |
| R14 | Extension | diagnostic pilot | stratified residual/acquisition audit + low-budget stealth scan | n/a | NICE | PARTIAL | label/FAS/acquisition audit generated; 10-15 swap scans reduce but do not eliminate MAE/R2 visibility |
| R15 | Paper | paper-facing evidence tables | raw run CSV to CSV/Markdown tables | n/a | MUST | DONE | latest generated in `artifacts/paper_tables/20260527T205147Z` |
| R16 | Paper | paper-facing evidence figures | raw run CSV to PNG figures | n/a | MUST | DONE | latest generated in `artifacts/paper_figures/20260527T204639Z` |
| R17 | Paper | result-to-claim review | external claim audit | n/a | MUST | DONE | verdict `partial` in `refine-logs/RESULT_TO_CLAIM_REVIEW_20260527.md` |
| R18 | M4 | second scientific domain | ESOL molecular scaffold false regularity | 3 | MUST | DONE | MLP and XGBoost positive in `refine-logs/M6_SECOND_DOMAIN_MOLECULE_ESOL_RESULT_20260527.md`; stealth weaker than GFP |
| R19 | Paper | result-to-claim review after ESOL | external claim audit | n/a | MUST | DONE | verdict remains `partial` in `refine-logs/RESULT_TO_CLAIM_REVIEW_AFTER_ESOL_20260527.md` |
| R20 | M4 | second ESOL scaffold boundary | alternate low-solubility scaffold | 3 | SHOULD | DONE | negative boundary result recorded in `refine-logs/M6_SECOND_DOMAIN_MOLECULE_ESOL_RESULT_20260527.md` |
