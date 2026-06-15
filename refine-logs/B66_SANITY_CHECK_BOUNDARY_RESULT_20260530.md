# B66 Sanity-Check Boundary Result

Date: 2026-05-30

## Purpose

Respond to the reviewer criticism that the paper only says marginal label
checks are blind, while realistic scientists may run simple per-batch,
per-provenance or per-slice checks.

## Hypothesis

In the primary GFP and materials paired-misbinding constructions:

1. marginal label multiset and global mean checks remain blind by construction;
2. known-slice or paired donor-target conditional checks expose the rewrite;
3. same-distribution predictive audits do not by themselves identify correct
   scientific binding;
4. same-budget true-response shortfall and trace concentration expose harm only
   after feedback or with a monitored/calibrated trace surface.

## Budget

- No new training.
- Reuse B18 and B19 primary run artifacts.
- Add one deterministic analysis script and one unit test.

## Artifacts

Script:

- `scripts/analyze_b66_sanity_check_boundaries.py`

Config:

- `configs/b66_sanity_check_boundaries_20260530.json`

Outputs:

- `review-stage/b66_sanity_check_boundaries_20260530.csv`
- `review-stage/b66_sanity_check_boundaries_20260530.json`
- `review-stage/b66_sanity_check_boundaries_20260530.md`
- `paper-nature-main/tables/table_sanity_check_boundaries.tex`

Test:

- `tests/test_b66_sanity_check_boundaries.py`

## Main Results

| Dataset | Seeds | History multiset preserved vs random | Global mean abs delta vs random | Known-slice shift vs random | Pair shift |
|---|---:|---:|---:|---:|---:|
| GFP B19 | 10 | 1.000 | 0.000000 | 2.750 | 2.774 |
| Materials B18 | 10 | 1.000 | 0.000000 | 8.500 | 8.540 |

Interpretation:

- Marginal label multiset and global recorded-label mean checks do not detect
  the primary paired swaps.
- Known target-slice mean checks detect the shift strongly, but only if the
  implicated slice/provenance group is already known and populated.
- Pairwise donor-target conditional checks detect the construction, but require
  knowing or reconstructing the implicated pairing.
- Same-distribution audit R2 remains nontrivial in targeted runs, so predictive
  validation on the recorded distribution is not a binding-correctness
  certificate.
- Trace and feedback checks detect harm or concentration but are either
  calibrated, post-feedback, or slice/axis dependent.

## Claim Boundary

Supported:

> Marginal/global checks are blind in the primary paired-swap constructions,
> while known-slice, paired conditional, trace and feedback checks can expose
> the failure under explicit assumptions.

Not supported:

- universal stealth;
- failure of every possible provenance audit;
- calibration-free deployable detection;
- natural corruption prevalence.

## Verification

Commands:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b66_sanity_check_boundaries.py \
  tests/test_no_defaults_policy.py

conda run --no-capture-output -n agentconda \
  python scripts/analyze_b66_sanity_check_boundaries.py \
  --config configs/b66_sanity_check_boundaries_20260530.json
```

Observed test result:

```text
6 passed in 1.27s
```
