# B36 Binding-Invariance Boundary Result

Date: 2026-05-30

## Status

B36 passed as the lightweight quantitative companion to the binding-invariance
theorem.

It verifies the exact boundary:

> paired misbinding preserves the marginal label multiset while rewriting the
> conditional target/donor relation.

## Hypothesis

For the completed B19 GFP and B18 materials constructions:

1. the clean and recorded label multisets over swapped pairs are identical;
2. the target-side recorded mean shifts upward by the donor-target contrast;
3. the donor-side recorded mean shifts downward by the same amount.

## Budget

- No retraining.
- No new loop execution.
- Reuse completed `triggered_swap_pairs.csv` files from B19 and B18.
- Write one deterministic analysis script and one unit test.

## Acceptance Criteria

- All evaluated pair files preserve the label multiset.
- The target recorded shift is positive.
- The donor recorded shift has the same magnitude and opposite sign.

## Stop Conditions

- Stop if pair files are missing.
- Stop if required pair-level columns are missing.
- Treat seedless materials pair files as a single construction group instead of
  rewriting existing artifacts.

## Artifacts

Script:

- `scripts/analyze_b36_binding_invariance_boundary.py`

Config:

- `configs/b36_binding_invariance_boundary_20260530.json`

Outputs:

- `review-stage/b36_binding_invariance_boundary_20260530.csv`
- `review-stage/b36_binding_invariance_boundary_20260530.json`
- `review-stage/b36_binding_invariance_boundary_20260530.md`

Tests:

- `tests/test_b36_binding_invariance_boundary.py`

## Results

| Dataset | Axis | Seeds/groups | Pairs/group | Label multiset preserved | Target recorded shift | Donor recorded shift | Clean donor-target contrast |
|---|---|---:|---:|---|---:|---:|---:|
| B19 GFP greedy | `pos=27` | 10 | 25.0 | yes | +2.774 | -2.774 | 2.774 |
| B18 materials greedy | Co axis | 1 | 25.0 | yes | +8.540 | -8.540 | 8.540 |

The materials triggered-pair artifact is seedless because the same triggered
pair construction is reused across seeds in that script; B36 treats it as one
construction group.

## Interpretation

B36 supports the theorem-level statement:

> The corruption is not a marginal label anomaly. It is a binding rewrite that
> leaves \(P(Y)\) unchanged while changing the conditional relation seen by a
> scientific model.

This is the clean type-II boundary. A label histogram can be perfectly
unchanged even though the target-side scientific relation is inverted.

## Claim Boundary

Safe claim:

> In the evaluated GFP and materials constructions, label multisets are exactly
> preserved while the target-side recorded mean is shifted upward by 2.774 and
> 8.540 units, respectively.

Do not claim:

- all audits are blind;
- held-out clean feedback cannot detect the issue;
- every real dataset can suffer this construction;
- model-capacity inevitability.

## Validation

Commands:

```bash
conda run --no-capture-output -n agentconda python scripts/analyze_b36_binding_invariance_boundary.py --config configs/b36_binding_invariance_boundary_20260530.json
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_b35_hypothesis_axis_recovery.py tests/test_b36_binding_invariance_boundary.py tests/test_no_defaults_policy.py
```

Observed test result:

```text
7 passed in 1.61s
```
