# B15 Statistics And Effect-Size Result

Date: 2026-05-29

## Question

B15 adds uncertainty estimates and paired seed-level checks for the main materials trigger results. The purpose is to avoid relying only on mean tables.

## Artifacts

Config:

- `configs/b15_statistics_aggregate_20260529.json`

Command:

```bash
conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b15_statistics_aggregate_20260529.json
```

Outputs:

- `runs/b15_statistics_aggregate_20260529.csv`
- `runs/b15_statistics_aggregate_20260529.json`

## Results

| Effect | Model | Seed-level differences | Mean difference | Bootstrap 95% CI | Sign-flip p |
|---|---|---:|---:|---:|---:|
| B12 scale 0.01 final count vs random | MLP | [42, 41, 42] | 41.7 | [41.0, 42.0] | 0.25 |
| B12 scale 0.01 final count vs random | TabM-mini | [50, 49, 50] | 49.7 | [49.0, 50.0] | 0.25 |
| B14 post-round-5 gain vs random | MLP | [3, 6, 4] | 4.3 | [3.0, 6.0] | 0.25 |
| B14 post-round-5 gain vs random | TabM-mini | [2, 1, 1] | 1.3 | [1.0, 2.0] | 0.25 |

All tested seed-level differences are positive.

## Interpretation

B15 supports effect-size robustness but not formal significance claims. With only three seeds, the exact sign-flip two-sided test has a minimum attainable p-value of 0.25 when all signs align. Therefore the paper should not write "statistically significant" for these 3-seed pilot statistics.

The defensible statement is:

Across all tested seeds in the main B12 and B14 materials settings, targeted mode exceeds random controls by large margins, and bootstrap confidence intervals over seeds remain positive. Larger-seed confirmation is still desirable before a final submission package.

## Claim Impact

B15 supports:

- The observed effects are not driven by one bad seed in the tested configurations.
- The main targeted-vs-random margins are large and same-signed across seeds.

B15 does not support:

- A conventional p < 0.05 claim.
- Treating 3-seed bootstrap intervals as a replacement for larger independent replication.
