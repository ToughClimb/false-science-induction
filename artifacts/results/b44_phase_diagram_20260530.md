# B44 Minimum-Effective Corruption Phase Diagram

This table summarizes the tested swap-count boundary using existing dose-response runs. A dose is marked effective when targeted traces have both positive final acquisition excess over the configured control and positive false-association-strength lift.

| Dataset | Model | Tested doses | Min effective | Acquisition peak | Saturated/nonmonotonic |
|---|---|---|---:|---:|---|
| materials | mlp | 5,10,25,50 | 5 | 25 | True |
| materials | tabm_mini | 5,10,25,50 | 5 | 25 | True |
| gfp | mlp | 5,10,25,50 | 5 | 5 | True |
| gfp | tabm_mini | 5,10,25,50 | 5 | 5 | True |

Interpretation: the mechanism has a low tested onset in both domains, while acquisition counts saturate or become nonmonotonic at high corruption. This supports an operating-boundary claim rather than a linear-dose claim.
