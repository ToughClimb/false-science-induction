# B59 Feedback-Discordance Blind Monitor

Thresholds are derived from each campaign's own early traces and same-round peer-axis scores. Clean/random modes are used only after the fact to estimate false alarms.

| Dataset | Model | Eval-only control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |
|---|---|---:|---:|---:|---:|---:|
| gfp | mlp | 0.138 | 0.725 | 0.725 | 0.875 | 0.875 |
| materials | mlp | 0.863 | 0.625 | 0.625 | 0.975 | 0.975 |
| cameo | rf_ensemble_ucb | 0.675 | 0.300 | 0.100 | 0.800 | 0.800 |

Interpretation: this is a prospective-style triage test that removes external clean-control calibration but still relies on executed or otherwise trusted true feedback. It is not a complete detector, defense, or record-level correction.
