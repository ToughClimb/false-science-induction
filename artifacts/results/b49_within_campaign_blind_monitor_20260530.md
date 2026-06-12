# B49 Within-Campaign Blind Trace Monitor

Thresholds are derived from each campaign's own early proposed traces and same-round peer-axis distribution. Clean/random modes are used only after the fact to estimate false alarms.

| Dataset | Model | Eval-only control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |
|---|---|---:|---:|---:|---:|---:|
| gfp | mlp | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| materials | mlp | 0.925 | 1.000 | 1.000 | 1.000 | 1.000 |
| cameo | rf_ensemble_ucb | 0.838 | 0.375 | 0.200 | 0.800 | 0.800 |

Interpretation: this removes the clean-control threshold assumption but remains a triage heuristic over enumerable axes, not a complete detector or record-level correction.
