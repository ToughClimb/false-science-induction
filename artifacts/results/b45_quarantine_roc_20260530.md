# B45 Quarantine ROC and Threshold Sensitivity

The online trace-concentration rule is re-evaluated over a threshold sweep. Targeted rows are treated as positives and clean/random rows as controls.

| Dataset | Model | Zero-FPR threshold | Zero-FPR TPR | Zero-FPR prevented | Balanced threshold | Balanced FPR | Balanced TPR |
|---|---|---:|---:|---:|---:|---:|---:|
| gfp | mlp | 2.000 | 0.980 | 1.000 | 0.000 | 0.020 | 1.000 |
| materials | mlp | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| cameo | rf_ensemble_ucb | 3.000 | 0.767 | 0.837 | 2.500 | 0.008 | 0.800 |

Interpretation: this is a sensitivity analysis for a trace stop-loss rule, not a claim of complete detection. It shows how much false allocation can be prevented as the allowed control false-positive rate changes.
