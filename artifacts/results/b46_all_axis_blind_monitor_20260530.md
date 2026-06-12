# B46 All-Axis Blind Trace Monitor

The monitor scans every enumerable axis in the selected records and calibrates each axis from clean/random controls. The injected target axis is used only after scanning to evaluate recovery.

| Dataset | Model | Control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 | Target-axis top-2 |
|---|---|---:|---:|---:|---:|---:|
| gfp | mlp | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| materials | mlp | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cameo | rf_ensemble_ucb | 0.000 | 1.000 | 0.900 | 0.900 | 0.900 |

Interpretation: all-axis monitoring relaxes the known-slice assumption but exposes the multiple-axis false-positive boundary. It is a blind triage signal, not record-level correction.
