# B33 Acquisition-Trace Concentration Detector

This audit calibrates a per-run threshold from clean and random-swap controls, then tests whether targeted-swap traces concentrate in the monitored target slice above that threshold.

| Dataset | Round | Controls | Targeted | FPR | TPR | Control max | Target mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| b18_materials_greedy_mlp | 1 | 20 | 10 | 0.000 | 1.000 | 0.443 | 70.027 |
| b19_gfp_greedy_mlp | 1 | 20 | 10 | 0.000 | 1.000 | 0.950 | 55.057 |
| b25_materials_epsilon_greedy_mlp | 1 | 20 | 10 | 0.000 | 1.000 | 1.328 | 55.327 |
| b25_gfp_epsilon_greedy_mlp | 1 | 20 | 10 | 0.000 | 1.000 | 0.950 | 45.092 |
| b31_cameo_rf_ucb | 1 | 20 | 10 | 0.000 | 1.000 | 0.266 | 2.378 |

Interpretation: this is a trace-level audit signal, not a complete defense. It assumes that a provenance or scientific slice is being monitored and that controls are available for calibration.
