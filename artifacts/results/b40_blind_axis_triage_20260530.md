# B40 Blind Hypothesis-Axis Triage

Axes are scored without using the injected target. The target-axis aliases are used only after ranking to evaluate whether the blind ranking recovered the implicated axis.

Score: selected-budget fraction multiplied by positive true-feedback deficit.

| Dataset | Model | Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |
|---|---|---|---:|---:|---|---:|
| b19_gfp_greedy | mlp | clean | 0/10 (0.000) | 0/10 (0.000) | pos=219 | 36 |
| b19_gfp_greedy | mlp | random_swap | 0/10 (0.000) | 0/10 (0.000) | pos=219 | 28 |
| b19_gfp_greedy | mlp | targeted_swap | 8/10 (0.800) | 10/10 (1.000) | pos=27 | 1 |
| b19_gfp_greedy | tabm_mini | clean | 0/10 (0.000) | 0/10 (0.000) | pos=110 | 182 |
| b19_gfp_greedy | tabm_mini | random_swap | 0/10 (0.000) | 0/10 (0.000) | pos=59 | 18 |
| b19_gfp_greedy | tabm_mini | targeted_swap | 9/10 (0.900) | 10/10 (1.000) | pos=27 | 1 |
| b18_materials_greedy | mlp | clean | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 129 |
| b18_materials_greedy | mlp | random_swap | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 55 |
| b18_materials_greedy | mlp | targeted_swap | 10/10 (1.000) | 10/10 (1.000) | element=Co | 1 |
| b18_materials_greedy | tabm_mini | clean | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 122 |
| b18_materials_greedy | tabm_mini | random_swap | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 126 |
| b18_materials_greedy | tabm_mini | targeted_swap | 10/10 (1.000) | 10/10 (1.000) | element=Co | 1 |
| b31_cameo_rf_ucb | rf_ensemble_ucb | clean | 9/10 (0.900) | 10/10 (1.000) | dft_region=2 | 1 |
| b31_cameo_rf_ucb | rf_ensemble_ucb | random_swap | 9/10 (0.900) | 9/10 (0.900) | dft_region=2 | 1 |
| b31_cameo_rf_ucb | rf_ensemble_ucb | targeted_swap | 10/10 (1.000) | 10/10 (1.000) | dft_region=2 | 1 |

Interpretation: targeted-trace recovery means the false hypothesis itself acts as a diagnostic probe for the implicated scientific axis. This is axis-level triage, not causal discovery or record-level repair.
