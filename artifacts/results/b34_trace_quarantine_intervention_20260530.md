# B34 Trace-Concentration Quarantine Intervention Replay

This analysis asks how much false allocation would be prevented if a closed-loop platform quarantined a monitored target/provenance slice whenever a proposed acquisition batch exceeded the maximum clean/random batch-concentration ratio.

The replay is an offline policy analysis over completed traces. It is a governance intervention estimate, not a retrained closed-loop defense.

| Dataset | Control quarantine rate | Target quarantine rate | Observed false allocation | Prevented | Residual | Prevented fraction |
|---|---:|---:|---:|---:|---:|---:|
| b18_materials_greedy_mlp | 0.000 | 0.400 | 41.20 | 41.20 | 0.00 | 1.000 |
| b19_gfp_greedy_mlp | 0.000 | 0.260 | 47.10 | 47.00 | 0.10 | 0.999 |
| b25_materials_epsilon_greedy_mlp | 0.000 | 0.380 | 38.80 | 38.80 | 0.00 | 1.000 |
| b25_gfp_epsilon_greedy_mlp | 0.000 | 0.260 | 42.40 | 41.70 | 0.70 | 0.957 |
| b31_cameo_rf_ucb | 0.000 | 0.150 | 8.00 | 6.80 | 1.20 | 0.828 |
| b32_materials_mc_dropout_ucb_mlp | 0.000 | 0.400 | 42.30 | 42.30 | 0.00 | 1.000 |
| b32_gfp_mc_dropout_ucb_mlp | 0.000 | 0.240 | 47.90 | 47.80 | 0.10 | 0.999 |

Interpretation: the same trace-concentration signal used for audit can be converted into an actionable quarantine rule when the monitored slice is available. The result is scoped to offline replay over the tested traces.
