# B76 Blind Binomial Axis Scan

## Hypothesis

A candidate-pool normalized all-axis binomial scan can identify over-enriched scientific axes in proposed closed-loop batches without pre-naming the injected target axis during scanning.

## Budget and Stop Conditions

- Reuse B37 GFP, B38 materials and B39 CAMEO online traces.
- Reconstruct each round's candidate pool from saved history, audit and executed-selection artifacts.
- Stop after one Bonferroni-corrected alpha threshold.

Alpha: 0.05

## Summary

| dataset       | model           |   trace_count |   control_trace_count |   target_trace_count |   control_any_axis_flag_rate |   target_any_axis_flag_rate |   target_axis_flag_rate |   target_axis_top1_rate |   target_axis_top2_rate |   target_axis_top5_rate |   control_target_axis_flag_rate |   target_median_first_flag_round |   target_flagged_axis_allocation_fraction |   mean_flagged_axes_per_target_trace |
|:--------------|:----------------|--------------:|----------------------:|---------------------:|-----------------------------:|----------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|--------------------------------:|---------------------------------:|------------------------------------------:|-------------------------------------:|
| gfp_b37       | mlp             |            30 |                    20 |                   10 |                       1.0000 |                      1.0000 |                  1.0000 |                  1.0000 |                  1.0000 |                  1.0000 |                          0.0000 |                           0.0000 |                                    0.9995 |                               8.0000 |
| materials_b38 | mlp             |            30 |                    20 |                   10 |                       1.0000 |                      1.0000 |                  1.0000 |                  1.0000 |                  1.0000 |                  1.0000 |                          0.0000 |                           0.0000 |                                    1.0000 |                              11.6000 |
| cameo_b39     | rf_ensemble_ucb |            30 |                    20 |                   10 |                       0.8000 |                      1.0000 |                  0.9000 |                  1.0000 |                  1.0000 |                  1.0000 |                          0.0500 |                           0.0000 |                                    0.9602 |                               4.8000 |

## Interpretation

The target axis is used only for after-the-fact evaluation. Low control any-axis flagging with high targeted-axis recovery supports a blind warning signal; high control flagging marks a multiple-axis boundary rather than a deployable detector.

## Non-Claims

- Not a complete detector or complete defense.
- Not record-level correction.
- Not evidence that public archives are corrupt.
- Not a faithful reproduction of original closed-loop controllers.
