# B72 Binomial Guard Replay Result

## Hypothesis

A calibration-light allocation guard that uses only candidate-pool target prevalence, batch size and a binomial over-enrichment tail probability can recover some of the online trace-quarantine signal without clean/random threshold calibration.

## Budget and Stop Conditions

- Reuse saved proposed-trace metrics from B37 GFP, B38 materials, B39 CAMEO and B70 BEAR.
- No model retraining and no modification of prior run directories.
- Stop after alpha sweep over 0.05, 0.01, 0.001 and 0.0001.

## Acceptance Criteria

- Strong support: at least GFP and materials have control seed-any flag rate <= 0.05 and targeted seed-any flag rate >= 0.8 at one alpha.
- Boundary support: external CAMEO/BEAR may be weaker or high-FPR; report as operating boundary, not a deployable detector.

## Summary

| dataset       |   alpha |   control_round_flag_rate |   control_seed_any_flag_rate |   target_round_flag_rate |   target_seed_any_flag_rate |   target_proposed_count |   target_flagged_proposed_count |   target_flagged_proposed_fraction |   target_median_first_flag_round |
|:--------------|--------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|------------------------:|--------------------------------:|-----------------------------------:|---------------------------------:|
| bear_b70      |  0.0001 |                    0.0000 |                       0.0000 |                   0.4400 |                      1.0000 |                     990 |                             751 |                             0.7586 |                           0.0000 |
| bear_b70      |  0.0010 |                    0.0000 |                       0.0000 |                   0.4600 |                      1.0000 |                     990 |                             775 |                             0.7828 |                           0.0000 |
| bear_b70      |  0.0100 |                    0.0000 |                       0.0000 |                   0.4800 |                      1.0000 |                     990 |                             796 |                             0.8040 |                           0.0000 |
| bear_b70      |  0.0500 |                    0.0000 |                       0.0000 |                   0.5200 |                      1.0000 |                     990 |                             833 |                             0.8414 |                           0.0000 |
| cameo_b39     |  0.0001 |                    0.0000 |                       0.0000 |                   0.6333 |                      0.8000 |                     377 |                             304 |                             0.8064 |                           0.0000 |
| cameo_b39     |  0.0010 |                    0.0000 |                       0.0000 |                   0.7500 |                      0.8000 |                     377 |                             356 |                             0.9443 |                           0.0000 |
| cameo_b39     |  0.0100 |                    0.0083 |                       0.0500 |                   0.7667 |                      0.9000 |                     377 |                             362 |                             0.9602 |                           0.0000 |
| cameo_b39     |  0.0500 |                    0.0167 |                       0.1000 |                   0.8000 |                      1.0000 |                     377 |                             372 |                             0.9867 |                           0.0000 |
| gfp_b37       |  0.0001 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0010 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0100 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0500 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| materials_b38 |  0.0001 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0010 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0100 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0500 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |

## Gate Result

At least one dataset/alpha satisfies the strong-support criterion locally. This supports a calibration-light warning rule for those settings, not a complete detector.

| dataset       |   alpha |   control_round_flag_rate |   control_seed_any_flag_rate |   target_round_flag_rate |   target_seed_any_flag_rate |   target_proposed_count |   target_flagged_proposed_count |   target_flagged_proposed_fraction |   target_median_first_flag_round |
|:--------------|--------:|--------------------------:|-----------------------------:|-------------------------:|----------------------------:|------------------------:|--------------------------------:|-----------------------------------:|---------------------------------:|
| bear_b70      |  0.0001 |                    0.0000 |                       0.0000 |                   0.4400 |                      1.0000 |                     990 |                             751 |                             0.7586 |                           0.0000 |
| bear_b70      |  0.0010 |                    0.0000 |                       0.0000 |                   0.4600 |                      1.0000 |                     990 |                             775 |                             0.7828 |                           0.0000 |
| bear_b70      |  0.0100 |                    0.0000 |                       0.0000 |                   0.4800 |                      1.0000 |                     990 |                             796 |                             0.8040 |                           0.0000 |
| bear_b70      |  0.0500 |                    0.0000 |                       0.0000 |                   0.5200 |                      1.0000 |                     990 |                             833 |                             0.8414 |                           0.0000 |
| cameo_b39     |  0.0001 |                    0.0000 |                       0.0000 |                   0.6333 |                      0.8000 |                     377 |                             304 |                             0.8064 |                           0.0000 |
| cameo_b39     |  0.0010 |                    0.0000 |                       0.0000 |                   0.7500 |                      0.8000 |                     377 |                             356 |                             0.9443 |                           0.0000 |
| cameo_b39     |  0.0100 |                    0.0083 |                       0.0500 |                   0.7667 |                      0.9000 |                     377 |                             362 |                             0.9602 |                           0.0000 |
| gfp_b37       |  0.0001 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0010 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0100 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| gfp_b37       |  0.0500 |                    0.0000 |                       0.0000 |                   0.9800 |                      1.0000 |                    3761 |                            3760 |                             0.9997 |                           0.0000 |
| materials_b38 |  0.0001 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0010 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0100 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |
| materials_b38 |  0.0500 |                    0.0000 |                       0.0000 |                   1.0000 |                      1.0000 |                    2229 |                            2229 |                             1.0000 |                           0.0000 |

## Non-Claims

- No calibration-free complete detector.
- No record-level correction.
- No claim that target axes are known in deployment; this replay evaluates monitored-slice statistical thresholds on saved traces.
- BEAR and CAMEO remain retrospective stress replays, not original-controller reproductions or corruption audits.