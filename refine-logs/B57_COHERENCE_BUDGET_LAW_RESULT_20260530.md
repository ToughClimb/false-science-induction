# B57 Predictive Coherence-to-Budget Law

Date: 2026-05-30

## Hypothesis

Budget misdirection should scale with a coherent conditional rewrite, not with the mere presence of label-preserving swaps. The tested empirical score is

\[
R = \rho\, (\Delta_{DT}/s_Y)\, \sqrt{m_c/n_T},
\]

where \(\rho\) is the coherent relinking fraction, \(\Delta_{DT}\) is the observed donor-target label lift in the relinked pairs, \(s_Y\) is the outcome scale, \(m_c\) is the coherent pair count, and \(n_T\) is the effective target-axis count.

## Budget And Stop Conditions

- Reuses completed B11, B22, B31, B48, B50 and B53 artifacts only.
- No new training jobs are launched.
- Acceptance: the mechanism score must outperform a swap-count-only score in the fixed-swap B48 coherence sweep and preserve positive direction on nonzero-risk cases.
- Stop condition: if cross-family residuals are large, write B57 as a semi-predictive operating law rather than a universal quantitative law.

## Main Result

- Nonzero-risk rows with positive budget excess: 21/21.
- In the fixed-swap B48 coherence sweep, the best predictor is `mechanism_risk_score` with R2=0.956; the mechanism-risk predictor has R2=0.956, while the swap-count-only score has R2=0.000.
- Across all included rows, the best single predictor is `coherent_count_score` with R2=0.189; mechanism-risk R2=0.032, swap-count-only R2=0.030.
- After normalizing each experiment family by its own maximum observed response, mechanism-risk R2=0.337; swap-count-only R2=0.009.

Interpretation: B57 supports a predictive operating law inside the controlled coherence sweep and a directional cross-family law. It also shows a hard boundary: a single global linear formula does not explain magnitudes across neural top-mean, GP-BO, RF-UCB and small public-SDL replay settings without family or policy susceptibility terms.

## Fit Summary

| target                          | predictor             |   n |   intercept |        slope |          r2 |      rmse |       mae |
|:--------------------------------|:----------------------|----:|------------:|-------------:|------------:|----------:|----------:|
| target_capacity_fraction_excess | swap_count_score      |  24 |   0.139378  | -0.102249    | 0.0296131   | 0.0698132 | 0.061123  |
| target_capacity_fraction_excess | coherent_count_score  |  24 |   0.0433036 |  0.175877    | 0.188828    | 0.0638296 | 0.051361  |
| target_capacity_fraction_excess | standardized_contrast |  24 |   0.0951507 |  0.000865225 | 0.000827016 | 0.0708412 | 0.0617022 |
| target_capacity_fraction_excess | mechanism_risk_score  |  24 |   0.082907  |  0.0118673   | 0.0318311   | 0.0697334 | 0.0602308 |

## Family-Normalized Fit Summary

| target                   | predictor             |   n |   intercept |     slope |        r2 |     rmse |      mae |
|:-------------------------|:----------------------|----:|------------:|----------:|----------:|---------:|---------:|
| family_normalized_excess | swap_count_score      |  24 |    0.516054 | 0.307973  | 0.0093574 | 0.377958 | 0.330826 |
| family_normalized_excess | coherent_count_score  |  24 |    0.146696 | 1.56418   | 0.520215  | 0.263032 | 0.203828 |
| family_normalized_excess | standardized_contrast |  24 |    0.419981 | 0.0543593 | 0.113701  | 0.3575   | 0.32349  |
| family_normalized_excess | mechanism_risk_score  |  24 |    0.364465 | 0.207     | 0.337326  | 0.309126 | 0.264899 |

## Within-Family Rank Checks

| family                  |   n |   spearman_mechanism_vs_excess |   min_risk |   max_risk |   min_excess |   max_excess |
|:------------------------|----:|-------------------------------:|-----------:|-----------:|-------------:|-------------:|
| b11_materials_dose      |   4 |                            0.4 |   1.57493  |   3.40551  |    0.0383481 |     0.147493 |
| b22_gfp_dose            |   4 |                           -0.2 |   0.384915 |   1.18413  |    0.176667  |     0.222917 |
| b48_materials_coherence |   5 |                            1   |   0        |   2.78008  |    0         |     0.132743 |
| b77_gfp_coherence       |   5 |                            1   |   0        |   0.845869 |    0         |     0.195833 |

## Leave-Family-Out Direction Check

| held_out_family                   | predictor            | target                          |   n_train |   n_test |   mean_actual |   mean_predicted |       mae | same_positive_direction   |
|:----------------------------------|:---------------------|:--------------------------------|----------:|---------:|--------------:|-----------------:|----------:|:--------------------------|
| b11_materials_dose                | mechanism_risk_score | target_capacity_fraction_excess |        20 |        4 |     0.0914454 |        0.122111  | 0.0409    | True                      |
| b22_gfp_dose                      | mechanism_risk_score | target_capacity_fraction_excess |        20 |        4 |     0.2       |        0.0613481 | 0.138652  | True                      |
| b31_cameo_replay                  | mechanism_risk_score | target_capacity_fraction_excess |        23 |        1 |     0.1125    |        0.0959178 | 0.0165822 | True                      |
| b43_materials_realistic_relinking | mechanism_risk_score | target_capacity_fraction_excess |        22 |        2 |     0.0663717 |        0.102     | 0.0578444 | True                      |
| b48_materials_coherence           | mechanism_risk_score | target_capacity_fraction_excess |        19 |        5 |     0.0716814 |        0.105416  | 0.0434262 | True                      |
| b50_materials_gp_bo               | mechanism_risk_score | target_capacity_fraction_excess |        22 |        2 |     0.0451327 |        0.138089  | 0.0929565 | True                      |
| b53_sample_replay                 | mechanism_risk_score | target_capacity_fraction_excess |        23 |        1 |     0.166667  |        0.0965095 | 0.0701571 | True                      |
| b77_gfp_coherence                 | mechanism_risk_score | target_capacity_fraction_excess |        19 |        5 |     0.0681667 |        0.105469  | 0.0804966 | True                      |

## Derived Rows

| case_id                             | family                            | domain    | policy               |   mechanism_risk_score |   family_normalized_excess |   family_susceptibility |   swap_count_score |   final_excess_count |   target_capacity_fraction_excess |
|:------------------------------------|:----------------------------------|:----------|:---------------------|-----------------------:|---------------------------:|------------------------:|-------------------:|---------------------:|----------------------------------:|
| b48_materials_coherence_000_mlp     | b48_materials_coherence           | materials | mlp_top_mean         |               0        |                 0          |               0         |           0.47036  |              0       |                        0          |
| b48_materials_coherence_025_mlp     | b48_materials_coherence           | materials | mlp_top_mean         |               0.427871 |                 0.28       |               0.0868677 |           0.47036  |              4.2     |                        0.0371681  |
| b48_materials_coherence_050_mlp     | b48_materials_coherence           | materials | mlp_top_mean         |               1.12807  |                 0.593333   |               0.0698194 |           0.47036  |              8.9     |                        0.0787611  |
| b48_materials_coherence_075_mlp     | b48_materials_coherence           | materials | mlp_top_mean         |               1.92559  |                 0.826667   |               0.0569873 |           0.47036  |             12.4     |                        0.109735   |
| b48_materials_coherence_100_mlp     | b48_materials_coherence           | materials | mlp_top_mean         |               2.78008  |                 1          |               0.047748  |           0.47036  |             15       |                        0.132743   |
| b77_gfp_coherence_000_mlp           | b77_gfp_coherence                 | gfp       | mlp_top_mean         |               0        |                 0          |               0         |           0.322749 |              0       |                        0          |
| b77_gfp_coherence_025_mlp           | b77_gfp_coherence                 | gfp       | mlp_top_mean         |               0.105331 |                 0.00851064 |               0.0158231 |           0.322749 |              0.4     |                        0.00166667 |
| b77_gfp_coherence_050_mlp           | b77_gfp_coherence                 | gfp       | mlp_top_mean         |               0.296451 |                 0.1        |               0.0660592 |           0.322749 |              4.7     |                        0.0195833  |
| b77_gfp_coherence_075_mlp           | b77_gfp_coherence                 | gfp       | mlp_top_mean         |               0.555753 |                 0.631915   |               0.222671  |           0.322749 |             29.7     |                        0.12375    |
| b77_gfp_coherence_100_mlp           | b77_gfp_coherence                 | gfp       | mlp_top_mean         |               0.845869 |                 1          |               0.231517  |           0.322749 |             47       |                        0.195833   |
| b43_materials_sorted_join_shift_mlp | b43_materials_realistic_relinking | materials | mlp_top_mean         |               0        |                 0          |               0         |           0.47036  |              0       |                        0          |
| b43_materials_block_cycle_shift_mlp | b43_materials_realistic_relinking | materials | mlp_top_mean         |               2.78008  |                 1          |               0.047748  |           0.47036  |             15       |                        0.132743   |
| b11_materials_dose_5_mlp            | b11_materials_dose                | materials | mlp_top_mean         |               1.57493  |                 0.26       |               0.024349  |           0.210352 |              4.33333 |                        0.0383481  |
| b11_materials_dose_10_mlp           | b11_materials_dose                | materials | mlp_top_mean         |               2.13176  |                 0.64       |               0.0442805 |           0.297482 |             10.6667  |                        0.0943953  |
| b11_materials_dose_25_mlp           | b11_materials_dose                | materials | mlp_top_mean         |               2.78008  |                 1          |               0.0530533 |           0.47036  |             16.6667  |                        0.147493   |
| b11_materials_dose_50_mlp           | b11_materials_dose                | materials | mlp_top_mean         |               3.40551  |                 0.58       |               0.0251198 |           0.66519  |              9.66667 |                        0.0855457  |
| b22_gfp_dose_5_mlp                  | b22_gfp_dose                      | gfp       | mlp_top_mean         |               0.384915 |                 1          |               0.579133  |           0.144338 |             53.5     |                        0.222917   |
| b22_gfp_dose_10_mlp                 | b22_gfp_dose                      | gfp       | mlp_top_mean         |               0.542232 |                 0.792523   |               0.325814  |           0.204124 |             42.4     |                        0.176667   |
| b22_gfp_dose_25_mlp                 | b22_gfp_dose                      | gfp       | mlp_top_mean         |               0.845869 |                 0.878505   |               0.231517  |           0.322749 |             47       |                        0.195833   |
| b22_gfp_dose_50_mlp                 | b22_gfp_dose                      | gfp       | mlp_top_mean         |               1.18413  |                 0.917757   |               0.172771  |           0.456435 |             49.1     |                        0.204583   |
| b50_materials_gp_ucb                | b50_materials_gp_bo               | materials | gp_ucb               |               2.78008  |                 0.821429   |               0.0146427 |           0.47036  |              4.6     |                        0.040708   |
| b50_materials_expected_improvement  | b50_materials_gp_bo               | materials | expected_improvement |               2.78008  |                 1          |               0.0178259 |           0.47036  |              5.6     |                        0.0495575  |
| b31_cameo_rf_ucb                    | b31_cameo_replay                  | cameo     | rf_ucb               |               1.15605  |                 1          |               0.0973139 |           0.36823  |              5.4     |                        0.1125     |
| b53_sample_gp_ucb                   | b53_sample_replay                 | sample    | gp_ucb               |               1.39356  |                 1          |               0.119597  |           0.57735  |              1.5     |                        0.166667   |

## Figure

- `figures/b57_coherence_budget_law.pdf`

## Safe Manuscript Claim

With total swaps and labels fixed, budget misdirection follows the coherent conditional rewrite rather than the swap count itself: in the B48 materials sweep, the mechanism-risk score explains the graded target-capacity allocation, whereas a swap-count-only score is uninformative. Across dose, GP-BO, CAMEO and SAMPLE rows, nonzero coherent-risk cases remain directionally positive, but magnitude requires a susceptibility term for domain, model, acquisition policy, and saturation.

## Unsupported Claims

- No universal quantitative law across all closed-loop systems.
- No claim that random relinking is sufficient.
- No claim that CAMEO or SAMPLE were naturally corrupt.
- No unbounded amplification claim.
