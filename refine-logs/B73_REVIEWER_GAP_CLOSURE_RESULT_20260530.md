# B73 Reviewer-Gap Closure Result

## Hypotheses

1. Generic label/loss/feature/spectral screens provide useful sanity checks but do not consistently identify the target side of paired binding errors.
2. Healthy closed-loop concentration differs from false pursuit because clean/random selections retain higher true outcomes while targeted concentration accumulates low-true-value target records.
3. The feedback-conflict triage idea should also be testable on BEAR real-measurement replay traces.

## Budget and Stop Conditions

- Read existing B18, B19 and B70 artifacts only; no model retraining beyond lightweight out-of-bag screen fits on initial histories.
- Stop after generating screen, concentration and BEAR triage tables.
- Do not overwrite existing run directories or baselines.

## Standard screen summary

| Dataset | Screen | Metric | Clean | Random | Targeted | Targeted - max control |
|---|---|---|---:|---:|---:|---:|
| GFP B19 | feature_knn_residual | target_topk_recall | 0.036 | 0.032 | 0.000 | -0.036 |
| GFP B19 | label_z_extremeness | target_topk_recall | 0.132 | 0.128 | 0.832 | 0.700 |
| GFP B19 | pca_spectral_score | target_topk_recall | 0.028 | 0.028 | 0.028 | 0.000 |
| GFP B19 | rf_oob_loss_residual | target_topk_recall | 0.000 | 0.004 | 0.300 | 0.296 |
| GFP B19 | ridge_loss_residual | target_topk_recall | 0.012 | 0.012 | 0.104 | 0.092 |
| Materials B18 | feature_knn_residual | target_topk_recall | 0.000 | 0.004 | 0.680 | 0.676 |
| Materials B18 | label_z_extremeness | target_topk_recall | 0.000 | 0.000 | 0.972 | 0.972 |
| Materials B18 | pca_spectral_score | target_topk_recall | 0.000 | 0.000 | 0.000 | 0.000 |
| Materials B18 | rf_oob_loss_residual | target_topk_recall | 0.000 | 0.004 | 0.304 | 0.300 |
| Materials B18 | ridge_loss_residual | target_topk_recall | 0.000 | 0.004 | 0.496 | 0.492 |

## Healthy versus false concentration

| Dataset | Model | Mode | Selected true mean | High-true fraction | Triggered-target fraction | Triggered-target true mean |
|---|---|---|---:|---:|---:|---:|
| GFP B19 | mlp | clean | 3.665 | 0.512 | 0.000 | 1.301 |
| GFP B19 | mlp | random_swap | 3.648 | 0.479 | 0.000 | 1.301 |
| GFP B19 | mlp | targeted_swap | 3.285 | 0.330 | 0.094 | 1.301 |
| GFP B19 | tabm_mini | clean | 3.696 | 0.490 | 0.000 | -- |
| GFP B19 | tabm_mini | random_swap | 3.613 | 0.439 | 0.000 | -- |
| GFP B19 | tabm_mini | targeted_swap | 3.330 | 0.305 | 0.072 | 1.301 |
| Materials B18 | mlp | clean | 2.776 | 0.340 | 0.000 | -- |
| Materials B18 | mlp | random_swap | 2.698 | 0.324 | 0.000 | 0.400 |
| Materials B18 | mlp | targeted_swap | 2.418 | 0.270 | 0.165 | 0.049 |
| Materials B18 | tabm_mini | clean | 2.807 | 0.346 | 0.000 | -- |
| Materials B18 | tabm_mini | random_swap | 2.782 | 0.342 | 0.000 | -- |
| Materials B18 | tabm_mini | targeted_swap | 2.358 | 0.274 | 0.199 | 0.041 |

## BEAR feedback-conflict triage

| Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |
|---|---:|---:|---|---:|
| clean | 0/10 | 0/10 | PrinterNumber=3 | 4 |
| random_swap | 2/10 | 4/10 | PrinterNumber=3 | 4 |
| targeted_relink | 10/10 | 10/10 | NozzleSize=0.5 | 1 |

## Interpretation

- Generic screens are not all blind: label/loss screens can flag some target-side records, especially in materials. Their behavior is mixed and control-sensitive, so they are best treated as sanity checks rather than binding-aware stop rules.
- Clean/random runs can concentrate on high-scoring candidates, but their selected true means and high-true fractions remain higher than targeted false-pursuit runs. The pathology is not concentration alone; it is concentration on a low-true-value axis induced by record relinking.
- BEAR feedback-conflict triage ranks the small-nozzle target axis at aggregate rank 1 under targeted relinking and not under clean/random controls, extending the axis-triage result to the real-measurement stress replay.

## Non-claims

- No complete standard-defense benchmark.
- No complete detector or record-level correction.
- No claim that BEAR contains natural corruption.
