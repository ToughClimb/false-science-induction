# B68 Standard Data-Quality Screen Boundary

Hypothesis: non-provenance-aware data-quality screens can flag extremeness or local feature-label inconsistency, but they do not by themselves constitute a binding-aware detector or record-level repair.

| Dataset | Screen | Metric | Clean | Random | Targeted | Targeted - max control |
|---|---|---|---:|---:|---:|---:|
| GFP B19 | feature_knn_residual | pair_auroc_mean | 0.710 | 0.708 | 0.771 | 0.061 |
| GFP B19 | feature_knn_residual | pair_topk_recall_mean | 0.040 | 0.042 | 0.264 | 0.222 |
| GFP B19 | feature_knn_residual | target_auroc_mean | 0.861 | 0.851 | 0.572 | -0.289 |
| GFP B19 | feature_knn_residual | target_topk_recall_mean | 0.036 | 0.032 | 0.000 | -0.036 |
| GFP B19 | label_z_extremeness | pair_auroc_mean | 0.982 | 0.966 | 0.982 | 0.000 |
| GFP B19 | label_z_extremeness | pair_topk_recall_mean | 0.786 | 0.760 | 0.786 | 0.000 |
| GFP B19 | label_z_extremeness | target_auroc_mean | 0.987 | 0.976 | 0.965 | -0.021 |
| GFP B19 | label_z_extremeness | target_topk_recall_mean | 0.132 | 0.128 | 0.832 | 0.700 |
| Materials B18 | feature_knn_residual | pair_auroc_mean | 0.713 | 0.716 | 0.918 | 0.202 |
| Materials B18 | feature_knn_residual | pair_topk_recall_mean | 0.406 | 0.400 | 0.500 | 0.094 |
| Materials B18 | feature_knn_residual | target_auroc_mean | 0.445 | 0.451 | 0.988 | 0.536 |
| Materials B18 | feature_knn_residual | target_topk_recall_mean | 0.000 | 0.004 | 0.680 | 0.676 |
| Materials B18 | label_z_extremeness | pair_auroc_mean | 0.782 | 0.770 | 0.782 | 0.000 |
| Materials B18 | label_z_extremeness | pair_topk_recall_mean | 0.500 | 0.478 | 0.500 | 0.000 |
| Materials B18 | label_z_extremeness | target_auroc_mean | 0.552 | 0.551 | 1.000 | 0.448 |
| Materials B18 | label_z_extremeness | target_topk_recall_mean | 0.000 | 0.000 | 0.972 | 0.972 |

## Interpretation

The screens are intentionally generic and do not use provenance, donor-target pairing, acquisition traces or the injected target name. Label z-score extremeness often ranks records that are intrinsically extreme even in clean/random histories; this is useful context but not a binding-specific signal. Feature-neighbour residuals provide a stronger warning in some materials histories, but the signal is domain-dependent and does not recover GFP target-side misbinding reliably. The result strengthens the paper's boundary claim: standard data-quality checks can be useful sanity checks, but they are not a complete replacement for binding-aware validation, calibrated trace guards or feedback-conflict triage.

## Non-Claims

- No universal stealth claim.
- No claim that all standard screens fail.
- No record-level correction.
- No calibration-free complete detector.
