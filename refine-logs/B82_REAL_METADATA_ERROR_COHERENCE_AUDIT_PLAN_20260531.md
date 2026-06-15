# B82 Real Metadata-Error Coherence Audit Plan

Date: 2026-05-31

## Hypothesis

Documented sample-annotation errors in public high-throughput data are not
only isolated record anomalies. In at least some public transcriptomic
archives, annotation/genetic-sex mismatches are concentrated by study and
direction. Such concentration is a real-world analogue of coherent binding
surfaces: it motivates closed-loop stress tests in which a block or provenance
axis is wrong, without claiming that our studied closed-loop archives are
naturally corrupt.

## Input Version

Primary input is the public reproducibility repository for Toker, Feng and
Pavlidis, "Whose sample is it anyway? Widespread misannotation of samples in
transcriptomics studies", F1000Research 2016.

Downloaded artifact:

- `review-stage/b82_real_error_audit/toker_mislabeled_samples.zip`
- SHA256:
  `dbb1a87b522ff56855bc70cc4660d9120ee69fb10d9d71e73f21aae5c2907cfd`
- Extracted root:
  `review-stage/b82_real_error_audit/mislabeled.samples.identification-master`

The audit uses only the released output CSV files:

- `output/GPL96 all information.csv`
- `output/GPL96.97 all information.csv`
- `output/GPL570 all inforamtion.csv`

These files include study/dataset IDs, sample IDs, annotated gender,
genetic-expression-derived gender calls and mismatch indicators.

## Smallest Falsifiable Experiment

Aggregate the released mismatch calls and measure:

1. total sample mismatch rate;
2. number and fraction of datasets with at least one mismatch;
3. whether mismatch counts are concentrated in a minority of datasets;
4. whether mismatch directions are coherent within datasets
   (`annotated female -> predicted male` or `annotated male -> predicted female`);
5. whether the released data contain concrete high-confidence dataset-level
   error surfaces suitable for narrative anchoring.

## Metrics

- `sample_count`
- `mismatch_count`
- `mismatch_rate`
- `dataset_count`
- `datasets_with_mismatch`
- `dataset_mismatch_fraction`
- top dataset mismatch count and rate
- top dataset dominant direction and purity
- number of datasets with at least two mismatches and direction purity >= 0.8
- binomial enrichment p-value for each dataset's mismatch count relative to the
  global platform-level rate

## Budget

- One deterministic CPU script.
- No model training.
- No external wet-lab or hidden data.
- Outputs under `review-stage/b82_real_error_audit/`.

## Acceptance Criteria

Strong support:

- at least 40% of datasets have at least one released mismatch, consistent
  with the paper-level claim;
- at least one dataset has >= 5 mismatches;
- at least one dataset has >= 2 mismatches with dominant direction purity >=
  0.8;
- output includes a clear table of top mismatch-concentrated datasets.

Boundary support:

- if mismatches are sparse or directionally mixed, use the result only as
  literature grounding and do not add it as a quantitative audit.

## Stop Conditions

- Stop if the repository output files cannot be parsed reproducibly.
- Stop if column meanings are ambiguous after inspecting the released README.
- Do not infer original corrected sample identities beyond the released
  annotated-vs-genetic-sex mismatch calls.
- Do not claim closed-loop corruption, prevalence in CAMEO/SAMPLE/BEAR/GFP, or
  record-level correction.

## Planned Outputs

- `scripts/b82_real_metadata_error_coherence_audit.py`
- `tests/test_b82_real_metadata_error_coherence_audit.py`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_summary.csv`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_by_dataset.csv`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_summary.json`
- `refine-logs/B82_REAL_METADATA_ERROR_COHERENCE_AUDIT_RESULT_20260531.md`

