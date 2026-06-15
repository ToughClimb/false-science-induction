# B82 Real Metadata-Error Coherence Audit Result

Date: 2026-05-31

## Status

B82 adds a documented real metadata-error audit. It does not show a natural
closed-loop false-science incident. It shows that public high-throughput
scientific archives can contain documented sample annotation mismatches that
are grouped by study and sometimes directionally coherent. This supports the
realism of provenance/block/coherence stress tests.

## Input

Primary source:

- Toker, Feng and Pavlidis, "Whose sample is it anyway? Widespread
  misannotation of samples in transcriptomics studies", F1000Research 2016.
- Public reproducibility repository:
  `https://github.com/min110/mislabeled.samples.identification`
- Downloaded ZIP:
  `review-stage/b82_real_error_audit/toker_mislabeled_samples.zip`
- ZIP SHA256:
  `dbb1a87b522ff56855bc70cc4660d9120ee69fb10d9d71e73f21aae5c2907cfd`

Released output files audited:

- `review-stage/b82_real_error_audit/mislabeled.samples.identification-master/output/GPL96 all information.csv`
- `review-stage/b82_real_error_audit/mislabeled.samples.identification-master/output/GPL96.97 all information.csv`
- `review-stage/b82_real_error_audit/mislabeled.samples.identification-master/output/GPL570 all inforamtion.csv`

These tables include dataset IDs, sample IDs, recorded gender annotations,
expression-derived gender calls and mismatch labels.

## Command

```bash
conda run --no-capture-output -n agentconda \
  python scripts/b82_real_metadata_error_coherence_audit.py
```

## Result

From `review-stage/b82_real_error_audit/b82_real_metadata_error_summary.json`:

| Metric | Value |
|---|---:|
| Samples | 4160 |
| Mismatched samples | 95 |
| Sample mismatch rate | 0.0228 |
| Datasets | 70 |
| Datasets with at least one mismatch | 35 |
| Dataset mismatch fraction | 0.500 |
| Datasets with >=2 mismatches and direction purity >=0.8 | 7 |
| Top mismatch-count dataset | GSE14333 |
| Top dataset mismatch count | 9 |
| Top dataset mismatch rate | 0.0310 |

The strongest directionally coherent surfaces include:

| Platform | Dataset | Samples | Mismatches | Rate | Dominant direction | Purity | Enrichment p |
|---|---|---:|---:|---:|---|---:|---:|
| GPL570 | GSE10327 | 62 | 5 | 0.0806 | annotated female -> predicted male | 0.80 | 0.0266 |
| GPL96 | GSE7638 | 160 | 5 | 0.0313 | annotated male -> predicted female | 1.00 | 0.1346 |
| GPL96 | GSE5389 | 21 | 3 | 0.1429 | annotated female -> predicted male | 1.00 | 0.0051 |
| GPL570 | GSE22138 | 63 | 3 | 0.0476 | annotated female -> predicted male | 1.00 | 0.2446 |
| GPL570 | GSE55609 | 24 | 2 | 0.0833 | annotated female -> predicted male | 1.00 | 0.1376 |
| GPL96 | GSE20295 | 64 | 2 | 0.0313 | annotated male -> predicted female | 1.00 | 0.2931 |
| GPL570 | GSE16581 | 68 | 2 | 0.0294 | annotated female -> predicted male | 1.00 | 0.5543 |

## Interpretation

This is a real documented metadata-error surface, not an attack and not a
closed-loop replay. Its value for the paper is conceptual and evidential:

- real public high-throughput datasets have documented object/metadata
  mismatches;
- the mismatches are not uniformly isolated across the full corpus: 35/70
  datasets contain at least one mismatch;
- some studies contain multiple mismatches with a dominant direction, giving a
  real analogue of a coherent metadata/provenance surface;
- such surfaces justify stress tests based on coherent plate, batch, sorted-key
  or provenance relinking.

## Safe Claim

> A re-audit of a public transcriptomic sample-misannotation release found 95
> expression/metadata sex mismatches across 4160 samples and 35/70 datasets
> with at least one mismatch; seven datasets contained at least two mismatches
> with direction purity at least 0.8. This documents real coherent metadata
> error surfaces in high-throughput science and motivates closed-loop binding
> stress tests.

## Claims Not Supported

- This does not show false-science induction occurred in a deployed closed-loop
  campaign.
- This does not audit CAMEO, SAMPLE, BEAR, GFP or Matbench for natural
  corruption.
- This does not identify corrected record-level bindings beyond the released
  expression-derived gender mismatch labels.
- This does not establish a universal rate of scientific metadata errors.

## Artifacts

- `scripts/b82_real_metadata_error_coherence_audit.py`
- `tests/test_b82_real_metadata_error_coherence_audit.py`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_rows.csv`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_by_dataset.csv`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_summary.csv`
- `review-stage/b82_real_error_audit/b82_real_metadata_error_summary.json`
- `paper-nature-main/tables/table_b82_real_metadata_error_audit.tex`

## Verification

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b82_real_metadata_error_coherence_audit.py \
  tests/test_no_defaults_policy.py
```

Result: `8 passed`.
