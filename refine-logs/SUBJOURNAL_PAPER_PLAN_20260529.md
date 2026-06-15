# Subjournal Paper Plan

Date: 2026-05-29

## Working Title

False scientific regularities from targeted data-binding errors in neural closed-loop discovery

## Venue Fit

Primary fit:

- Nature Machine Intelligence
- Nature Computational Science
- Science Advances

Best framing:

> Scientific integrity and reliability of neural AI-for-science discovery loops.

Not the best framing:

> Generic data poisoning, benchmark attack, or endpoint performance degradation.

## Abstract Skeleton

Scientific discovery models are increasingly used in closed-loop systems that decide which experiments to run next. These systems assume that input records, experimental outputs, and provenance metadata are correctly bound. We show that small numbers of realistic binding errors can induce a different failure mode: a neural surrogate learns a false scientific regularity and closed-loop acquisition pursues a non-existent phenomenon. In controlled protein and materials discovery benchmarks, targeted paired misbindings and provenance-like distributed triggers cause models to allocate experiments toward specified low-performing mutation-position or composition basins. The effect replicates across domains, target basins, neural architectures, swap budgets, trigger strengths, and 10-seed confirmations. Standard validation metrics and label-distribution checks are not sufficient safeguards: some conditional settings preserve plausible non-trigger audit behavior, while other settings reveal detectable degradation. Under true feedback, false pursuit persists with attenuation or early saturation. These results identify input-output/provenance binding as a distinct scientific-integrity risk for neural closed-loop discovery.

## Claims-Evidence Matrix

| Contribution | Evidence | Figure/Table | Status |
|---|---|---|---|
| C1. Define false scientific regularity induction as a binding-error failure mode. | Formal setup and paired-swap/trigger mechanisms. | Fig. 1, Methods | Supported |
| C2. Show closed-loop false pursuit in protein and materials benchmarks. | GFP B19/B21; Materials B18. | Fig. 2, Table 1 | Supported |
| C3. Show mechanism robustness across basins, doses, trigger strengths, and architectures. | Materials B10/B11/B12/B24; GFP B22. | Fig. 3, Fig. 4, Table 2 | Supported |
| C4. Characterize detection boundary. | Audit R2, non-trigger R2, label histogram preservation, XGBoost/simple baselines. | Fig. 5, Table 3 | Partially supported; needs matrix-style presentation |
| C5. Characterize feedback dynamics. | Materials B18 long loop; GFP B21. | Fig. 6 | Supported with attenuation/saturation |

## Section Plan

### §1 Introduction

Purpose:

- Introduce closed-loop scientific discovery systems as decision-making systems, not passive predictors.
- State the gap: current robustness discussions focus on noisy labels, outliers, or endpoint degradation, but not false scientific regularity induction from binding errors.
- Present two realistic provenance/binding scenarios: protein assay joins and materials database curation.

Core message:

> The danger is not that the model performs worse; the danger is that it learns and pursues the wrong science.

Contributions:

1. Define targeted input-output/provenance misalignment as a false-science induction mechanism.
2. Demonstrate closed-loop false pursuit in GFP and materials discovery.
3. Show robustness across basins, doses, triggers, architectures, and seed-level statistics.
4. Map the detection boundary: validation metrics sometimes remain plausible and sometimes expose degradation.
5. Show that true feedback attenuates but does not immediately eliminate false pursuit.

Estimated length: 1.5-2 pages.

### §2 Problem Formulation

Define:

- Scientific record: `(x_i, c_i, p_i, y_i)` where `x` is input, `c` condition, `p` provenance, `y` output.
- Correct binding: `y_i` belongs to `(x_i, c_i, p_i)`.
- Binding failure: `x` and `y` are individually real but associated with the wrong condition/provenance/input.
- Targeted paired swap: low target-basin records receive high donor labels and donors receive low target labels, preserving the label multiset.
- Distributed trigger: a provenance-like state embedded across existing features.
- False scientific regularity: positive learned association between a non-causal target basin and high predicted performance.

Metrics:

- Final cumulative target or triggered-target acquisitions.
- False-association strength: mean target prediction minus matched-control prediction.
- Trigger-toggle delta.
- Target rank percentile and top-k fraction.
- Audit MAE/R2 and non-trigger audit R2.
- Label-multiset preservation.

Estimated length: 1.5 pages.

### §3 Experimental Systems

Datasets:

- GFP protein fitness benchmark.
- Matbench experimental band-gap benchmark.
- ESOL may appear only as secondary/appendix weak transfer evidence, not a central pillar.

Models:

- MLP: neural tabular baseline.
- TabM-mini: modern tabular neural model.
- FT-Transformer-style: transformer-style tabular architecture for materials architecture extension.
- XGBoost: conservative non-neural anchor, not main claim.

Closed-loop protocol:

- Initial history.
- Targeted paired misbinding.
- Candidate pool and audit split.
- Acquisition rounds.
- True feedback after selected experiments.

Estimated length: 1.5 pages.

### §4 Main Results

#### §4.1 Protein And Materials False Pursuit

Evidence:

- GFP B19/B21.
- Materials B18 short/long.

Main result sentence:

> In both protein and materials domains, targeted misbinding causes neural surrogates to repeatedly select low-performing non-causal basins that clean and random-swap controls almost never select.

#### §4.2 Dose, Trigger Strength, And Basin Robustness

Evidence:

- Materials B10 multi-basin.
- Materials B11 dose response.
- Materials B12 trigger ablation.
- GFP B22 dose response.

Main result sentence:

> False-association strength increases with misbinding budget and survives weaker distributed triggers, while acquisition saturates in some settings.

#### §4.3 Architecture Generality

Evidence:

- MLP and TabM-mini in GFP/materials.
- FT-Transformer-style materials B24.

Main result sentence:

> The materials false-science effect persists in a transformer-style tabular neural surrogate, with 41.1 final triggered-target acquisitions versus 0.0 under clean/random controls.

### §5 Detection Boundary

Purpose:

- Prevent overclaiming.
- Show why ordinary validation is insufficient as a sole safeguard.
- Separate label-distribution invisibility from endpoint stealth.

Required table:

- Rows: GFP B19, GFP B22, Materials B18, Materials B24, Materials B11/B12.
- Columns: label multiset, global audit R2, non-trigger audit R2, target-slice diagnostics, trigger-toggle delta, simple baseline.
- Entries: PASS/WARN/FAIL plus numbers.

Main result sentence:

> Detection is configuration-dependent: paired swaps preserve label distributions, and successful trigger-gated settings can preserve non-trigger audit behavior, but some stronger or architecture-shifted settings visibly degrade aggregate audit metrics.

### §6 Feedback Dynamics

Evidence:

- Materials B18 long loop.
- GFP B21 long loop.

Main result sentence:

> True feedback attenuates or saturates the false pursuit, but does not eliminate the cumulative false allocation over 10 rounds.

### §7 Discussion

Must include:

- Why this is a scientific-integrity issue.
- Threat model and realistic binding/provenance scenarios.
- Why endpoint validation alone is insufficient.
- Defense implications: provenance audits, binding consistency checks, target-slice validation, trigger/provenance perturbation tests, data-lineage checks.

Limitations:

- Controlled benchmark corruption, not naturally observed corruption.
- Does not prove universal stealth.
- Does not prove unbounded closed-loop failure.
- Domain scope is protein and materials benchmarks.

### §8 Conclusion

One-paragraph conclusion:

- Restate false science induction.
- Emphasize bounded but serious risk.
- Call for provenance-aware validation in AI-driven scientific discovery.

## Figure And Table Plan

| ID | Type | Content | Data Source | Priority |
|---|---|---|---|---|
| Fig. 1 | Conceptual schematic | Correct binding vs paired misbinding vs distributed provenance trigger; closed-loop pursuit of false basin. | Manual / FigureSpec | High |
| Fig. 2 | Cross-domain bar chart | Final target/triggered-target acquisitions in GFP and materials, clean/random/targeted, MLP and TabM-mini. | B20 figures / B18/B19/B21 stats | High |
| Fig. 3 | Dose and trigger curves | Materials and GFP dose response; materials trigger-strength ablation. | B11/B12/B22 aggregates | High |
| Fig. 4 | Multi-basin and architecture extension | Materials multi-basin replication plus FT-Transformer-style B24. | B10 aggregate, B24 summary/statistics | High |
| Fig. 5 | Detection boundary | Audit R2 and non-trigger R2 by setting; PASS/WARN/FAIL matrix. | B18/B19/B22/B24 summaries | High |
| Fig. 6 | Long-loop dynamics | 10-round cumulative false-pursuit trajectories; attenuation/saturation. | B18 long, B21 long, B20 trajectory figure | Medium |
| Table 1 | Main results table | Final counts, FAS lift, trigger delta, audit R2 for flagship settings. | B18/B19/B24 summaries | High |
| Table 2 | Statistical confirmation | Seed differences, bootstrap CIs, sign-flip p-values. | B18/B19/B21/B24 statistics | High |
| Table 3 | Claims and caveats | Supported vs unsupported claims. | Claim matrix | High |

## Minimal Remaining Work Before Drafting

1. Generate the detection-boundary table as CSV/figure from existing summary files.
2. Create Fig. 1 as a clean mechanism schematic.
3. Clean the current B20/B22/B24 figure/report set into a unified `paper/` artifact directory.
4. Run an external result-to-claim review on this narrowed manuscript plan.
5. Start writing the LaTeX manuscript only after the detection-boundary table is generated.

Status update:

- Item 1 is complete: `runs/subjournal_detection_boundary_20260529.csv` and `docs/figures/subjournal_detection_boundary.png/svg`.
- External review of the narrowed plan is complete and recommended drafting rather than new experiments.

## Positioning

Related work buckets:

- AI-driven closed-loop scientific discovery and Bayesian/active learning.
- Data poisoning and label noise in machine learning.
- Scientific data provenance, metadata, and dataset curation.
- Robustness and validation of AI-for-science surrogates.

Positioning sentence:

> Unlike generic poisoning work that measures endpoint degradation, this paper studies a scientific-integrity failure mode: neural discovery systems can internalize and act on false regularities created by realistic record-binding errors.

## One-Sentence Submission Framing

This paper identifies and experimentally characterizes false scientific regularity induction: a bounded but reproducible failure mode in which neural closed-loop discovery systems learn and pursue non-existent scientific phenomena from small numbers of realistic input-output or provenance binding errors.
