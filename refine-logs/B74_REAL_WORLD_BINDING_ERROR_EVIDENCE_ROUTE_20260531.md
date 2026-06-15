# B74 Real-World Binding-Error Evidence Route

Date: 2026-05-31

## Purpose

Close as much of the Nature-main "real-world relevance" gap as possible without
claiming that false-science induction has already occurred in a deployed
closed-loop campaign.

The safe claim is:

> Binding and record-matching failures are documented in high-throughput
> scientific data workflows; our contribution is to show how the consequence
> changes when an AI-driven loop converts a coherent binding error into
> experimental budget.

The unsafe claim is:

> A natural closed-loop false-science event has occurred in CAMEO, SAMPLE, BEAR
> or any other named archive.

## Confirmed Literature Anchors

### 1. Baggerly and Coombes forensic bioinformatics

Reference already in `paper-nature-main/references.bib`:
`baggerly2009forensic`.

Why it matters:
- high-throughput biology case study;
- simple row/column offsets, label reversals and duplicate/mislabeled samples
  are explicitly discussed;
- the arXiv abstract states that patients in clinical trials were being
  allocated to treatment arms based on the analyzed results;
- this is the strongest "real downstream consequence" citation, but it is not
  a closed-loop AI-discovery campaign.

Safe manuscript use:
- "record-binding failures have affected high-throughput predictive workflows";
- "closed-loop discovery changes the consequence because the model can spend
  new experimental budget before independent provenance audit."

Do not write:
- "our failure has happened in the wild";
- "this proves deployed self-driving labs are currently corrupt";
- "this is a natural validation of our closed-loop mechanism."

### 2. MixupMapper

Reference already in `paper-nature-main/references.bib`:
`westra2011mixupmapper`.

Why it matters:
- detects/corrects sample mix-ups in genome-wide datasets;
- reports that correcting mix-ups can increase power to detect small genetic
  effects;
- supports the claim that sample identity and expression/genotype bindings are
  a recognized data-quality layer.

Safe manuscript use:
- "sample-matching tools exist because binding errors are operationally common
  enough to require forensic correction."

### 3. SMaSH

Reference already in `paper-nature-main/references.bib`:
`yu2019smash` in the current BibTeX appears to have the wrong author list/DOI
relative to the public BMC Genomics paper. The web-verified public title is
`SMaSH: Sample matching using SNPs in humans`, BMC Genomics 20, 1001 (2019),
DOI `10.1186/s12864-019-6332-7`.

Action needed:
- fix the BibTeX entry before final citation audit.

Why it matters:
- explicitly frames inadvertent sample swaps and barcode-adapter assignment
  errors as threats in medium-to-large omics studies;
- supports a modern, general sample-matching citation.

## Recommended Manuscript Integration

### Main text

Keep only one concise sentence in the setup paragraph:

> Documented forensic-bioinformatics cases and sample-matching tools show that
> row/column offsets, sample mix-ups and label-binding errors are a real
> high-throughput data-management class; closed-loop discovery changes their
> consequence by converting a coherent false relation into experimental budget.

The current main text already has a version of this sentence. Do not expand it
into a long historical case study in the main text.

### Supplementary Information

Add one short paragraph in `Supplementary Note 4`:

> These citations motivate the stress tests as realistic binding-error classes,
> not as evidence of corruption in the public archives used here.

This keeps the real-world support visible without weakening the central
experimental claims.

### Cover letter

Use Baggerly/Coombes as an editor hook only if needed:

> High-throughput science already has a history of forensic bioinformatics
> cases where record-binding mistakes affected downstream decisions; our result
> identifies the closed-loop analogue, where the downstream decision is the next
> experiment.

Do not say "has occurred in autonomous discovery."

## Optional Dry-Lab Experiment Route

If we decide to add a new retrospective experiment, the most feasible route is
not a faithful reconstruction of the Potti clinical analyses. It is a dry-loop
counterfactual:

1. Choose a documented sample-mix-up or row/column-offset case with public
   corrected and erroneous tables.
2. Define a simple active-learning objective over the published feature table,
   e.g. prioritize a treatment-sensitive gene signature or high-response cell
   line region.
3. Run the same loop under the erroneous binding and corrected binding.
4. Measure whether acquisition priorities differ and whether the erroneous
   priorities have lower corrected-response value.

Acceptance criteria:
- corrected and erroneous bindings are both public and versioned;
- the loop uses no hidden labels beyond the published corrected table;
- results are framed as a retrospective counterfactual, not a recreation of the
  original clinical process.

Stop condition:
- if the public data cannot unambiguously reconstruct both bindings, do not
  force the experiment; keep the literature as motivation only.

## Current Decision

For the immediate Nature-main sprint, integrate the literature as motivation
and keep the manuscript claims bounded. A full natural-binding-error replay
would be a larger B75/B76 task after the current compile/review cycle.

