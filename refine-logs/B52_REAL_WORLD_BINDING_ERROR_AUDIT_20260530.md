# B52 Real-World Binding and Provenance Error Audit

Date: 2026-05-30

## Status

B52 supports the realism framing, not a prevalence claim.

The literature and data-package audit supports this bounded statement:

> Sample-binding, provenance, batch, and source-assignment errors are real
> workflow classes in high-throughput science and scientific data
> infrastructure. They motivate structured retrospective relinking stress
> tests in closed-loop discovery systems.

It does not support:

- natural coherent relinking occurred in the systems we study;
- CAMEO or SAMPLE were corrupt;
- universal vulnerability or universal stealth;
- live wet-lab validation of our detector or intervention.

## What Counts as Evidence Here

B52 separates three evidence levels.

| Level | Supported? | Role in paper |
|---|---:|---|
| Binding/provenance errors are known workflow classes | yes | Introduction and motivation |
| Public closed-loop archives can be stress-tested by relinking real records and real measurements | yes | Retrospective external validation |
| Our exact coherent relinking has occurred naturally in a deployed campaign | no | Do not claim |

## External Workflow Classes

### High-throughput sample mix-ups

Forensic bioinformatics and genomics papers document that sample identity,
labeling, and cross-assay matching errors can materially change scientific
conclusions. The relevant analogy is not "our attack happened", but that
modern scientific pipelines already need sample-matching checks because the
object-to-measurement binding can fail.

Local bibliography already includes:

- `baggerly2009forensic`
- `westra2011mixupmapper`
- `yu2019smash`

Safe manuscript use:

> High-throughput sciences have long treated sample mix-ups as a data-integrity
> problem; closed-loop AI changes the consequence by turning a binding mistake
> into an action policy.

### Provenance and data governance

Provenance work in databases and e-science formalizes why scientific records
need lineage, source, and transformation context. Datasheets and materials
informatics papers make the same point at the dataset-governance level.

Local bibliography already includes:

- `buneman2001why`
- `simmhan2005survey`
- `gebru2021datasheets`
- `ramprasad2017machine`

Safe manuscript use:

> Our stress tests instantiate a provenance-level failure mode: the measured
> value can be real, and the object can be real, while the binding between
> object, condition/source, and value is wrong.

### Closed-loop AI-for-science archives

Two public archives are now central to the realism evidence.

| Archive | Local status | What it supports | Boundary |
|---|---|---|---|
| CAMEO Fe-Ga-Pd materials campaign | B31/B39 complete | External real closed-loop materials data can exhibit the same mechanism under retrospective relinking | Surrogate replay, not an audit of the original campaign |
| SAMPLE protein self-driving lab | downloaded and inspected | Real self-driving-lab protein archive with robot assignments, per-agent sequence tables, and round traces | B53 must be framed as a reduced-pool retrospective replay unless full robot code is exactly reproduced |

SAMPLE local artifacts:

- `review-stage/SAMPLE_code-1.0.0_github_20260530.zip`
- SHA256: `b1018ddde2a4e2ea82122174932c5c997cbe4199ea8934bd1e73e2d186fb1549`
- extracted root: `review-stage/SAMPLE_code-1.0.0/`
- `Experiment_Summary.csv`: 20 rounds, 12 sequence assignments per round
- `Seq_Data_1.csv`..`Seq_Data_4.csv`: four agent-specific sequence tables
- numeric T50 records across agents: 105 measurements over 59 unique sequence IDs
- source-data files:
  - `review-stage/sample_source/source_data_fig3.xlsx`
  - `review-stage/sample_source/source_data_fig4.xlsx`
  - `review-stage/sample_source/supp_data3.csv`
  - `review-stage/sample_source/supplementary_information.pdf`

## Why This Matters for the Paper

B52 changes the paper's realism posture from:

> We created synthetic corruptions in benchmarks.

to:

> We isolate a mechanism using controlled corruptions, and then stress-test the
> same mechanism in public scientific closed-loop archives whose schemas contain
> the same object-assignment, source, and provenance surfaces that real
> laboratories must govern.

This is still not a real-world incident claim. It is a stronger and more honest
external-validity claim.

## Recommended Manuscript Wording

Use:

> Motivated by known sample-mix-up and provenance failures in high-throughput
> science, we ask whether a closed-loop discovery policy is sensitive to
> record-valid but binding-invalid relinking of real measurements.

Use:

> In external retrospective replays, the original archive is treated as clean;
> the relinking is a controlled stress test applied to real records and real
> measured values.

Avoid:

> Real closed-loop laboratories are already suffering this attack.

Avoid:

> The original SAMPLE or CAMEO campaign contained binding corruption.

Avoid:

> Provenance relinking is universally stealthy.

## B53 Decision

SAMPLE is worth running because it is the closest available public archive to a
real self-driving protein laboratory. The first honest B53 should be a
reduced-pool retrospective replay over the subset with observed numeric T50:

- train on a small history of real sequence/T50 records;
- relink a small number of high donor T50 measurements onto a low target axis;
- preserve the label multiset exactly;
- compare clean, random relinking, and targeted relinking;
- score candidates with a SAMPLE-style GP-UCB surrogate;
- update with true feedback from the archive after each replayed acquisition.

Acceptance criteria:

- every mode's initial recorded labels preserve the clean label multiset where
  paired swaps are used;
- targeted relinking yields higher final target-axis acquisition than random
  relinking in most seeds;
- clean/random do not already heavily select the target axis;
- selected target-axis records have low true T50 relative to donor records.

If this fails, B53 should be reported as a boundary and not integrated as a
positive main-text result.
