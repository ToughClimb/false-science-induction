# Checkable Goals and Exit Criteria

Date: 2026-05-27

This file is written for autonomous goal-mode execution. The agent should keep
advancing the project until the completion checklist is satisfied, or until a
real blocker occurs: missing data, missing credentials, unavailable hardware,
unresolvable external dependency, or repeated experiment failure that falsifies
the core claim.

## Frozen Objective

Demonstrate, with traceable runs, that targeted data-record integrity failures
can induce a specified false scientific regularity in neural scientific
surrogates and cause closed-loop discovery to allocate experiments toward a
non-existent target region while ordinary aggregate audits remain
non-diagnostic or insufficiently localizing.

Do not optimize for endpoint degradation. Do not reframe the paper as generic
poisoning.

## Completion Checklist

The project is complete only when all MUST items are done.

| ID | Goal | Checkable Evidence | Status |
| --- | --- | --- | --- |
| G1 | Frozen claim and scope | `CLAIMS_AND_EXPERIMENT_SPEC.md` exists and excludes endpoint-degradation framing | DONE |
| G2 | Reproducible setup | Git repo initialized, tests pass, data SHA and commands recorded in run metadata | DONE |
| G3 | Target construction | M0 identifies at least one low-true target with exact paired-swap label preservation | DONE |
| G4 | Neural false association | M1 shows targeted > random on FAS and rank for a neural surrogate | DONE |
| G5 | Closed-loop false pursuit | M2 shows targeted > clean/random target allocation under true-label feedback | DONE |
| G6 | Oracle contradiction | Selected target records are not genuinely high-performing under true labels | DONE |
| G7 | Controls | Random swap, donor-only, target-only upper bound, and alternate targets are recorded | DONE/PARTIAL |
| G8 | Paper-facing neural support | Frozen protein-LM embedding + neural head shows static FAS/rank lift | DONE |
| G9 | Audit boundary | Label-distribution invariance and aggregate-audit limitations are documented without overclaiming | DONE/PARTIAL |
| G10 | Paper artifacts | Main tables/figures and result-to-claim audit are generated from raw run files | DONE |
| G11 | Robustness closure | Longer-horizon or additional-seed run confirms persistence or defines half-life | DONE |
| G12 | Generality support | Either a second domain/binding axis or a true null/negative target is tested | DONE |

## Phase Gates

### Phase A: Feasibility

Complete when:

- target region is pre-specified from true-label statistics;
- targeted paired swap preserves the label multiset exactly;
- neural M1 FAS lift vs random is positive;
- M2 target allocation lift vs clean and random is positive;
- true labels contradict the target-high regularity.

Current status: COMPLETE for GFP `pos=27`.

### Phase B: Paper-Feasible Core

Complete when:

- primary GFP MLP result has at least 5 seeds;
- at least one alternate target or target definition is positive;
- random and donor-only controls do not reproduce target pursuit;
- ESM/protein-LM static evidence is present;
- audit wording is scoped to what the data actually show.

Current status: COMPLETE for the GFP-focused paper-feasible core. Remaining
risk for a Nature/Science-family version is breadth beyond GFP.

### Phase C: Nature/Science-Family Ambition

Complete when:

- evidence covers either a second scientific domain or a distinct
  condition/provenance binding axis;
- mechanistic figures show model belief formation, acquisition rank lift, and
  oracle contradiction;
- a lightweight diagnostic baseline is reported honestly, even if it fails;
- every numerical claim is traceable to raw run artifacts;
- result-to-claim review says the final claims are supported.

Current status: PARTIAL. ESOL molecular-scaffold evidence now adds a second
domain, but the broad claim still requires a fresh result-to-claim review after
the ESOL artifacts are included.

## Stop Conditions

Stop and mark the goal blocked only if the same blocker persists for at least
three consecutive goal turns and there is no meaningful local work left.

Hard blockers:

- GFP data file cannot be accessed and no verified replacement is available.
- GPU/CPU execution cannot run even smoke tests.
- ESM or another external model cannot be downloaded after proxy/mirror
  attempts, and the remaining project goal explicitly requires that model.
- New experiments repeatedly show targeted swap indistinguishable from random
  swap on both FAS and target allocation.
- True oracle labels show the target region is actually high-performing, making
  the regularity not false.
- The only positive results require out-of-range labels, synthetic trigger
  fields, or post-hoc target replacement.

Non-blockers:

- MAE/R2 are partially diagnostic. Scope the audit claim instead.
- XGBoost closed-loop pursuit is weak. It is an anchor, not the main model.
- ESM static evidence is weaker than compact MLP. Report the boundary.
- Endpoint recommendation quality does not drop. This is not the main claim.

## Next Autonomous Run Order

1. Re-run result-to-claim after adding the ESOL second-domain evidence.
2. Strengthen protein-LM evidence with more seeds or a stronger ESM/ESM-C head
   only if it materially improves the paper-facing neural surrogate claim.
3. Add acquisition robustness beyond greedy top-mean.
4. Add audit-sufficiency tests against group-wise or influence-style diagnostics.
