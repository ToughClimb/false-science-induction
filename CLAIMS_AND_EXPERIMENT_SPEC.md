# Claims and Experiment Specification

Status: frozen v0.1
Date: 2026-05-27

This document freezes the research claim and the experiment specification for
the false-science induction project. It is intentionally not an implementation
plan. Any future implementation should be checked against this document before
adding models, datasets, metrics, or baselines.

## 1. Problem Anchor

### Bottom-line problem

AI-driven scientific discovery systems can form and act on false scientific
regularities when scientific records have targeted data-integrity failures.
The failure is not merely worse prediction accuracy. The system can learn a
specific non-existent association and allocate experimental budget toward it.

### Must-solve bottleneck

Existing endpoint metrics and aggregate validation checks can miss whether a
closed-loop discovery system has learned the wrong scientific explanation. We
need evidence that a model has internalized a specified false regularity and
that the acquisition loop pursues it as if it were real.

### Non-goals

- Do not frame the paper as generic data poisoning or attack success rate.
- Do not make endpoint degradation the primary success criterion.
- Do not claim universal vulnerability of all foundation models or all
  closed-loop systems.
- Do not rely on unrealistic labels, out-of-range values, or visible artificial
  triggers as the main mechanism.
- Do not let model popularity override the false-science claim.

### Success condition

The project succeeds if a small number of realistic targeted record-integrity
failures cause neural scientific surrogates to learn a specified false
association, and the closed-loop system allocates more experiments toward the
corresponding non-existent phenomenon, while common aggregate audits remain
plausible.

## 2. Frozen Main Claim

### Primary claim

Targeted data-record integrity failures can implant specified false scientific
regularities into neural scientific surrogates, causing closed-loop discovery
systems to allocate experiments toward non-existent motifs, conditions, or
provenance basins while standard aggregate checks remain non-diagnostic.

### Conservative form of the claim

Even without fabricating labels, real input-output misbinding is sufficient to
induce false regularities: the input is real, the output is real, but the
record-level binding is wrong.

### Expanded form of the claim

Plausible false records provide a second, stronger mechanism. They should be
treated as a higher-severity data-record integrity failure, not as the main
paper's only mechanism.

### One-sentence paper thesis

True data can still produce false science when record bindings are wrong:
small, realistic, targeted integrity failures can make neural closed-loop
discovery systems learn and pursue specified non-existent phenomena.

## 3. Mechanism Hierarchy

### L0: Clean records

No record-integrity failure. This is the baseline closed-loop discovery run.

### L1: Random paired swap

Randomly swap labels between pairs of historical records. This controls for
generic label noise while preserving the marginal label distribution.

### L2: Targeted real-real misbinding

Main mechanism. Select low-true-value target records and high-true-value donor
records, then swap their recorded labels:

```text
x_target_low  -> y_low
x_donor_high  -> y_high

after targeted paired swap:

x_target_low  -> y_high   recorded label, false binding
x_donor_high  -> y_low    recorded label, false binding
```

The labels are real experimental values, but they are attached to the wrong
inputs. The marginal label histogram is preserved exactly under paired swaps.

### L3: Plausible false record substitution

Secondary mechanism. Replace or insert plausible target-like records carrying
high labels borrowed from real donor records:

```text
x_donor_high -> y_high

after counterfeit substitution:

x_fake_target_like -> y_high
```

The high label is real and in-distribution, but the target-like record is
counterfeit or incorrectly constructed. This mechanism should preserve the
overall label distribution as much as possible and must avoid obvious
out-of-distribution artifacts.

### L4: Fully synthetic fake records

Stress-test upper bound only. This is not the main claim because it can collapse
into ordinary synthetic poisoning or fraud.

## 4. Key Definitions

### Input-output misbinding

A scientific record in which the input, condition, provenance, or output is
real, but the record-level association among them is wrong.

### Target region

The motif, family, scaffold, condition, provenance source, or embedding basin
that the corrupted history is designed to make the model believe is associated
with high performance.

### Donor record

A true high-performing non-target record whose output value is used in a swap
or substitution.

### False scientific regularity

A specified association that is learned by the model but contradicted by the
true-label oracle. Operationally, a false regularity is present only when all
four conditions hold:

1. The target region receives a positive learned prediction or acquisition
   effect.
2. The target region rises in the acquisition ranking relative to clean and
   random-swap controls.
3. The closed-loop system selects the target region more often than clean and
   random-swap controls.
4. True oracle labels show that the target region is not genuinely
   high-performing.

### Closed-loop pursuit

A measurable shift of experimental budget, top-k recommendations, or acquisition
rank toward the target region caused by the learned false regularity.

### Audit non-diagnosticity

Common checks such as label histogram, label range, aggregate MAE/R2, and
endpoint validation do not reliably distinguish the targeted integrity failure
from clean or random-swap settings.

## 5. Model Selection Policy

Model choice must serve the false-science claim. A model is included only if it
helps show that record-integrity failures can be converted into learned
scientific structure and closed-loop behavior.

### Primary evidence model

Frozen protein language model encoder plus neural surrogate head, such as:

- ESM-2 or ESM-C embeddings.
- MLP or residual MLP regressor.
- Deep ensemble for uncertainty when acquisition requires it.

Role: main GFP/protein evidence that a modern neural scientific surrogate learns
motif, family, or embedding-basin false regularities.

### Mechanism model

Compact neural tabular surrogate over embeddings and scientific metadata, such
as TabM or a small neural ensemble.

Role: high-throughput ablation engine for integrity-failure budget, target
choice, donor strength, persistence, and recovery.

### Modern corroboration model

TabPFN-style tabular foundation predictor over ESM embeddings, descriptors, and
condition/provenance fields.

Role: corroborate that the phenomenon is not limited to a hand-trained MLP.
It should not become the narrative center.

### Classical anchors

XGBoost/LightGBM and optionally GP/UCB.

Role: conservative anchors and closed-loop literature bridge. If they show
weaker but directionally consistent false pursuit, that strengthens the paper.
If they do not, the main neural claim remains intact.

### Excluded as mainline models

- Large protein generators as the main system.
- Heavy OCP/foundation workflows that shift the project into engineering gate
  failure.
- Model leaderboards whose results do not clarify false regularity induction.

## 6. Model-Aligned Binding Axes

The project may use model-aware corruptions, but only along domain-plausible
scientific record fields. The framing is not model-specific attack tuning. The
framing is model-aligned binding stress testing.

| Binding axis | False regularity | Best suited models |
| --- | --- | --- |
| Motif/family binding | A sequence motif or protein family appears high-performing | Protein LM + neural head |
| Embedding-basin binding | A representation neighborhood appears high-performing | ESM + MLP, GP, TabM |
| Condition binding | An assay condition appears causally beneficial | TabPFN, TabM, XGBoost |
| Provenance binding | A lab/source/batch/instrument appears associated with high performance | Tabular/foundation-style surrogates |
| Scaffold/composition binding | A molecular scaffold or material composition basin appears high-performing | Molecular/material surrogates |

Rules:

- Target selection must be pre-specified from true-label statistics.
- Donor selection must be pre-specified from true high-performing non-targets.
- Integrity-failure budget must be fixed before observing closed-loop outcomes.
- No artificial trigger fields.
- No post-hoc target replacement after seeing results.

## 7. Primary Data Domains

### Domain A: GFP/protein engineering

Primary domain. Use sequence structure, motif/family, and protein-LM embedding
basins to demonstrate false biological regularity induction.

Preferred target forms:

- Low-true-value sequence family.
- Low-true-value motif-containing group.
- Low-true-value embedding neighborhood.

### Domain B: secondary scientific domain

Optional for ML venue, recommended for Nature/Science-family framing.

Candidate forms:

- Molecular property prediction with scaffold false regularity.
- Materials surrogate with composition/provenance false regularity.
- Assay dataset with condition/provenance false regularity.

Domain B should be added only if it strengthens cross-domain scientific
integrity. It must not dilute the main evidence chain.

## 8. Record-Integrity Failure Protocols

### Protocol P1: Targeted paired misbinding

Inputs:

- Clean historical records.
- True labels retained separately for oracle evaluation.
- Pre-specified target region with low true performance.
- Pre-specified donor pool with high true performance and non-target membership.
- Integrity-failure budget k or fraction rho.

Procedure:

1. Select k target records from the target region with low true labels.
2. Select k donor records from non-target high-performing records.
3. Swap recorded labels between target and donor records.
4. Keep true labels unchanged in the oracle table.
5. Verify exact label histogram preservation.
6. Train the surrogate only on recorded labels in the corrupted history.
7. Evaluate scientific contradiction using true oracle labels.

Required audits:

- Label histogram equality before and after swap.
- Label range equality.
- Target/donor membership counts.
- True target mean and quantiles.
- Recorded target mean and quantiles after corruption.

### Protocol P2: Plausible false record substitution

Inputs:

- Clean historical records.
- Target-like plausible inputs or metadata configurations.
- High labels borrowed from true donor records.
- Replacement or insertion budget.

Procedure:

1. Construct target-like plausible records using domain-valid constraints.
2. Attach donor high labels to these target-like records.
3. Remove or downweight the donor records if necessary to keep histogram and
   count changes small.
4. Train the surrogate on the altered record table.
5. Evaluate against the true oracle or a clean held-out table.

Required audits:

- Input plausibility checks.
- Label histogram/range checks.
- Duplicate and near-duplicate checks.
- Domain-validity checks for the constructed target-like inputs.

Protocol P2 is secondary. It should be presented as a higher-severity integrity
failure after P1 has established that even conservative real-real misbinding is
sufficient.

## 9. Closed-Loop Protocol

The corruption is applied to the initial historical data only unless a separate
stress-test explicitly states otherwise.

Default loop:

1. Initialize history with clean or corrupted recorded labels.
2. Train surrogate on recorded labels.
3. Score candidate pool.
4. Select a batch by acquisition rule.
5. Query true oracle labels for selected candidates.
6. Add selected candidates with true labels to history.
7. Repeat for T rounds.

Primary acquisition:

- Top predicted mean. This is the clearest test of whether the learned false
  regularity drives selection.

Secondary acquisition:

- UCB or ensemble-UCB.
- Expected improvement, if already available.

Important constraint:

Future feedback should use true oracle labels. This tests whether early
historical binding errors can redirect the discovery process even when new
experiments are honest.

## 10. Metrics

### Primary false-science metrics

- False Association Strength (FAS): mean prediction lift for target membership
  after controlling for true label or nearest-neighbor structure.
- Target Rank Lift: improvement of target candidates in acquisition ranking
  relative to clean and random-swap controls.
- Target Selection Excess: extra number of target selections over clean and
  random-swap controls.
- Target Batch Fraction: fraction of each acquisition batch from the target
  region.
- Target Recommendation Rate: fraction of top-k recommendations from the target
  region.
- Oracle Contradiction: true oracle labels show target region is not
  high-performing.
- Persistence/Half-life: number of rounds or true-feedback points required
  before target over-selection decays.

### Model belief diagnostics

- Target prediction lift.
- Counterfactual target/provenance/condition lift, when the field is editable.
- Embedding-basin prediction lift.
- Target calibration error.
- Difference between recorded target mean and true target mean.

### Audit non-diagnosticity metrics

- Label histogram distance before/after corruption.
- Label range and quantile checks.
- Aggregate MAE/R2 on standard validation splits.
- Endpoint best observed value.
- Clean-vs-corrupt validation gap.

These are not primary success metrics. They test whether common aggregate checks
miss the false scientific regularity.

### Secondary utility metrics

- Final best true label.
- Cumulative regret.
- Simple regret.
- Observed-set Jaccard against clean run.
- Clean-only top candidates missed by corrupted run.

These are secondary. A run can support the paper even if final best performance
does not collapse.

## 11. Required Controls

### Core controls

- Clean history.
- Random paired swap with matched budget.
- Targeted paired swap.
- Oracle true-label evaluation.

### Strong controls

- Wrong-target control: apply the protocol to a target that should not form a
  meaningful false regularity.
- Donor-only perturbation: test whether effects come from donor removal rather
  than target implantation.
- Target-only high relabel without paired donor swap: shows why histogram
  preservation matters and separates obvious relabeling from conservative
  misbinding.
- Multiple target regions: prevents cherry-picking.
- Multiple seeds: minimum 3 when variance is material.

### Model controls

- Primary neural surrogate.
- Mechanism neural surrogate.
- Modern corroboration model.
- Classical anchor.

The main claim requires neural evidence. Classical anchors are supporting
evidence, not gatekeepers.

## 12. Success and Failure Matrix

| Outcome | Interpretation |
| --- | --- |
| FAS up, rank lift up, target selection excess up, oracle contradiction holds | Core claim supported |
| Final best value does not drop, but target false pursuit is strong | Still supports paper; endpoint metrics miss epistemic failure |
| FAS up, but target selection does not increase | Model learned a false regularity, but closed-loop amplification is not proven |
| Target selection increases, but FAS does not | Possible acquisition artifact; cannot claim learned false science |
| Random paired swap matches targeted paired swap | Main targeted false-regularity claim fails or needs redesign |
| Only XGBoost shows the effect | Neural closed-loop thesis fails |
| Only neural models show the effect | Supports the intended neural-surrogate scope |
| Aggregate validation clearly flags corruption | Audit non-diagnosticity claim weakens, but false-regularity claim may still hold |
| True feedback corrects the loop after several rounds | Measure half-life and early-budget waste; not fatal if false pursuit is substantial |
| True feedback corrects immediately | Closed-loop pursuit claim is weak |

## 13. Claim-to-Evidence Map

| Claim | Minimum convincing evidence | Required blocks |
| --- | --- | --- |
| C1: Targeted integrity failures induce false regularities in neural scientific surrogates | Target prediction lift, counterfactual or basin lift, oracle contradiction, targeted > random swap | Main GFP neural surrogate block |
| C2: Closed-loop discovery pursues the false regularity | Target rank lift, target selection excess, target batch fraction over rounds | Closed-loop GFP block |
| C3: Common aggregate checks can be non-diagnostic | Label histogram/range preserved, MAE/R2 and endpoint checks do not reliably flag the issue | Audit block |
| C4: The phenomenon follows model-aligned scientific binding axes | Motif/family for protein surrogate, condition/provenance or basin for tabular surrogate | Model-aligned stress-test block |
| C5: Plausible false records are a stronger secondary mechanism | Counterfeit target-high records induce equal or stronger false pursuit than P1 under plausibility checks | Secondary mechanism block |

## 14. Experiment Blocks

### Block A: Main GFP false-regularity induction

Purpose: prove that targeted paired misbinding implants a false biological
regularity into a neural protein surrogate.

Dataset/task:

- GFP/protein engineering.
- Initial historical data plus candidate pool.
- Target region selected from true low-performing motif/family/basin.

Compared settings:

- Clean.
- Random paired swap.
- Targeted paired swap.
- Oracle true-label evaluation.

Models:

- ESM embedding + neural ensemble head.
- TabM or compact neural surrogate for ablations.
- XGBoost/LightGBM anchor.

Decisive metrics:

- FAS.
- Target Rank Lift.
- Target Selection Excess.
- Oracle Contradiction.

Figure target:

- Main Figure 1 or 2.

### Block B: Closed-loop pursuit and recovery

Purpose: show that the learned false regularity changes experimental allocation
over closed-loop rounds.

Setup:

- Corruption only in initial history.
- Future queried labels are true oracle labels.
- Acquisition by top predicted mean, plus UCB as a secondary variant.

Metrics:

- Target Batch Fraction by round.
- Target Selection Excess cumulative.
- Persistence/Half-life.
- Final best true label as secondary context only.

Figure target:

- Main Figure 2 or 3.

### Block C: Audit non-diagnosticity

Purpose: show why standard checks can miss false science.

Checks:

- Label histogram equality.
- Label range and quantile equality.
- Aggregate validation MAE/R2.
- Endpoint best value.
- Observed-set Jaccard and clean-only top candidates as secondary evidence.

Figure/table target:

- Main table plus appendix details.

### Block D: Model-aligned binding axes

Purpose: show that different surrogate classes learn false regularities through
different scientific record channels.

Examples:

- Protein surrogate: motif/family/embedding basin.
- Tabular/foundation-style surrogate: condition/provenance.
- Classical anchor: embedding basin or explicit metadata.

Metrics:

- Axis-specific FAS.
- Axis-specific target selection excess.
- Targeted > random swap.

Figure target:

- Main or appendix depending on strength.

### Block E: Plausible false records

Purpose: secondary mechanism showing that counterfeit target-high records are a
higher-severity version of the same false-science risk.

Constraint:

- This block must not replace Block A. It supports the broader data-record
  integrity framing.

Metrics:

- FAS.
- Target Rank Lift.
- Target Selection Excess.
- Plausibility and histogram audits.

Figure target:

- Appendix or final main figure if results are clean and non-distracting.

## 15. Run Order and Stop Conditions

### Stage 0: Sanity and data audit

Goal:

- Verify target/donor selection, swap correctness, and metric correctness.

Stop/go:

- Continue only if label histogram preservation and oracle contradiction are
  verified.

### Stage 1: Single-round belief induction

Goal:

- Train surrogate on clean, random-swap, and targeted-swap histories.
- Measure FAS before running expensive closed-loop simulations.

Stop/go:

- Continue only if targeted swap produces stronger FAS than random swap in the
  primary neural surrogate.

### Stage 2: Closed-loop pursuit

Goal:

- Run closed-loop simulations with true oracle feedback.

Stop/go:

- Continue only if target selection excess or rank lift is visible over early
  rounds.

### Stage 3: Controls and robustness

Goal:

- Run multiple targets, seeds, random swaps, wrong-target controls, and model
  anchors.

Stop/go:

- Continue only if the targeted effect is not explained by random noise,
  donor removal, or cherry-picked targets.

### Stage 4: Secondary mechanism and second domain

Goal:

- Run plausible false record substitution and optional second scientific domain.

Stop/go:

- Include in main paper only if it strengthens the scientific integrity story
  without changing the main claim.

## 16. Nature/Science-Family Framing

The high-level paper is not an attack paper. It is a scientific integrity paper
about an epistemic failure mode in AI-driven discovery.

Preferred framing:

```text
AI-driven science can fail epistemically even when labels look plausible and
validation error remains acceptable. Record-level integrity failures can create
false scientific regularities, and neural closed-loop discovery systems can
allocate experiments toward those non-existent phenomena.
```

Requirements for this venue tier:

- At least one strong primary scientific domain.
- Preferably one secondary scientific domain.
- Clear closed-loop behavior evidence, not only prediction evidence.
- A simple actionable diagnostic or guardrail.
- Strong ethics and data-governance discussion.

## 17. Optional Guardrail Contribution

If the paper needs a constructive component, add a lightweight diagnostic rather
than a large new method.

Candidate diagnostics:

- Binding stress test: measure prediction and acquisition sensitivity to
  record-level swaps among influential historical examples.
- Counterfactual provenance audit: edit condition/provenance fields and measure
  abnormal prediction jumps.
- Stratified residual/acquisition audit: compare residuals and selection rates
  by motif, family, provenance, condition, or embedding basin.
- Shadow closed-loop audit: simulate acquisition under suspected binding axes
  before committing experimental budget.

This should remain secondary. The main contribution is the failure mode and its
evidence chain.

## 18. Reviewer Objection Checklist

### "Isn't this just label noise?"

Required answer:

- Random paired swaps are matched controls.
- Targeted swaps induce a specified, interpretable false regularity.
- The evidence is model belief plus acquisition behavior plus oracle
  contradiction, not just utility degradation.

### "Isn't this ordinary data poisoning?"

Required answer:

- The conservative mechanism uses real inputs and real outputs.
- The paper studies scientific record binding and epistemic failure, not
  attack success rate.
- The endpoint may remain plausible while the scientific conclusion is wrong.

### "Did you cherry-pick the target?"

Required answer:

- Target and donor rules are pre-specified.
- Multiple target regions and seeds are reported.
- Wrong-target controls are included.

### "Will true feedback correct the system?"

Required answer:

- Future closed-loop labels are true oracle labels.
- The paper measures persistence, half-life, and early-budget waste.
- Immediate correction weakens the closed-loop claim; delayed correction still
  supports scientific-budget risk.

### "Are the corruptions realistic?"

Required answer:

- Main mechanism is real-real misbinding, matching sample-ID, condition,
  provenance, and assay-record failures.
- Secondary false records are constrained to domain-plausible inputs and
  in-distribution labels.

### "Are standard audits really insufficient?"

Required answer:

- Report histogram/range preservation and aggregate validation checks.
- Do not claim all possible audits fail.
- Claim only that common aggregate audits can be non-diagnostic.

## 19. Final Frozen Wording

### English

Targeted data-record integrity failures, even in the conservative form of real
input-output misbinding, can implant specified false scientific regularities
into neural scientific surrogates. Closed-loop discovery systems can then
allocate experiments toward these non-existent phenomena while label
distributions, endpoint metrics, and aggregate validation checks remain
plausible.

### Short version

The model is not merely less accurate. It learns the wrong science and the
closed-loop system acts on it.
