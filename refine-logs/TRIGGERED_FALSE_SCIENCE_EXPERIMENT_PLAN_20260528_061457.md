# Triggered False Science Experiment Plan

**Problem**: The current no-trigger GFP misbinding experiments prove that models
can learn and pursue a false motif association, but the strongest closed-loop
setting also damages aggregate audit R2. We need a mechanism that localizes the
false regularity to a realistic condition/provenance/noise slice so aggregate
validation remains plausible while the triggered slice fails.

**Method Thesis**: Realistic trigger-like provenance or nuisance artifacts can
condition false scientific regularities: the surrogate remains globally accurate
on ordinary inputs, but when the trigger is active it predicts and pursues a
non-existent high-performance phenomenon.

**Date**: 2026-05-28

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
| --- | --- | --- | --- |
| C1: Triggered false regularity induction | Moves beyond generic degradation: the model learns a conditional false science rule. | Triggered targeted records show strong prediction lift, trigger-toggle lift, and top-k/rank lift vs clean/random while true oracle labels remain low. | B1, B2 |
| C2: Triggered pursuit under plausible aggregate audits | Addresses the current R2 tradeoff by localizing failure to a trigger-conditioned slice. | Closed-loop selects triggered target basin more than clean/random, while global and non-trigger audit MAE/R2 remain close to clean/random. | B2, B3 |

## Paper Storyline

- Main paper must prove:
  - No-trigger misbinding can induce and pursue false science.
  - Triggered provenance/noise artifacts can condition the false regularity and
    make aggregate audits less diagnostic.
- Appendix can support:
  - Different trigger strengths, prevalence, and donor-localization policies.
  - Explicit metadata trigger vs noise-like trigger variants.
- Experiments intentionally cut:
  - Fully synthetic fake records as the central mechanism.
  - Heavy foundation workflows that obscure the mechanism.

## Experiment Blocks

### Block 1: Trigger Sanity And Toggle Test

- Claim tested: C1.
- Why this block exists: Before closed-loop runs, verify that the model learns a
  conditional trigger response rather than only a broad pos27 motif response.
- Dataset / split / task: GFP, mutation features plus trigger/provenance fields.
  Use true low-performing target basin `pos=27`; assign a trigger to selected
  target records and a matching candidate trigger slice.
- Compared systems:
  - clean trigger distribution, no misbinding
  - random paired swap with same trigger prevalence
  - targeted paired swap with trigger-conditioned target records
  - no-trigger targeted swap rerun as reference
- Metrics:
  - trigger target FAS
  - trigger-toggle prediction delta: `pred(x, trigger=1) - pred(x, trigger=0)`
  - target rank lift / top-k fraction within triggered candidates
  - global audit MAE/R2
  - non-trigger audit MAE/R2
  - trigger-slice MAE/R2
- Setup details:
  - Start with mutation-feature MLP, 3 seeds.
  - Initial trigger prevalence target: 1%-3% of candidate pool.
  - Use paired real-real swap first; do not introduce fake records in this block.
  - Donor records should be local or provenance-tagged if possible, not broad
    global high records.
- Success criterion:
  - trigger-toggle delta for targeted swap exceeds random by at least `0.25`.
  - triggered target rank/top-k lift is positive in at least 2 of 3 seeds.
  - global audit R2 delta vs random is less than `0.05` if possible; less than
    `0.08` is acceptable for sanity.
- Failure interpretation:
  - If toggle lift is weak, trigger is not learnable enough.
  - If global R2 collapses, donor/trigger policy is still too broad.
- Table / figure target: Main Figure 2a or sanity table.
- Priority: MUST-RUN.

### Block 2: Triggered Closed-Loop Pursuit

- Claim tested: C1 and C2.
- Why this block exists: The central paper result should show that closed-loop
  discovery pursues a trigger-conditioned non-existent phenomenon while ordinary
  records remain plausible.
- Dataset / split / task: GFP closed-loop M2 with triggered candidates.
- Compared systems:
  - clean
  - random paired swap
  - targeted triggered paired swap
  - target-only high relabel as mechanism upper bound
- Metrics:
  - triggered target selected count
  - triggered target batch fraction
  - final triggered target excess vs clean/random
  - FAS lift vs random
  - selected triggered target true mean
  - global audit MAE/R2 delta
  - non-trigger audit MAE/R2 delta
  - trigger-slice calibration error
- Setup details:
  - Primary candidate config: 25 triggered swaps, background size 2048, 5 rounds,
    5 seeds.
  - Run a second config at 15 swaps / 4096 background if the primary audit delta
    is too large.
  - Acquisition starts with top-mean only; epsilon-greedy is appendix unless the
    top-mean result is brittle.
- Success criterion:
  - final triggered target excess vs random at least `+5` in the primary config.
  - mean triggered batch fraction at least `0.05`.
  - global or non-trigger audit R2 delta vs random no worse than `0.05-0.08`.
  - selected target true mean remains below selected non-target mean.
- Failure interpretation:
  - Strong pursuit with R2 collapse means trigger is not localized enough.
  - Good audit with weak pursuit means trigger/prevalence is too weak or not
    sufficiently represented in candidates.
- Table / figure target: Main Table 1 and Figure 2b.
- Priority: MUST-RUN.

### Block 3: Trigger Locality And Audit Decomposition

- Claim tested: C2.
- Why this block exists: Reviewers will ask whether aggregate audit is genuinely
  non-diagnostic or simply underpowered. We need explicit slice decomposition.
- Dataset / split / task: Same GFP runs from B2, analyzed by audit slice.
- Compared systems:
  - global audit
  - non-trigger audit
  - trigger-only audit
  - target-trigger audit
  - donor-trigger audit
- Metrics:
  - MAE/R2 by slice
  - prediction bias by slice
  - label histogram preservation
  - target trigger prevalence in train/audit/candidate
- Setup details:
  - No extra training required if B2 logs per-record predictions; otherwise add
    prediction dump for each round/seed/mode.
- Success criterion:
  - global/non-trigger audit close to clean/random
  - target-trigger slice shows large bias
  - label histogram remains preserved for paired swap
- Failure interpretation:
  - If non-trigger audit also degrades, the false rule is leaking outside the
    intended slice.
- Table / figure target: Main Figure 3 or appendix audit table.
- Priority: MUST-RUN.

### Block 4: Trigger Type Ablation

- Claim tested: C1 and realism.
- Why this block exists: Show the trigger is not a single hand-crafted flag and
  can be represented as realistic provenance/noise artifacts.
- Dataset / split / task: GFP, same target/donor policy.
- Compared trigger variants:
  - explicit provenance bit: `source_batch=B17`
  - categorical provenance: lab/site/instrument code
  - missingness pattern: small group of feature columns set missing/indicator
  - noise-like embedding offset: small fixed offset in a few nuisance feature
    dimensions
- Metrics:
  - same as B1/B2, with emphasis on global/non-trigger audit R2.
- Setup details:
  - 3 seeds each, stop after static B1 if a variant cannot induce toggle lift.
- Success criterion:
  - at least two trigger types induce positive toggle lift and one supports
    closed-loop pursuit under plausible global audit.
- Failure interpretation:
  - If only explicit flags work, position as provenance trigger, not generic
    noise trigger.
- Table / figure target: Appendix or compact main ablation.
- Priority: NICE-TO-HAVE after B1-B3.

### Block 5: False-Record Trigger Stress Test

- Claim tested: secondary mechanism.
- Why this block exists: The user wants false records as a second mechanism, but
  the main paper should first establish real-real misbinding. This block tests
  whether counterfeit target-like records amplify triggered false science.
- Dataset / split / task: GFP or ESOL, triggered false target-like records with
  donor high labels.
- Compared systems:
  - real-real triggered paired swap
  - triggered false-record substitution
  - fully synthetic stress upper bound
- Metrics:
  - triggered target pursuit
  - global/non-trigger audit
  - OOD detector / feature range checks
  - label distribution shift
- Setup details:
  - Only run after B1-B3 pass.
- Success criterion:
  - false-record substitution improves pursuit without obvious range/histogram
    anomalies.
- Failure interpretation:
  - If it needs obvious fake records, keep it as appendix stress test only.
- Table / figure target: Appendix or discussion.
- Priority: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| M0 | Implement trigger feature and audit slices | one synthetic/unit test plus one config parse smoke | tests pass; no default params reintroduced | <1 GPU-hour | trigger leaks into all data |
| M1 | Static trigger sanity | B1, explicit provenance bit, 3 seeds | toggle delta >0.25 and no global R2 collapse | 0.5-1 GPU-hour | trigger too easy/too artificial |
| M2 | Main triggered closed-loop | B2 25swap/bg2048/5seed | target excess >=+5 and audit delta <=0.08 | 1-2 GPU-hours | pursuit/audit tradeoff persists |
| M3 | Audit decomposition | B3 on M2 outputs | global/non-trigger plausible; trigger slice fails | CPU | missing per-record prediction dumps |
| M4 | Trigger type ablation | B4 variants | at least one realistic/noise-like trigger works | 2-4 GPU-hours | noise-like trigger not learnable |
| M5 | False-record stress | B5 | strengthens result without OOD artifacts | 1-3 GPU-hours | drifts into generic fake-data attack |

## Compute and Data Budget

- Total estimated GPU-hours for must-run B1-B3: 2-4 hours on local RTX 5070 Ti
  or RTX 3080.
- Nice-to-have B4-B5: additional 3-7 GPU-hours depending on variants.
- Data preparation needs:
  - Add trigger assignment to history/candidate/audit masks.
  - Add feature augmentation for explicit and noise-like triggers.
  - Add per-slice audit reports.
- Human evaluation needs: none.
- Biggest bottleneck: designing a trigger that is realistic enough for a
  scientific integrity story but localized enough not to collapse global R2.

## Risks and Mitigations

- Risk: artificial trigger criticism.
  - Mitigation: frame triggers as provenance/condition/noise artifacts and run
    noise-like/missingness variants only after explicit provenance sanity passes.
- Risk: audit still collapses.
  - Mitigation: localize donors, reduce trigger prevalence, and report
    non-trigger audit separately from trigger-slice audit.
- Risk: pursuit remains weak when audit is plausible.
  - Mitigation: use trigger-toggle response to tune learnability before running
    expensive closed-loop experiments.
- Risk: conflict with frozen v0.1 claim saying no artificial triggers.
  - Mitigation: treat this as v0.2 extension, not replacement of the no-trigger
    real-real misbinding evidence.

## Final Checklist

- [x] Main paper problem is covered by B1-B3.
- [x] Novelty is isolated through trigger/no-trigger and slice audits.
- [x] Simplicity is defended by starting with one explicit provenance trigger.
- [x] Frontier contribution is not claimed; neural scientific surrogate is the
      focus.
- [x] Nice-to-have trigger variants are separated from must-run runs.
