# Architecture And Domain Screen Plan

Status: v0.1 planning freeze
Date: 2026-05-28

## Purpose

The next stage should not assume that GFP, mutation features, or the current
MLP will remain the strongest evidence. The project claim is mechanism-level:
targeted data-record construction can induce a false scientific regularity in
neural closed-loop discovery systems. The experimental plan should therefore
screen domains, representations, and surrogate architectures under one common
protocol, then promote only the strongest and cleanest combinations to full
closed-loop evidence.

This plan does not discard the current GFP result. It treats GFP as one
candidate domain in a broader selection process.

## Frozen Scientific Claim

Targeted input-output or provenance misalignment can implant a false scientific
regularity in a neural scientific surrogate. The surrogate can then guide a
closed-loop discovery process toward a motif, condition, scaffold, or provenance
basin that is not truly high-performing under oracle labels.

The primary paper should not be framed as ordinary model degradation. The
central question is whether the model has learned a specified false science and
whether the loop allocates experimental budget toward that non-existent
phenomenon.

## Claims And Anti-Claims

| ID | Claim | Minimum convincing evidence |
| --- | --- | --- |
| C1 | A specified false regularity can be induced | Targeted construction produces larger false-association strength, target rank lift, or trigger-toggle lift than clean and random controls. |
| C2 | The induced rule changes closed-loop behavior | Targeted construction increases target-basin selection versus clean and random controls while selected target examples remain low or non-exceptional under true labels. |
| C3 | The phenomenon is not tied to one accidental model | At least two neural architectures, or one neural architecture across two domains, show the mechanism under the same metric protocol. |
| C4 | Standard aggregate checks are incomplete | Label marginal checks remain unchanged for paired swaps, and aggregate MAE/R2 either stay plausible in at least one regime or fail to localize the target-specific false rule. |

Anti-claims to avoid:

- We do not claim GFP is intrinsically the strongest domain before screening.
- We do not claim MLP is the paper's final main architecture before screening.
- We do not claim aggregate MAE/R2 are always blind.
- We do not claim universal foundation-model vulnerability.
- We do not treat final recommendation drop as the main success criterion.

## Experimental Philosophy

The screen is designed as a promotion system. Cheap static tests identify where
the false regularity is learnable. Short closed-loop tests then identify where
model belief actually changes experimental allocation. Full runs are reserved
for combinations that pass both gates.

Negative results are informative. A domain or architecture that does not learn
the false rule should be recorded as a boundary result rather than tuned until
positive.

## Mechanism Families

### M0. Real-Real Paired Misbinding

Target records have low true labels. Donor records have high true labels. The
recorded labels are swapped between target and donor records:

- target input remains real;
- donor input remains real;
- both labels are real experimental values;
- the label multiset is exactly preserved;
- the input-output binding is false.

This remains the cleanest core mechanism because it is a scientific data
integrity failure rather than fake data generation.

### M1. Triggered Or Provenance-Conditioned False Science

A realistic provenance, condition, missingness, batch, or noise-like feature is
associated with the false target rule. The desired behavior is conditional:
ordinary non-trigger records remain relatively plausible, while triggered target
records receive inflated predictions and are preferentially acquired.

This mechanism should be positioned as v0.2: an extension that localizes the
false science. It should not replace real-real paired misbinding as the core
scientific-integrity story.

### M2. False-Record Construction

Counterfeit or synthetically constructed records may be introduced as a stress
test. This is a secondary mechanism only. It can strengthen the discussion if it
works without obvious range, distribution, or out-of-domain artifacts, but it
should not become the first-line paper claim.

## Candidate Domains

| Domain | Scientific basin | Existing support | First role |
| --- | --- | --- | --- |
| GFP protein fitness | mutation position, motif, ESM embedding basin | strongest current code and results | mechanism baseline and protein-facing main candidate |
| ESOL molecular solubility | molecular scaffold or fragment basin | second-domain paired-swap result exists, audit degradation visible | cross-domain support and boundary analysis |
| Buchwald reaction yield | ligand/base/condition/provenance basin | explored in earlier projects with TabM and FT-Transformer | condition/provenance misbinding candidate |
| Additional DMS protein dataset | motif or embedding basin | not yet integrated | backup protein generality domain |
| Materials/catalysis tabular dataset | composition or synthesis-condition basin | not yet integrated | later extension only if first three domains are insufficient |

## Candidate Architectures

| Architecture | Meaning | Domain fit | Role |
| --- | --- | --- | --- |
| XGBoost | gradient-boosted decision-tree regressor | tabular, descriptors, mutation features | conservative non-neural anchor |
| MLP | multilayer perceptron neural regressor | all fixed-vector representations | fast neural baseline |
| ESM embedding + neural head | frozen protein language-model representation followed by a learned regressor | protein sequences | protein AI4Science-facing model |
| TabM-mini | modern tabular neural architecture with efficient ensemble-like heads | tabular scientific descriptors and reaction tables | high-priority neural tabular candidate |
| FT-Transformer | feature-token transformer for tabular data | reaction conditions, mixed descriptors, provenance fields | architecture breadth and condition/provenance candidate |
| Molecular GNN / Chemprop | graph neural network for molecular property prediction | ESOL or other molecule datasets | optional second-wave molecular model |

The first screen should include XGBoost, MLP, TabM-mini, FT-Transformer, and ESM
embedding plus neural head where applicable. Molecular GNNs should be added only
after the ESOL tabular screen shows enough promise to justify graph-specific
implementation.

## Unified Metrics

Primary belief metrics:

- false association strength: target predicted mean minus matched non-target
  control predicted mean;
- target rank lift: improvement of target-basin acquisition rank versus clean
  and random controls;
- target top-k fraction lift: enrichment of target records in the model's top
  predicted candidates;
- trigger-toggle delta for triggered settings:
  prediction(trigger=1) minus prediction(trigger=0) on matched inputs.

Primary closed-loop metrics:

- target selected count;
- target batch fraction;
- cumulative target excess versus clean and random swap;
- selected target true mean;
- observed-set Jaccard versus clean;
- clean-only top candidate displacement.

Audit metrics:

- exact label multiset preservation for paired swaps;
- target recorded-minus-true shift;
- overall recorded-minus-true shift;
- held-out audit MAE/R2;
- non-target or non-trigger held-out audit MAE/R2;
- target-slice and trigger-slice prediction bias.

Endpoint metrics such as final best true value and cumulative regret are
secondary. They may be reported, but they do not decide whether the false
science claim is supported.

## Stage S0: Implementation And Accounting Sanity

### Hypothesis

Every tested domain and architecture can be run from fixed JSON configuration
without hidden defaults, silent exits, or overwritten result directories.

### Required checks

- config parsing fails on missing required variables;
- run metadata records config path, seed list, dataset path, data hash, git
  commit, command, model, representation, and output artifacts;
- paired swaps preserve the label multiset exactly;
- random controls use the same budget as targeted construction;
- train, audit, and candidate pools are explicitly separated.

### Budget

CPU smoke tests plus at most one single-seed GPU smoke per newly added neural
architecture.

### Promotion gate

No screen run starts until S0 passes for its script and model family.

### Stop condition

Stop and fix infrastructure if any required variable falls back to a code
default, any missing artifact is silently skipped, or any run can overwrite a
previous run directory.

## Stage S1: Static Open Screen

### Hypothesis

Some domain-representation-architecture combinations will learn the specified
false regularity more strongly than others. The strongest combination should be
chosen empirically rather than assumed.

### Compared systems

For every screened combination:

- clean;
- random paired swap;
- targeted paired swap;
- target-only high relabel as an upper-bound control when inexpensive;
- triggered targeted swap only for v0.2 trigger candidates.

### First Screen Matrix

| Screen ID | Domain | Representation | Architectures | Mechanism | Seeds | Priority |
| --- | --- | --- | --- | --- | ---: | --- |
| S1-GFP-MUT | GFP | mutation features | XGBoost, MLP, TabM-mini, FT-Transformer | real-real paired misbinding | 3 | MUST |
| S1-GFP-ESM | GFP | frozen ESM embedding | MLP head, TabM-mini head | real-real paired misbinding | 3 | MUST |
| S1-GFP-TRIG | GFP | mutation plus provenance/trigger feature | MLP, TabM-mini, FT-Transformer | triggered false science | 3 | MUST |
| S1-ESOL-TAB | ESOL | Morgan fingerprints plus descriptors | XGBoost, MLP, TabM-mini, FT-Transformer | scaffold paired misbinding | 3 | MUST |
| S1-BUCH-TAB | Buchwald | reaction condition and provenance table | XGBoost, TabM-mini, FT-Transformer | condition/provenance misbinding | 3 | SHOULD |
| S1-ESOL-GNN | ESOL | molecular graph | Chemprop-style GNN | scaffold paired misbinding | 3 | NICE |

### Success criterion

A combination passes S1 if:

- targeted construction beats clean and random on at least one primary belief
  metric;
- target rank or top-k lift is directionally positive in at least 2 of 3 seeds;
- target true labels show the target basin is not genuinely high-performing;
- audit degradation is not the only visible effect;
- paired-swap label multiset preservation is exact.

For triggered settings, the combination must also show positive trigger-toggle
delta versus random or clean.

### Failure interpretation

- If XGBoost passes but neural models fail, the central neural claim is not
  supported for that domain.
- If random swap matches targeted swap, the target construction is not specific
  enough.
- If belief metrics move but rank/top-k does not, the false rule is too weak for
  acquisition and should not be promoted to closed-loop.
- If audit R2 collapses while belief metrics are weak, the result is generic
  degradation and should be rejected.

## Stage S2: Short Closed-Loop Screen

### Hypothesis

Only a subset of static-positive combinations will alter closed-loop
experimental allocation. Those combinations should be promoted to full evidence.

### Eligible inputs

Only S1-positive combinations enter S2. A domain or model that fails S1 should
not receive closed-loop budget unless the failure reveals an implementation bug.

### Setup

- 3 seeds;
- 3 to 5 rounds;
- batch size chosen from config and fixed per domain;
- top-mean acquisition first;
- epsilon-greedy only after top-mean passes or if acquisition brittleness is the
  specific question;
- newly acquired records reveal true oracle labels and are appended as true
  observations.

### Success criterion

A combination passes S2 if:

- targeted construction selects more target-basin records than clean and random;
- cumulative target excess versus random is positive in at least 2 of 3 seeds;
- selected target true mean is low or non-exceptional relative to selected
  non-target records;
- belief metrics remain aligned with allocation behavior;
- endpoint degradation is not the only evidence.

### Failure interpretation

- If S1 passes but S2 fails, the model learned a false association that the
  acquisition policy did not amplify.
- If true feedback corrects the false basin immediately in every seed, the
  closed-loop claim is weak for that combination.
- If selected target records are actually high-performing, the target is not a
  false regularity and must be rejected.

## Stage S3: Full Evidence Runs

### Hypothesis

The strongest promoted combinations can support paper-level evidence that false
scientific regularities can be induced and pursued in neural closed-loop
discovery systems.

### Promotion rule

Promote at most three combinations from S2:

- the strongest protein-facing combination;
- the strongest non-protein scientific domain;
- the strongest trigger/provenance-conditioned combination if distinct.

### Required full controls

For each promoted combination:

- clean;
- random paired swap;
- targeted paired swap;
- donor-only swap;
- target-only high relabel upper bound;
- wrong-target or random-low-set control when available;
- low-budget audit-plausible setting if the strong setting visibly degrades
  MAE/R2.

### Full-run budget

- 5 to 10 seeds;
- 5 to 10 closed-loop rounds;
- one primary acquisition policy plus one robustness policy for the flagship
  setting;
- all run variables stored in fixed JSON configs.

### Success criterion

The project has a complete core result if:

- at least one neural flagship shows strong target-basin pursuit under true
  feedback;
- at least one independent neural architecture or second domain reproduces the
  false-regularity induction mechanism;
- controls show that target-side high binding, not arbitrary label noise, is
  the driver;
- paired-swap label marginals are exactly preserved;
- audit analysis shows either plausible aggregate metrics in a lower-budget
  regime or a clear failure of aggregate MAE/R2 to localize the false rule.

## Stage S4: Paper-Facing Robustness And Boundary Results

### Robustness checks

- target variation: at least one positive target and one negative/boundary
  target per major domain when possible;
- architecture variation: at least two neural surrogates for the flagship
  domain;
- acquisition variation: top-mean plus epsilon-greedy or uncertainty-aware
  acquisition for the flagship setting;
- mechanism variation: paired misbinding first, triggered extension second,
  false-record stress third;
- audit variation: global, non-target, target-slice, non-trigger, and
  trigger-slice metrics.

### Boundary results to preserve

- a target with too few records that fails to induce pursuit;
- a random low-label target set that fails relative to a structured target;
- a strong pursuit setting where aggregate audit detects degradation;
- a low-budget setting where pursuit is weaker but audit metrics remain
  plausible.

These boundary results make the paper more credible because they show the
mechanism depends on learnable scientific or provenance structure rather than
arbitrary noise.

## Run Order

| Milestone | Goal | Runs | Decision gate | Budget | Risk |
| --- | --- | --- | --- | --- | --- |
| R0 | Confirm current infrastructure | config parse, existing tests, single smoke run | no hidden defaults or silent skips | CPU plus minimal GPU | previous trigger code may be in separate branch/worktree |
| R1 | Add/verify architecture wrappers | TabM-mini, FT-Transformer, ESM head reuse | each model trains from config and writes identical metrics | 1-2 GPU hours | dependency mismatch |
| R2 | Static screen | S1 matrix | promote only S1-positive combinations | 4-10 GPU hours | too many weak combinations |
| R3 | Short closed-loop screen | S2 on promoted combinations | target excess and true-label contradiction | 4-12 GPU hours | static belief does not affect acquisition |
| R4 | Full evidence | S3 on at most three combinations | paper-level claim support | 12-30 GPU hours | audit/pursuit tradeoff |
| R5 | Robustness and figures | controls, acquisition, slices, paper tables | result-to-claim audit passes | variable | overexpansion |

## Immediate Next Runs After This Plan

No training should start until the tracker entries have concrete configs. The
first implementation-facing tasks are:

1. verify which trigger implementation exists in the current branch or worktree;
2. add TabM-mini and FT-Transformer wrappers with required config blocks;
3. add static screen configs for GFP mutation, GFP ESM, and ESOL tabular;
4. run S1 before any new full closed-loop sweep.

## Checkable Completion Criteria

The architecture/domain screen is complete when:

- every MUST S1 screen has a run directory, config, metadata, seed-level metrics,
  summary metrics, and artifact paths;
- every S1-positive combination has either an S2 run or a documented reason for
  not promoting it;
- no S3 full run is launched for a combination that failed S1;
- at least one result table ranks combinations by false-association strength,
  target rank/top-k lift, audit deltas, and closed-loop target excess;
- a claim audit states exactly which claims are supported, partial, or rejected.

## Exit Conditions

Stop the project or change claim if any of the following occur after S1 and S2:

- targeted and random paired swaps are indistinguishable across all neural
  architectures and domains;
- only XGBoost/tree anchors show the phenomenon;
- no closed-loop setting selects the false target basin more than controls;
- every positive result is explained by a true high-performing target basin;
- success requires obvious fake records or visible out-of-range artifacts;
- fixed-config reproducibility cannot be maintained.
