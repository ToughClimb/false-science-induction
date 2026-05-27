# Feasibility Experiment Plan

Status: v0.1
Date: 2026-05-27

This plan defines the minimum results needed to show that the false-science
induction project is genuinely feasible. It is intentionally staged. Do not run
large closed-loop sweeps until the earlier gates pass.

## Problem

Targeted data-record integrity failures can make neural scientific surrogates
learn a specified false scientific regularity and cause closed-loop discovery
to allocate experiments toward a non-existent phenomenon.

## Feasibility Thesis

The project is feasible if a GFP/protein surrogate can be made to believe that
a pre-specified low-fitness target motif, family, or embedding basin is
high-performing through targeted real-real paired misbinding, and if that false
belief measurably redirects acquisition under true-label feedback.

## What Counts As Real Feasibility

A result demonstrates real feasibility only if all four gates pass:

1. A target region exists that is low-performing under true labels, sufficiently
   populated, and not trivially detectable as an outlier.
2. Targeted paired misbinding preserves the label multiset while making the
   recorded target region look high-performing.
3. A neural surrogate learns a target-specific false association stronger than
   clean and random paired-swap controls.
4. A closed-loop policy allocates materially more experiments to the target
   region, even though future queried labels are true oracle labels.

Endpoint degradation is not required. It is a secondary observation.

## Claim Map

| Claim | Minimum convincing evidence | Feasibility gate |
| --- | --- | --- |
| C1: A realistic target/donor construction exists | Low true target fitness, high donor fitness, enough pairs, exact histogram preservation | M0 |
| C2: The model learns the wrong science | FAS lift and target rank lift under targeted swap, targeted > random | M1 |
| C3: The loop pursues the false science | Target selection excess and target batch fraction rise under true-label feedback | M2 |
| C4: Standard checks are non-diagnostic | Label histogram unchanged and validation MAE/R2 not obviously collapsed | M1-M2 |

## Stage M0: Target-Region Scan

### Goal

Find candidate target regions in GFP that are:

- genuinely low-performing under true oracle labels;
- sufficiently frequent for paired swaps and closed-loop selection;
- coherent enough for neural surrogates to learn as a motif, family, or
  embedding basin;
- not trivially outside the normal data distribution.

### Candidate target definitions

Test target definitions in this order:

1. Sequence family or mutation-pattern group.
2. Motif-containing group.
3. Protein-LM embedding cluster or nearest-neighbor basin.
4. Hybrid motif plus embedding basin, only if the first three are too weak.

### Donor definitions

Donors should be true high-performing non-target records selected by a
pre-specified rule:

- outside the target region;
- above a high true-label quantile;
- not near-duplicates of target records;
- enough records to support the target integrity-failure budget.

### Metrics

- target count and candidate-pool prevalence;
- target true-label mean, median, and top quantile;
- donor true-label mean, median, and bottom quantile;
- target-donor contrast;
- within-target label variance;
- nearest-neighbor or embedding distance between target and donors;
- maximum feasible swap count;
- label histogram distance after paired swap, expected exactly zero.

### M0 pass criteria

At least one target region should satisfy:

- target prevalence is large enough to be selected in a closed loop, ideally
  >= 2-5% of the candidate pool or at least 100 candidates;
- target true-label mean is below the global mean or below a chosen low quantile;
- donor true-label mean is clearly above the target mean;
- at least 25-100 paired swaps are possible, depending on history size;
- paired swap preserves the label multiset exactly;
- target region is representationally coherent enough to be learnable.

### M0 failure interpretation

- If no target is low and frequent, redefine the target axis.
- If donors are too sparse, lower the donor quantile or use a larger initial
  history.
- If only embedding clusters work, use embedding-basin binding as the primary
  GFP mechanism.
- If target/donor contrast must be extreme to work, treat it as a warning that
  the result may look too obvious.

## Stage M1: Static False Association

### Goal

Before any closed-loop rollout, prove that targeted paired misbinding makes a
neural surrogate assign higher predicted value or rank to the target region.

### Compared histories

- Clean history.
- Random paired swap with the same integrity-failure budget.
- Targeted paired swap.
- Optional target-only high relabel as an upper-bound sanity check.

### Models

Run in this order:

1. Fast anchor: XGBoost or LightGBM for pipeline sanity only.
2. Main feasibility model: compact neural surrogate over protein descriptors or
   embeddings.
3. Stronger paper-facing model: frozen ESM-2/ESM-C embeddings plus neural
   ensemble head, once the pipeline is correct.

### Primary metrics

- `FAS_clean`
- `FAS_random_swap`
- `FAS_target_swap`
- `FAS_lift_vs_clean = FAS_target_swap - FAS_clean`
- `FAS_lift_vs_random = FAS_target_swap - FAS_random_swap`
- target acquisition rank under clean/random/target-swap;
- target top-k recommendation fraction before any closed-loop update;
- validation MAE/R2 and their clean-vs-target-swap deltas;
- label histogram distance.

### M1 pass criteria

A first feasibility-positive M1 result should show:

- label histogram distance is exactly zero for targeted paired swap;
- `FAS_lift_vs_clean > 0`;
- `FAS_lift_vs_random > 0`;
- target acquisition rank or top-k target fraction improves materially under
  targeted swap;
- validation MAE/R2 does not collapse enough to trivially explain the effect;
- the effect appears in at least one neural surrogate, not only XGBoost.

Suggested initial numerical gates:

- FAS lift versus random is positive by at least one pooled standard error, or
  consistently positive in at least 4/5 seeds;
- target top-k recommendation fraction increases by at least 2x over clean, or
  by at least +10 percentage points when baseline prevalence is high;
- validation MAE degradation is modest relative to the target lift and does not
  create an obvious "bad model" explanation.

### M1 failure interpretation

- If XGBoost works but neural models fail, the central neural claim is not yet
  feasible; improve representation or use protein-LM embeddings.
- If random swaps match targeted swaps, the target construction is not specific
  enough.
- If validation collapses, the setting is too crude; lower the integrity-failure
  budget or choose a more coherent target.
- If FAS is positive but rank does not move, the model belief is too weak for
  closed-loop pursuit.

## Stage M2: Closed-Loop False Pursuit

### Goal

Show that the learned false regularity changes experimental allocation, not
just static predictions.

### Closed-loop setup

- Corrupt only the initial history.
- Train on recorded labels.
- Select batches from the unobserved candidate pool.
- Reveal true oracle labels for all newly selected candidates.
- Append true observations after each round.
- Track target allocation over rounds.

### Acquisition policies

Start with:

- greedy top predicted mean.

Then add if M2 passes:

- ensemble-UCB or Thompson-style acquisition.

### Primary metrics

- target selected count per round;
- cumulative target selection excess versus clean;
- cumulative target selection excess versus random swap;
- target batch fraction by round;
- target acquisition rank lift by round;
- persistence or half-life of target over-selection;
- true oracle fitness of selected target examples.

### Secondary metrics

- final best true label;
- cumulative regret;
- observed-set Jaccard versus clean;
- clean-only top candidate displacement.

### M2 pass criteria

A feasibility-positive M2 result should show:

- targeted swap selects materially more target candidates than clean;
- targeted swap selects materially more target candidates than random swap;
- selected target candidates remain low or non-exceptional under true labels;
- early closed-loop rounds show false pursuit before true feedback corrects it;
- endpoint metrics may remain plausible or mixed.

Suggested initial numerical gates:

- target allocation lift is at least 2x clean target fraction or at least +10
  percentage points in early rounds;
- cumulative target selection excess is positive in at least 4/5 seeds;
- true-label mean of selected target candidates is not high enough to justify
  the learned association;
- validation and endpoint metrics are not the only explanation for the shift.

### M2 failure interpretation

- If M1 passes but M2 fails, the surrogate learned the false association but the
  acquisition policy did not amplify it; use top-mean acquisition first and
  check rank-lift magnitude.
- If true feedback corrects after one round, the closed-loop claim is weak for
  that target; try lower batch size, larger initial history influence, or a
  more coherent target.
- If target allocation rises but selected target true labels are actually high,
  the regularity may be real, not false; reject that target.

## Stage M3: Robustness and Paper Feasibility

### Goal

Determine whether the result is strong enough for a real paper, not just a
single feasibility demo.

### Required robustness checks

- 5-10 seeds for the primary target.
- At least 2-3 target regions or target definitions.
- Random paired-swap controls matched by budget.
- Wrong-target control.
- Donor-only perturbation.
- Target-only relabel upper bound.
- At least one neural main model and one classical anchor.

### M3 pass criteria

The project is paper-feasible if:

- targeted paired misbinding beats random paired swap across targets or target
  definitions;
- the effect is visible in a neural scientific surrogate;
- false pursuit persists long enough to waste nontrivial early experimental
  budget;
- common audits are non-diagnostic rather than obviously flagging the run;
- the result does not depend on one cherry-picked target or seed.

### M3 failure interpretation

- If only one target works, keep the idea alive but do not claim generality.
- If only the strongest artificial construction works, move P2 plausible false
  records to a secondary stress test and weaken the conservative P1 claim.
- If only classical models work, the current neural false-science claim should
  be paused.

## Initial Run Order

| Step | Purpose | Runs | Gate |
| --- | --- | --- | --- |
| R0 | Verify data and target scan | GFP M0 scan, no model | At least one low, populated target |
| R1 | Check swap accounting | one target, one budget | exact label multiset preservation |
| R2 | Fast static sanity | XGBoost clean/random/target, 3 seeds | targeted > random on FAS/rank |
| R3 | Neural static feasibility | compact neural surrogate, 5 seeds | neural FAS lift and rank lift |
| R4 | Paper-facing static model | ESM embeddings + neural head, 5 seeds | stronger or consistent FAS/rank |
| R5 | Short closed-loop | top-mean acquisition, 5 seeds, 5-10 rounds | target allocation lift |
| R6 | Closed-loop robustness | 10-20 rounds, 5-10 seeds | persistence and oracle contradiction |
| R7 | Controls | wrong-target, donor-only, target-only upper bound | rule out artifacts |

## Minimum Result That Justifies Continuing

The smallest result that justifies fully pursuing the project is:

```text
On GFP, for a pre-specified low-true-fitness target region:

1. targeted paired swaps preserve the label multiset exactly;
2. a neural surrogate shows positive FAS_lift_vs_clean and FAS_lift_vs_random;
3. target acquisition rank or top-k target fraction rises materially;
4. a short closed-loop run selects substantially more target candidates than
   clean and random-swap controls;
5. true labels show the selected target region is not actually high-performing;
6. validation MAE/R2 and endpoint metrics do not trivially reveal the issue.
```

If these six conditions hold, the project has passed feasibility.

## Minimum Result That Justifies Paper Planning

The smallest result that justifies moving from feasibility to paper planning is:

```text
Across at least two target regions or target definitions, and at least one
neural scientific surrogate plus one conservative anchor:

1. targeted paired misbinding consistently induces FAS and rank lift;
2. targeted paired misbinding consistently increases closed-loop target
   allocation;
3. random paired swaps do not reproduce the effect;
4. true oracle labels contradict the target-high regularity;
5. common aggregate audits remain plausible;
6. results are stable over at least 5 seeds.
```

For a Nature/Science-family version, add:

```text
7. a second scientific domain or a convincing condition/provenance binding axis;
8. a lightweight diagnostic that catches the false regularity better than
   aggregate validation.
```

## Stop Conditions

Pause or redesign if any of the following happen:

- no viable low-performing target region exists in GFP;
- targeted and random paired swaps are indistinguishable on FAS and allocation;
- validation collapse explains the effect;
- true feedback corrects target over-selection immediately in all settings;
- the effect appears only in XGBoost/tree anchors and not in any neural model;
- the target region turns out to be genuinely high-performing under oracle
  labels;
- success requires unrealistic synthetic values or visible trigger fields.

## Immediate Open Decisions

- Which GFP dataset and preprocessing path to use.
- Whether the first target axis is motif/family or protein-LM embedding basin.
- Initial history size, batch size, and number of rounds.
- First neural model: compact descriptor MLP/TabM versus ESM embedding head.
- Initial integrity-failure budget grid.

