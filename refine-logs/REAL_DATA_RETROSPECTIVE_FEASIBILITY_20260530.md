# Real-Data Retrospective Feasibility Assessment

Date: 2026-05-30

## Decision

Yes. Adding external real-data evidence is worth doing, and it is probably the
highest-impact realism improvement left after B25/B31-style acquisition-policy
checks.

The claim it can unlock is not "natural corruption has already occurred" and
not "real wet-lab failure was observed." The defensible claim is:

> In an externally collected scientific closed-loop or self-driving-lab dataset,
> a small retrospective input-output/provenance relinking perturbation can
> redirect an acquisition replay toward a false target basin while preserving
> real input records and real measured outputs.

This directly addresses the reviewer objection that the current package proves
only that controlled benchmarks can be attacked. It moves the evidence toward
"real scientific data infrastructure / real closed-loop data streams can support
the same failure mechanism under retrospective perturbation."

## Current Evidence Gap

Current manuscript evidence is strong for mechanism:

- GFP and materials benchmarks.
- 10-seed confirmations.
- Neural architecture replication.
- Dose response and trigger ablations.
- B25 epsilon-greedy robustness.
- Long-loop persistence with attenuation/saturation.

But the realism gap remains:

- The corruptions are controlled by us.
- The closed-loop trajectories are simulated benchmark loops.
- The paper does not audit a deployed closed-loop stream.
- The paper cannot claim naturally observed real-world misbinding.

This is acceptable for a subjournal-level controlled mechanism paper, but it is
the most likely Nature-family reviewer attack.

## Candidate Ranking

| Candidate | Realism gain | Implementation cost | Risk | Recommendation |
|---|---:|---:|---:|---|
| CAMEO / NIST closed-loop materials discovery | High | Low-medium | Medium | Run first |
| SAMPLE protein self-driving lab | Very high | Medium-high | Medium-high | Run second if download/schema cooperate |
| Materials Project/OQMD/AFLOW provenance relinking | High | High | Medium | Good follow-up, not fastest |
| OpenCatalyst/PubChem/ProteinGym external datasets | Medium | Medium | Medium | Useful but weaker unless tied to real loop/provenance |
| Historical mix-up/correction case studies | Medium narrative gain | Low | Low | Add to intro/discussion, not a main experiment |

## Candidate 1: CAMEO / NIST Materials Closed Loop

Source inspected:

- NIST package: `https://data.nist.gov/od/ds/mds2-2480/CAMEO_NComm-master.zip`
- Local cached copy: `review-stage/CAMEO_NComm-master_20260530.zip`
- README cites: Kusne et al., "On-the-fly closed-loop materials discovery via
  Bayesian active learning", Nature Communications 2020.

Package facts from inspection:

- Size: 884,934 bytes.
- Contains real Fe-Ga-Pd data:
  - `FeGaPd_CMP.txt`: 278 compositions, 3 composition dimensions.
  - `FeGaPd_XRD.txt`: 278 XRD traces, 551 intensity dimensions.
  - `FeGaPd_Mag.txt`: 278 magnetic-property rows, 2 columns.
  - `FeGaPd_DFT_regions.txt`: 278 phase-region labels.
- Contains MATLAB active-learning / Bayesian-optimization code:
  - `running scripts/FeGaPd_ALBO_200801a.m`
  - `shared/run_active_learning_experiment_190618a2.m`
  - UCB-style phase-region BO is explicitly present.

Why it matters:

- This is not merely a real tabular benchmark.
- It is a published closed-loop materials discovery workflow with data and
  acquisition code.
- A Python retrospective replay can be presented as external real-data
  validation, not as another synthetic benchmark.

Best B31 experiment:

> Replay a simplified CAMEO-style closed-loop acquisition on the Fe-Ga-Pd data.
> Choose a low-true-magnetization composition/phase basin as the false target.
> Preserve all real measured magnetic values, but relink a small number of high
> magnetic values to target-basin records in the initial or early observed set.
> Compare clean, random relinking, and targeted relinking over 10 seeds.

Minimum viable implementation:

- Features: composition + compressed/standardized XRD, or composition-only first.
- Output: modified magnetization column from `FeGaPd_Mag.txt`, second column.
- Target basin: selected from low true-property composition/phase region with
  enough candidate support.
- Donors: high true-property records outside the target basin.
- Acquisition: uncertainty-aware GP/UCB or ensemble random-forest UCB; top-mean
  can be a smoke baseline but UCB is better because CAMEO used BO/UCB logic.
- Metrics:
  - target-basin acquisition count;
  - target-basin excess vs random relinking;
  - true property of selected target candidates;
  - prediction lift or rank lift for target basin;
  - exact sign-flip p-value across seeds if 10 seeds run.

Claim unlocked if positive:

> The false-science induction mechanism also appears in a retrospective replay
> of an external, published closed-loop materials discovery dataset.

What not to claim:

- Do not claim the original CAMEO campaign contained corruption.
- Do not claim the original paper was wrong.
- Do not claim natural deployment prevalence.
- Do not claim full fidelity to the MATLAB CAMEO algorithm unless we port it
  exactly. A simplified replay should be called a surrogate replay.

Stop conditions:

- Stop if no target basin has both low true property and at least 20 candidate
  records.
- Stop if clean/random acquisition already heavily targets the same basin.
- Stop if targeted relinking does not exceed random relinking in most seeds.
- Stop if the only positive result requires a target definition chosen after
  seeing the acquisition outcomes.

## Candidate 2: SAMPLE Protein Self-Driving Lab

Sources inspected:

- GLBRC page:
  `https://www.glbrc.org/data-and-tools/glbrc-data-sets/self-driving-laboratories-autonomously-navigate-protein-fitness`
- Zenodo record:
  `https://doi.org/10.5281/zenodo.10048592`

Facts from inspection:

- The GLBRC page describes the SAMPLE platform as a fully autonomous protein
  engineering / protein fitness landscape dataset.
- It cites Rapp et al., "Self-driving laboratories to autonomously navigate the
  protein fitness landscape", Nature Chemical Engineering 2024.
- Zenodo record `10048592` contains a code archive:
  `RomeroLab/SAMPLE_code-v1.0.0.zip`, size 131,499,259 bytes.
- The current network route downloaded too slowly during this audit, so schema
  has not yet been verified locally.

Why it matters:

- Strongest narrative fit for real closed-loop AI4Science.
- It directly answers "real SDL data" criticism.
- Protein domain pairs naturally with current GFP evidence.

Risk:

- Download and schema may cost time.
- Data may be organized around robot/code logs rather than one clean table.
- If round/acquisition metadata are absent or hard to parse, it becomes a
  longer engineering task.

Recommended role:

- Second real-data experiment after CAMEO, unless the code/data download becomes
  fast and the schema exposes rounds cleanly.

## Candidate 3: Provenance Relinking Across Materials Databases

Concept:

For the same or similar structure/composition, bind a label produced under one
calculation/measurement provenance to another provenance state, e.g. PBE vs
r2SCAN or different measurement conditions.

Why it is strong:

- It is closer to the "real scientific data infrastructure" story.
- It does not require fake labels or outliers.
- Both values can be true under their own provenance.

Why it is not first:

- Requires careful source alignment and provenance metadata.
- May require API access, large downloads, or deduplication by structure hash.
- Closed-loop replay must still be built on top.

Recommended role:

- Excellent B32/B33 if time remains.
- Not the fastest way to patch the current realism gap.

## Recommended Immediate Plan: B31 CAMEO

Hypothesis:

Small retrospective relinking of real measured magnetic-property values in an
external CAMEO Fe-Ga-Pd closed-loop dataset can cause a neural/ensemble
acquisition replay to over-select a low-true-property composition or phase basin.

Budget:

- 1 day for parser, target-basin scan, and smoke replay.
- 1 day for 10-seed run, statistics, and one figure.
- No GPU required; CPU scikit-learn is enough for the first pass.

Acceptance criteria:

- Targeted relinking produces higher final target-basin acquisition than random
  relinking in at least 8/10 seeds.
- Mean targeted excess is practically visible, not just statistically positive.
- Selected target candidates have low true magnetic property relative to the
  full pool or donor set.
- Label multiset preservation is verified.

Paper placement if positive:

- Main text: one concise paragraph in Results as "external retrospective
  closed-loop materials replay."
- Figure: a small panel in the robustness/realism figure or Extended Data.
- Methods: clearly label it as retrospective surrogate replay of a published
  dataset.

Paper placement if weak but informative:

- Supplementary note only, or discussion as a boundary.
- Do not dilute the current main result with a weak external experiment.

## Editorial Assessment

The AI feedback is directionally right: the current package is already strong
for a controlled mechanism paper, but the realism attack is the most obvious
reviewer path. A successful CAMEO or SAMPLE retrospective would materially
increase Nature Machine Intelligence / Nature Computational Science plausibility.

It still would not make a Nature main-journal claim by itself. For that, the
paper would likely need either naturally observed corruption, a true deployed
system audit, or unusually broad external replication. But for a Nature-family
subjournal, one well-executed external closed-loop retrospective is a large
upgrade.
