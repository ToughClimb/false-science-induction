# False-Science Induction Mechanistic Laws

Date: 2026-05-30

## Purpose

This note reframes the current evidence package from a warning study
("binding errors can cause false pursuit") into a mechanism study:

> targeted record misbinding rewrites conditional scientific relations, neural
> surrogates convert the rewritten relation into rank lift, closed-loop
> acquisition converts rank lift into budget concentration, and feedback
> attenuates the error only after cumulative false allocation.

The laws below are intended as paper-ready organizing principles. They are not
claims of universal vulnerability, universal stealth, real-world corruption, or
unbounded amplification.

## Status

COHERENT AFTER REFRAMING.

The evidence supports a set of empirical laws and one construction-level
invariance theorem. The strongest positive result is not merely that the failure
exists, but that the same mechanism predicts an early acquisition-trace audit.

## Law 1: Binding-Invariance / Conditional-Rewrite Law

### Statement

Paired record misbinding can preserve the marginal label multiset while
rewriting the conditional scientific relation observed by a model.

### Minimal form

For a paired swap corruption from clean dataset \(D\) to corrupted dataset
\(D'\),

\[
\{y_i : (x_i,c_i,p_i,y_i)\in D'\}
=
\{y_i : (x_i,c_i,p_i,y_i)\in D\}
\]

as a multiset, while

\[
\mathbb{E}_{D'}[Y \mid X \in \mathcal{T}, P=\tau]
\neq
\mathbb{E}_{D}[Y \mid X \in \mathcal{T}, P=\tau].
\]

### Evidence

- The paired swap construction preserves the label multiset by design.
- The detection-boundary table marks label histograms as PASS by construction.
- B18/B19/B22/B11 show false-pursuit effects despite preserved label multisets.

### What it explains

This explains why label histograms, range checks, and marginal outlier checks
are structurally insufficient: the corruption does not live in \(P(Y)\), but in
the binding that determines \(P(Y\mid X,C,P)\).

### What it does not support

It does not imply that all validation checks fail. Aggregate audit \(R^2\) can
degrade, and slice or trace diagnostics can expose the failure.

## Law 2: False-Association Learning Law

### Statement

When misbinding is coherent within a target basin or target-state slice, a
neural surrogate can internalize the induced binding relation as a false
scientific association.

### Minimal form

Define false-association strength

\[
\mathrm{FAS}
=
\frac{1}{|\mathcal{T}\cap\mathcal{C}|}
\sum_{i\in\mathcal{T}\cap\mathcal{C}} f(x_i)
-
\frac{1}{|\mathcal{B}|}\sum_{j\in\mathcal{B}} f(x_j).
\]

False-association learning is observed when
\(\mathrm{FAS}>0\) in targeted mode and not in clean/random controls.

For conditional-state settings, define trigger-toggle delta

\[
\Delta_{\tau}
=
\frac{1}{|\mathcal{T}\cap\mathcal{C}|}
\sum_{i\in\mathcal{T}\cap\mathcal{C}}
\left[f(x_i;\tau_{\mathrm{on}})-f(x_i;\tau_{\mathrm{off}})\right].
\]

Conditional false-association learning is observed when \(\Delta_{\tau}\) is
positive and target-specific.

### Evidence

- B23: trigger removal in targeted GFP removes the target advantage, with
  positive FAS, target-specific trigger interaction, rank-percentile lift, and
  individual triggered-target prediction lift.
- B22/B11: FAS and trigger-toggle effects grow with induced swap budget.
- B12: distributed trigger variants support conditional false pursuit.
- B24: FT-style neural surrogate learns the same mechanism.

### What it explains

This distinguishes the mechanism from a mere acquisition-count artifact. The
surrogate is learning a target/state-conditioned rule: the model's internal
ranking changes because the false relation is represented in its prediction
function.

### What it does not support

It does not imply all model classes learn the relation equally. Tree models and
other non-neural anchors are not the central vulnerable class in the present
evidence.

## Law 3: Rank-to-Budget Transduction Law

### Statement

Closed-loop acquisition turns a learned false association into experimental
budget concentration through top-ranked selection.

### Minimal form

For a monitored slice \(\mathcal{T}\), define acquisition concentration

\[
\Gamma_r
=
\frac{
  \text{cumulative acquisition fraction in } \mathcal{T} \text{ by round } r
}{
  \text{candidate prevalence of } \mathcal{T} \text{ at round } r
}.
\]

False pursuit is operationally visible when \(\Gamma_r\) is far above the
control-calibrated range.

### Evidence

- B18 materials: targeted MLP 41.2/250 versus random 0.1; TabM-mini 49.7 versus
  0.0.
- B19 GFP: targeted MLP 47.1/500 versus random 0.1; TabM-mini 36.1 versus 0.0.
- B25: 20% epsilon-greedy retains 89--91% of the greedy excess.
- B32: MC-dropout UCB remains full-scale: materials 42.3 versus 0.1 and GFP
  47.9 versus 0.1.
- B50: reduced-pool materials GP-BO replay remains positive but attenuated:
  coherent relinking produces 7.2 versus 2.6 final Co selections under GP-UCB,
  and 8.4 versus 2.8 under expected improvement.
- B33: round-1 concentration ratios separate targeted runs from controls across
  greedy, epsilon-greedy, MC-dropout UCB, and CAMEO replay families.

### What it explains

This explains why the failure is larger than ordinary prediction error. Once a
false association pushes a target slice above the acquisition threshold, the
closed loop repeatedly spends budget there. Moderate random exploration and
neural uncertainty scoring do not remove a systematic learned rank lift.

### What it does not support

It does not establish vulnerability for every acquisition policy. The current
GP-BO evidence is a reduced-pool replay with fixed kernels and smaller effect
sizes. Fully tuned GP-BO, Thompson sampling, and strong diversity constraints
remain future boundaries.

## Law 4: Saturating-Feedback Law

### Statement

True feedback attenuates or saturates false pursuit, but only after the loop has
already allocated cumulative budget to the false basin.

### Minimal form

Let \(N_T(r)\) be cumulative target acquisitions by round \(r\). The observed
regime is

\[
N_T(R) \gg N_T^{\mathrm{control}}(R),
\qquad
N_T(R)-N_T(r^*) \leq N_T(r^*) \text{ in late rounds},
\]

with the late gain depending on domain and acquisition batch structure.

### Evidence

- B18 materials long loop: cumulative targeted selections remain far above
  controls after ten rounds, with positive but attenuated later gains.
- B21 GFP long loop: false target count becomes high early, then post-round-5
  gain is approximately zero.
- B21 saturation check: GFP plateau is not simply exhaustion of the target pool;
  many triggered-target candidates remain unselected.

### What it explains

This explains why "feedback will fix it" is not enough. Feedback can correct or
flatten later acquisition, but it cannot recover already spent experiments. The
failure is bounded yet damaging.

### What it does not support

It does not support unbounded closed-loop amplification or indefinite growth.

## Law 5: Detection-Asymmetry Law

### Statement

False-science induction is hardest to see in marginal statistics and easiest to
see in the slice where the false relation drives acquisition.

### Minimal form

The supported audit hierarchy is:

1. label multiset: blind by construction;
2. global audit \(R^2\): configuration-dependent warning signal;
3. non-trigger audit slice: can remain plausible in conditional settings;
4. target/provenance slice diagnostics: exposes false association;
5. acquisition trace concentration: exposes downstream false pursuit in tested
   traces.

### Evidence

- Detection-boundary matrix: label histogram PASS; global audit often WARN;
  non-trigger audit can PASS.
- B23/B12: target-state counterfactual diagnostics reveal learned false
  association.
- B33: round-1 trace concentration flags all targeted traces with FPR 0 and TPR
  1 in the tested run families.

### What it explains

The false relation is localized in a target/provenance subspace and diluted in
global metrics. Once the loop acts on the relation, the acquisition trace
becomes a higher signal-to-noise observable than ordinary held-out accuracy.

### What it does not support

It does not make B33 a complete detector. The current detector requires that
the relevant target, provenance, source, or condition slice is monitored.

## Positive Result: Trace-Concentration Quarantine Governance

The most direct positive result is:

> false-science induction creates a predictable acquisition-trace signature, and
> a simple control-calibrated concentration audit can be turned into an
> actionable quarantine rule that prevents most or all false allocations in
> completed-trace replay.

Operationally, a closed-loop platform can monitor candidate slices by
composition family, mutation region, source, condition, batch, or provenance
state. If a slice receives acquisitions at a rate far above its candidate
prevalence and above clean/random control calibration, the platform flags the
run for provenance/binding audit before the full experimental budget is spent.

In B33, the audit detects targeted false-pursuit traces at round 1 in every
tested trace family. In B34, the same signal is used as an intervention replay:
if a proposed acquisition batch over-concentrates in the monitored slice above
the clean/random control-calibrated threshold, quarantine that slice before
executing the batch. This prevents 82.8%--100% of false allocations across the
tested greedy, epsilon-greedy, MC-dropout UCB, and CAMEO replay families, with
zero control quarantines under per-family calibration.

This is stronger than a warning but weaker than a complete defense. It is a
mechanism-derived governance intervention estimate, scoped to completed traces
and monitored slices.

## Suggested Paper Reframing

Current story:

> binding errors can cause closed-loop false discovery.

Mechanistic story:

> false-science induction follows a binding-to-budget transduction mechanism:
> misbinding rewrites conditional scientific relations, neural surrogates learn
> the rewritten relation as a false association, acquisition amplifies rank lift
> into budget concentration, feedback attenuates only after sunk false
> allocation, and the acquisition trace exposes the mechanism early when the
> relevant slice is monitored.

## Paper-Ready Paragraph

These results reveal a binding-to-budget transduction mechanism. Paired
misbinding leaves marginal label statistics unchanged but rewrites the
conditional relation between scientific object, provenance state, and
measurement. Neural surrogates can internalize this rewritten relation as a
target/state-conditioned false association, which closed-loop acquisition then
amplifies into experimental budget concentration. True feedback attenuates or
saturates the false pursuit, but only after cumulative budget has been spent.
The same mechanism yields an operational audit: if false pursuit is active, the
acquisition trace becomes anomalously concentrated in the induced target or
provenance slice, enabling early control-calibrated detection in the tested
settings.

## Next Manuscript Edits

1. Add a short subsection after the main result table:
   `Mechanistic laws of false-science induction`.
2. Rename B33 from "trace detector" to "trace-concentration governance audit".
3. Move B23 earlier, before broad robustness, because it is the mechanism proof.
4. State Law 1 and Law 3 in the Abstract or Introduction as the positive
   conceptual contribution.
5. In Discussion, describe B33/B34 as the first actionable mechanism-derived
   protocol: trace-concentration audit followed by slice quarantine in
   completed-trace replay, not as a complete defense.
