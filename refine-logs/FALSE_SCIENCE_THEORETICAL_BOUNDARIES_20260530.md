# False-Science Theoretical Boundaries

Date: 2026-05-30

## Target

Build a defensible type-II theory line for false-science induction:

> what can and cannot be inferred from marginal labels, ordinary predictive
> validation, and closed-loop traces when record bindings may be wrong.

## Status

COHERENT AFTER REFRAMING.

The current evidence does not support a capacity-threshold impossibility theorem
such as "above model capacity C, false discovery is inevitable." It does
support a cleaner and more defensible boundary:

> label-preserving misbinding can be statistically invisible to any diagnostic
> that only observes the marginal label multiset, while closed-loop acquisition
> can make the corrupted conditional relation visible through budget
> concentration and feedback conflict.

## Invariant Object

The invariant object is the observed scientific record

\[
Z=(X,C,P,Y),
\]

where:

- \(X\): scientific object, such as a protein sequence or material composition;
- \(C\): experimental or task condition;
- \(P\): provenance, batch, source, trace, or binding state;
- \(Y\): measured property.

The key distinction is:

- marginal label information: \(P(Y)\);
- scientific relation: \(P(Y\mid X,C,P)\);
- closed-loop action trace: \(A_{1:R}\), the acquired objects over rounds.

False-science induction lives in the conditional relation and its downstream
action trace, not necessarily in the marginal label distribution.

## Assumptions

### A1. Record-Binding Representation

Each row is interpreted as a binding between object, condition/provenance, and
measurement:

\[
z_i=(x_i,c_i,p_i,y_i).
\]

### A2. Paired Misbinding Operation

For selected pairs \((i,j)\), the operation swaps labels while keeping the
objects and metadata fixed:

\[
(x_i,c_i,p_i,y_i),(x_j,c_j,p_j,y_j)
\mapsto
(x_i,c_i,p_i,y_j),(x_j,c_j,p_j,y_i).
\]

### A3. Targeted Conditional Contrast

There exists a monitored slice \(\mathcal T\) and a donor set \(\mathcal D\)
such that the donor labels are systematically larger than target labels:

\[
\mathbb E[Y\mid Z\in \mathcal D]
>
\mathbb E[Y\mid Z\in \mathcal T].
\]

### A4. Acquisition Uses Model Ranking

The closed loop uses a learned score \(f_r(x,c,p)\) to rank or partially rank
candidates. Greedy, epsilon-greedy, and UCB variants in the experiments are
instances of this assumption.

### A5. Monitored Axis for Trace Recovery

For trace detection or axis recovery, the relevant target/provenance/source
axis must be enumerable or monitored. Without this, B33/B34/B35 are not
complete detectors.

## Notation

- \(D\): clean dataset.
- \(D'\): dataset after paired misbinding.
- \(\mathcal T\): target slice, such as `pos=27` or Co-containing materials.
- \(\mathcal D\): donor slice.
- \(H_r\): observed history before acquisition round \(r\).
- \(S_r\): candidate set at round \(r\).
- \(B_r\): proposed acquisition batch at round \(r\).
- \(q_r(\mathcal T)=|\mathcal T\cap S_r|/|S_r|\): candidate prevalence.
- \(b_r(\mathcal T)=|\mathcal T\cap B_r|/|B_r|\): batch fraction.
- \(\Gamma_r(\mathcal T)=b_r(\mathcal T)/q_r(\mathcal T)\): concentration
  ratio.

## Proposition 1: Binding-Invariance Boundary

### Statement

Paired misbinding preserves the marginal label multiset exactly while changing
the conditional scientific relation on targeted slices.

### Proof

For each swapped pair \((i,j)\), the clean labels \(\{y_i,y_j\}\) become
\(\{y_j,y_i\}\). Therefore the multiset of labels is unchanged for the pair.
All unswapped records are unchanged. Taking the union over swapped and
unswapped records gives:

\[
\{y : (x,c,p,y)\in D'\}
=
\{y : (x,c,p,y)\in D\}
\]

as a multiset.

If \(i\in\mathcal T\), \(j\in\mathcal D\), and \(y_j>y_i\), then the recorded
target value increases and the donor value decreases. Over many such pairs,

\[
\mathbb E_{D'}[Y\mid Z\in \mathcal T]
-
\mathbb E_D[Y\mid Z\in \mathcal T]
>0
\]

whenever enough target records are swapped with higher-valued donors. Thus
the marginal label distribution is invariant while the conditional relation is
rewritten.

### Consequence

Any diagnostic whose input is only the label multiset, label histogram, label
range, or marginal moments cannot distinguish \(D\) from \(D'\). This is an
exact impossibility for marginal-only diagnostics, not an empirical trend.

## Proposition 2: Prediction-Accuracy Non-Identifiability

### Statement

Held-out predictive accuracy is not, by itself, an identifier of correct
binding. A model can score well on a held-out set drawn from the same corrupted
record distribution while learning the corrupted conditional relation.

### Construction

Let \(D'\) define the observed training and validation distribution. A
sufficiently flexible predictor can approximate

\[
f'(x,c,p) \approx \mathbb E_{D'}[Y\mid X=x,C=c,P=p].
\]

If validation records are sampled from the same corrupted distribution \(D'\),
then the validation target is also governed by the corrupted conditional
relation. Good validation error against \(D'\) therefore shows consistency with
the recorded distribution, not consistency with the clean scientific binding in
\(D\).

### Boundary

This does not mean every accuracy audit is blind. If the audit set is clean,
independent, slice-aware, or uses true feedback after acquisition, prediction
metrics can expose degradation. The non-identifiability applies to
same-binding-distribution validation as a sole safeguard.

## Proposition 3: Rank-to-Budget Transduction

### Statement

If a learned false association lifts a target slice into the top acquisition
region, rank-based acquisition converts the score lift into budget
concentration.

### Derivation

Let \(S_r\) be the candidate set and let \(B_r\) be the top-\(k\) batch under
score \(f_r\). If the learned score ranks a fraction \(\alpha\) of target-slice
candidates inside the top-\(k\) region while the target prevalence in the
candidate set is \(q_r\), then

\[
b_r(\mathcal T)
=
\frac{|B_r\cap\mathcal T|}{|B_r|}
\]

can greatly exceed \(q_r\). The downstream concentration ratio is

\[
\Gamma_r(\mathcal T)
=
\frac{b_r(\mathcal T)}{q_r(\mathcal T)}.
\]

When \(\Gamma_r(\mathcal T)\) exceeds the control-calibrated range, the loop is
spending budget on the slice faster than its candidate prevalence justifies.

### Evidence

B18/B19/B25/B32 show large targeted acquisition counts under greedy,
epsilon-greedy, and MC-dropout UCB acquisition. B33 shows round-1 concentration
separation in tested traces. B34 shows that quarantining over-concentrated
batches would prevent 82.8%--100% of false allocations in completed-trace
replay.

## Proposition 4: Feedback-Conflict Recovery

### Statement

When the loop has true post-acquisition feedback, induced false hypotheses can
be used as probes: a false axis is visible as the conjunction of high allocation
and low true feedback.

### Operational Score

For an enumerable candidate axis \(a\), define

\[
S(a)
=
\frac{n_a}{n}
\cdot
\max(0,\bar y_{\mathrm{selected}}-\bar y_a).
\]

Here \(n_a/n\) is the fraction of selected budget on axis \(a\), and
\(\bar y_{\mathrm{selected}}-\bar y_a\) is the feedback deficit for the same
axis.

### Evidence

B35 recovers the injected axis at aggregate rank 1 in completed targeted traces:

- GFP B19: `pos=27` is aggregate rank 1 for MLP and TabM-mini; seed top-2
  recovery is 10/10 for both models.
- Materials B18: Co axis aliases are aggregate ranks 1 and 2 for MLP and
  TabM-mini; seed top-2 recovery is 10/10 for both models.
- Clean and random-swap controls never recover the target axis at seed top-1 or
  aggregate top-1.

### Boundary

This is axis-level recovery, not record repair. It requires an axis vocabulary
and true feedback on selected records.

## What This Gives the Manuscript

The stronger type-II/type-III result is:

> False science is not merely a bad outcome. It exposes a basic identifiability
> boundary: marginal labels and same-distribution validation cannot certify
> correct scientific binding. But the closed-loop action trace creates a new
> observable: budget concentration and feedback conflict can detect, quarantine,
> and sometimes recover the implicated hypothesis axis.

This is more than "be careful." It is a mechanism-to-governance line:

1. binding-invariance theorem explains why marginal checks fail by construction;
2. prediction non-identifiability explains why same-distribution validation is
   not enough;
3. rank-to-budget transduction explains why the error becomes experimentally
   expensive;
4. trace quarantine gives a stop-loss intervention;
5. feedback-conflict recovery uses the false hypothesis itself as an axis-level
   diagnostic probe.

## Non-Claims

Do not claim:

- a universal capacity threshold;
- inevitable false discovery for every model above some size;
- impossibility of all validation;
- unbounded closed-loop amplification;
- automatic correction of corrupted records;
- real-world natural corruption has already occurred in a named dataset;
- recovery without an enumerable or monitored axis vocabulary.
