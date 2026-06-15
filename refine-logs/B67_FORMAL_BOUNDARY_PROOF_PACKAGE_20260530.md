# B67 Formal Boundary Proof Package

Date: 2026-05-30

## Target

Formalize the mechanism-level theory in a way that is honest enough for a
Nature-main-first submission package:

1. prove the label-multiset invariance boundary for paired misbinding;
2. prove a same-distribution validation non-identifiability statement;
3. state the coherence-to-budget relation as an empirical operating
   relationship rather than a universal theorem.

## Status

PROVABLE AFTER WEAKENING / EXTRA ASSUMPTION.

The strong claim "all audits are blind" is false because known-slice,
provenance-aware and paired conditional checks can expose the rewrite. The
provable claim is narrower:

> any diagnostic whose input is only the label multiset, or any deterministic
> statistic of that multiset, must return the same value on a clean dataset and
> its paired-swap version.

The same-distribution validation statement is also a non-identifiability
construction rather than a claim that predictive validation is always useless.

## Assumptions

- A dataset is a finite multiset of records
  \(D=\{(x_i,c_i,p_i,y_i)\}_{i=1}^n\).
- A paired misbinding operation selects disjoint unordered index pairs
  \(\mathcal P\) and swaps only the measured labels inside each pair:
  \[
  (x_i,c_i,p_i,y_i),(x_j,c_j,p_j,y_j)
  \mapsto
  (x_i,c_i,p_i,y_j),(x_j,c_j,p_j,y_i).
  \]
- The object, condition and provenance coordinates remain fixed.
- A marginal-label diagnostic is any deterministic map
  \(\phi\) whose input is the multiset \(\{y_i\}_{i=1}^n\).
- Same-distribution validation means training and validation records are both
  sampled from the same observed record distribution \(D'\).

## Notation

- \(D\): clean record multiset.
- \(D'\): paired-swap record multiset.
- \(M_Y(D)\): marginal label multiset of \(D\).
- \(\mathcal T\): target slice.
- \(\mathcal D\): donor slice.
- \(g_D(x,c,p)=\mathbb E_D[Y\mid X=x,C=c,P=p]\): empirical or population
  conditional record function.

## Proposition 1: Marginal-Label Invariance

### Claim

For any paired misbinding operation, \(M_Y(D')=M_Y(D)\). Therefore every
diagnostic \(\phi(M_Y(\cdot))\) that depends only on the label multiset returns
the same value on \(D\) and \(D'\).

### Proof

Consider one swapped pair \((i,j)\). Before the operation, the pair contributes
the multiset \(\{y_i,y_j\}\) to \(M_Y(D)\). After the operation, the same two
records contribute \(\{y_j,y_i\}\), which is the same multiset. All unswapped
records contribute the same labels before and after the operation. Because the
selected pairs are disjoint, the multiset union over all swapped pairs and
unswapped records is unchanged. Thus \(M_Y(D')=M_Y(D)\).

Let \(\phi\) be any deterministic diagnostic that takes only \(M_Y(D)\) as
input. Since \(M_Y(D')=M_Y(D)\), \(\phi(M_Y(D'))=\phi(M_Y(D))\). Therefore no
such diagnostic can distinguish \(D\) from \(D'\).

### Boundary

This proposition says nothing about diagnostics that use \(X,C,P\), known
slices, donor-target pairing information, clean independent feedback, or
acquisition traces.

## Proposition 2: Conditional Rewrite

### Claim

If a nonempty target slice \(\mathcal T\) is swapped with donor records whose
mean label differs from the target mean, the conditional record function on
\(\mathcal T\) changes even though \(M_Y(D')=M_Y(D)\).

### Proof

Let \(I_T\) be the set of target records that are swapped and \(J_D\) their
paired donor records. The average recorded target-label change over the swapped
target records is
\[
\frac{1}{|I_T|}\sum_{i\in I_T} (y_{\pi(i)}-y_i),
\]
where \(\pi(i)\in J_D\) is the donor paired with target record \(i\). If this
average is nonzero, the empirical conditional mean of the target slice changes
on the swapped target records. In particular, if the donor mean exceeds the
target mean over the selected pairs, the recorded target-slice mean increases.

The object, condition and provenance coordinates are fixed, so the changed mean
is not a change in the scientific objects being queried. It is a change in the
label binding attached to those objects. Therefore \(g_{D'}\) and \(g_D\) need
not agree on \(\mathcal T\), although Proposition 1 shows their marginal label
multisets agree.

### Boundary

The proposition is conditional on a nonzero donor-target contrast. Random swaps
with zero expected conditional contrast need not create a coherent false
scientific relation.

## Proposition 3: Same-Distribution Validation Non-Identifiability

### Claim

Same-distribution predictive validation alone cannot identify whether a model
is approximating \(g_D\) or \(g_{D'}\), because a validation set sampled from
\(D'\) is scored against the corrupted conditional record function.

### Proof

Suppose training and validation records are sampled from the same observed
record distribution \(D'\). The validation loss for a predictor \(f\) is a
function of prediction error against labels distributed according to \(D'\). A
predictor that approximates \(g_{D'}(x,c,p)=\mathbb E_{D'}[Y\mid X=x,C=c,P=p]\)
can therefore obtain good validation loss on records drawn from \(D'\).

If \(g_{D'}\ne g_D\) on a target slice, good validation loss against \(D'\) does
not imply that \(f\) approximates \(g_D\) on that slice. It establishes
consistency with the observed record distribution, not correctness of the
underlying binding.

### Boundary

This is not a statement that all predictive validation fails. Clean independent
validation, slice-aware validation, true feedback after execution, or
provenance-aware audits can reveal disagreement with the intended scientific
relation.

## Empirical Operating Relationship: Coherence-to-Budget Transduction

The experiments support the following bounded empirical relationship:

> holding swap count and label multiset fixed, false-axis acquisition increases
> when relinking becomes coherent along a learnable target/provenance axis.

This is not presented as a universal theorem. It is supported by B48 and related
stress tests:

- B48 fixed-swap materials sweep: Co acquisitions rise from 0.0 to 15.0 as
  coherence rises from 0 to 1.
- B43 realistic relinking: a generic sorted join shift yields 0.0 Co
  selections, while block or plate-like relinking yields 15.0.
- B57 cross-family analysis: nonzero coherent-risk cases are directionally
  positive, but a single global linear predictor does not explain magnitudes
  across model/policy/domain families.

## Non-Claims

- No universal vulnerability.
- No universal stealth.
- No claim that natural corruption occurred.
- No theorem that model capacity makes false discovery inevitable.
- No calibration-free complete detector.
- No record-level correction.
