# Derivation Package

## Target

Strengthen the Nature-main manuscript with a compact mathematical mechanism for
false-science induction. The target is not a universal theorem over all
closed-loop discovery systems. It is a coherent finite-record derivation that
explains how targeted binding errors can be invisible to marginal label checks,
rewrite a conditional scientific record function, be faithfully learned by a
surrogate, and be converted by rank-based acquisition into experimental-budget
concentration.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The exact part is the finite-record binding-invariance boundary. The
budget-transduction part is a local approximation under stated smoothness and
score-density assumptions. This should be presented as a mechanism equation,
not as a distribution-free theorem.

## Invariant Object

The organizing object is the empirical record distribution \(P_D\) induced by
a finite scientific archive
\[
D=\{z_i\}_{i=1}^n,\quad z_i=(x_i,c_i,p_i,y_i),
\]
where \(x_i\) is the object, \(c_i\) is the experimental condition, \(p_i\) is
provenance or processing state and \(y_i\) is the response.

The key invariant under paired misbinding is the marginal label multiset
\(M_Y(D)=\{y_1,\ldots,y_n\}\). The non-invariant object is the conditional
record function
\[
g_D(u)=\mathbb{E}_D[Y\mid U=u],
\quad U=(X,C,P).
\]

## Assumptions

- The archive is finite and empirical expectations are over the finite record
  distribution.
- Paired misbinding swaps labels between disjoint target-donor pairs while
  leaving \(U=(X,C,P)\) fixed.
- A target slice \(\mathcal T\subseteq\mathcal U\) and donor set differ in
  mean response, so the donor-target contrast is nonzero.
- Same-distribution validation samples train and validation records from the
  same corrupted empirical distribution \(P_{D'}\).
- The local acquisition approximation assumes a fitted score
  \(f_D(u)\approx g_D(u)\), a target-score shift \(\Delta f_T\), and a
  rank-based acquisition threshold whose score density is locally nonzero.

## Notation

- \(U=(X,C,P)\): scientific object, condition and provenance tuple.
- \(D,D'\): clean and misbound finite archives.
- \(M_Y(D)\): multiset of labels in archive \(D\).
- \(g_D(u)=\mathbb{E}_D[Y\mid U=u]\): empirical conditional record function.
- \(\mathcal T\): target slice.
- \(n_T\): number of target records in the fitted history.
- \(m_c\): number of coherent target-side relinks.
- \(\Delta_{DT}\): donor-target label contrast over coherent relinks.
- \(\delta_T\): induced target-slice conditional mean shift.
- \(\Delta f_T\): fitted score shift on the target slice.
- \(q(\mathcal T)\): target prevalence in candidate pool.
- \(b(\mathcal T)\): target prevalence in a proposed acquisition batch.
- \(\eta_k\): local rank-cutoff sensitivity for a top-\(k\) acquisition rule.

## Derivation Strategy

1. Define records through \(U=(X,C,P)\) and response \(Y\).
2. Show exact label-multiset invariance under paired misbinding.
3. Show conditional-function rewrite on a target slice when donor-target
   contrast is nonzero.
4. State the same-distribution validation boundary: fitting \(D'\) certifies
   consistency with the recorded world, not with correct binding.
5. Use a first-order local ranking approximation to connect conditional
   score shift to acquisition-budget excess.

## Derivation Map

1. Exact identity:
   \[
   M_Y(D')=M_Y(D)
   \]
   under paired label swaps.
2. Exact consequence:
   any diagnostic \(\phi(D)=h(M_Y(D))\) is identical on \(D\) and \(D'\).
3. Conditional rewrite:
   \[
   \delta_T
   =
   \bar y_{\mathcal T,D'}-\bar y_{\mathcal T,D}
   =
   \frac{1}{n_T}
   \sum_{(i,j)\in\mathcal P_c,\ i\in\mathcal T}(y_j-y_i)
   \approx
   \frac{m_c}{n_T}\Delta_{DT}.
   \]
4. Fitting interpretation:
   if \(f_{D'}\approx g_{D'}\), then the surrogate is learning the corrupted
   record function faithfully.
5. Local acquisition approximation:
   \[
   b(\mathcal T)-q(\mathcal T)
   \approx
   \eta_k q(\mathcal T)\{1-q(\mathcal T)\}
   \frac{\Delta f_T}{\sigma_f}.
   \]

## Main Derivation

### Step 1: Exact label-multiset invariance

Let a paired misbinding operation choose disjoint pairs \((i,j)\) and replace
\[
(u_i,y_i),(u_j,y_j)
\mapsto
(u_i,y_j),(u_j,y_i).
\]
For each pair, the unordered label contribution \(\{y_i,y_j\}\) is unchanged.
All unpaired records are unchanged. Hence
\[
M_Y(D')=M_Y(D).
\]
Therefore, for any deterministic diagnostic depending only on the label
multiset, \(\phi(D)=h(M_Y(D))\), we have
\[
\phi(D')=h(M_Y(D'))=h(M_Y(D))=\phi(D).
\]
This covers label histograms, ranges, quantiles and marginal moments when
computed only from \(Y\).

### Step 2: Conditional-function rewrite

Let \(I_T=\{i:u_i\in\mathcal T\}\). Suppose \(m_c\) target-side records are
coherently relinked with donor labels. The target-slice mean shift is
\[
\delta_T
=
\frac{1}{n_T}\sum_{i\in I_T}(y'_i-y_i)
=
\frac{1}{n_T}
\sum_{(i,j)\in\mathcal P_c,\ i\in I_T}(y_j-y_i).
\]
Define
\[
\Delta_{DT}
=
\frac{1}{m_c}
\sum_{(i,j)\in\mathcal P_c,\ i\in I_T}(y_j-y_i).
\]
Then
\[
\delta_T=\frac{m_c}{n_T}\Delta_{DT}
\]
for exactly \(m_c\) coherent target-side swaps. The approximate form in the
manuscript allows heterogeneous partial coherence or mixed target support.
When \(\Delta_{DT}\neq0\), \(g_{D'}\) differs from \(g_D\) on the target slice
even though \(M_Y(D')=M_Y(D)\).

### Step 3: Same-distribution validation boundary

If train and validation data are both sampled from \(P_{D'}\), a sufficiently
flexible surrogate can reduce predictive loss by approximating
\[
g_{D'}(u)=\mathbb E_{D'}[Y\mid U=u].
\]
This validates consistency with the recorded distribution. It does not identify
whether \(D'\) is the correctly bound archive, because the same validation
procedure does not compare \(g_{D'}\) against \(g_D\) or against independent
trusted binding.

### Step 4: Local budget transduction

Let the acquisition score shift on \(\mathcal T\) be
\[
\Delta f_T
=
\mathbb E[f_{D'}(U)-f_D(U)\mid U\in\mathcal T].
\]
Under local surrogate response,
\[
\Delta f_T\approx \lambda_T\delta_T.
\]
For a top-\(k\) acquisition rule, the batch fraction of target records changes
when the score distribution near the rank cutoff is shifted. A first-order
threshold approximation gives
\[
b(\mathcal T)-q(\mathcal T)
\approx
\eta_k q(\mathcal T)\{1-q(\mathcal T)\}
\frac{\Delta f_T}{\sigma_f},
\]
where \(\sigma_f\) is a local score scale and \(\eta_k\) absorbs density at the
cutoff. Combining the preceding steps gives the local mechanism chain
\[
b(\mathcal T)-q(\mathcal T)
\approx
\eta_k q(\mathcal T)\{1-q(\mathcal T)\}
\frac{\lambda_T}{\sigma_f}
\frac{m_c}{n_T}\Delta_{DT}.
\]
This is the mathematical form of binding-to-budget transduction. It is local:
feedback, finite candidate pools, exploration, saturation and model capacity
can change the magnitude or threshold while preserving the mechanism.

## Remarks and Interpretation

- The exact theorem is not that all audits fail. It is that audits restricted
  to preserved marginal label information cannot distinguish \(D\) from \(D'\).
- The neural model is not treated as malfunctioning. If it approximates
  \(g_{D'}\), it is doing exactly what predictive training asks it to do.
- The false-science failure arises because the empirical record distribution is
  scientifically misbound, and acquisition turns the fitted conditional shift
  into experimental allocation.
- Coherence matters because random swaps do not concentrate \(\delta_T\) on a
  stable learnable direction, whereas block, plate, provenance or axis-aligned
  errors do.

## Boundaries and Non-Claims

- No claim of universal vulnerability across all closed-loop systems.
- No claim of universal stealth against all audits.
- No claim that public CAMEO, SAMPLE or BEAR archives are corrupt.
- No claim that the local budget formula is globally linear or
  distribution-free.
- No record-level correction guarantee; trace and feedback signals are
  stop-loss and triage tools.
