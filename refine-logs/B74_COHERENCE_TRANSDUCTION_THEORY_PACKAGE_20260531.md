# B74 Coherence-Transduction Theory Package

Date: 2026-05-31

## Target

Formalize the mechanism behind false-science induction without overclaiming a
universal theorem:

1. coherent paired misbinding rewrites a conditional scientific relation;
2. random paired misbinding with the same swap count is mean-zero with respect
   to any fixed target axis unless it aligns with a donor-target contrast;
3. a correctly functioning surrogate can faithfully fit the rewritten record
   distribution;
4. rank-based acquisition converts the induced score lift into budget
   concentration.

The intended manuscript role is a mechanism principle and supplementary theory
note, not a complete impossibility theorem for all detectors or all closed-loop
systems.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.

The clean statement is not "large models inevitably hallucinate false science."
The clean statement is:

> False pursuit requires a coherent conditional rewrite. At a fixed swap budget
> and fixed label multiset, the expected target-score shift is controlled by
> the alignment between swapped labels and a learnable target axis. Random
> relinking contributes noise around the global mean; coherent relinking
> contributes a systematic conditional shift that rank acquisition can turn
> into budget.

## Invariant Object

The invariant object is the empirical record distribution
\[
D=\{(x_i,c_i,p_i,y_i)\}_{i=1}^n,
\]
where \((x,c,p)\) defines the scientific object, condition and provenance state,
and \(y\) is the measured response. The surrogate does not optimize the hidden
clean scientific function directly; it optimizes the observed conditional
record function
\[
g_D(x,c,p)=\mathbb{E}_D[Y\mid X=x,C=c,P=p].
\]

The false-science mechanism is therefore a record-function rewrite:
\[
g_D \longrightarrow g_{D'}.
\]
The model can behave normally by fitting \(g_{D'}\). The scientific failure
arises because the closed loop treats \(g_{D'}\) as if it were the intended
scientific relation \(g_D\).

## Assumptions

- A1. Records have a feature vector \(\phi_i=\phi(x_i,c_i,p_i)\in\mathbb{R}^d\).
- A2. A target axis is represented by an indicator or score
  \(t_i\in\{0,1\}\), where \(t_i=1\) means record \(i\) belongs to the
  monitored target slice \(\mathcal T\).
- A3. A donor set \(\mathcal D\) has higher true/recorded labels than the
  target records selected for relinking. The donor-target contrast is
  \[
  \Delta_{DT}
  =
  \frac{1}{m}\sum_{(i,j)\in\mathcal P,\,i\in\mathcal T}
  (y_j-y_i).
  \]
- A4. Coherent relinking means a fraction \(\rho\) of the swapped target
  records are aligned with the same learnable axis \(\mathcal T\). The coherent
  pair count is \(m_c=\rho m\).
- A5. The surrogate score after fitting is monotone, at least locally, in the
  estimated conditional mean. This covers greedy top-mean and approximates the
  exploitation component of UCB-style acquisition.
- A6. The top acquisition batch has size \(k\), candidate target prevalence
  \(q=|\mathcal T\cap S|/|S|\), and target-score noise scale \(\sigma_f\).

## Notation

- \(D\): clean record distribution.
- \(D'\): observed record distribution after paired misbinding.
- \(\mathcal T\): target slice.
- \(\mathcal D\): donor slice.
- \(m\): total paired swaps.
- \(m_c=\rho m\): coherent target-aligned swaps.
- \(n_T\): number of effective target records in the fitted history.
- \(\Delta_{DT}\): average donor-target label lift in swapped pairs.
- \(\delta_T\): induced target conditional-mean shift.
- \(f_D,f_{D'}\): fitted surrogate scores under clean and relinked records.
- \(B_k(f)\): top-\(k\) acquisition batch under score \(f\).
- \(b(\mathcal T)=|B_k(f)\cap\mathcal T|/k\): target fraction in the batch.
- \(\Gamma(\mathcal T)=b(\mathcal T)/q\): trace-concentration ratio.

## Derivation Strategy

Use a local slice model rather than a universal closed-loop theorem:

1. derive the conditional-mean rewrite on the target slice;
2. show why coherent swaps have nonzero expected target shift and random swaps
   do not, conditional on a fixed axis;
3. connect the shift to a fitted score under a stable learner;
4. map score lift to budget concentration through rank acquisition;
5. state the empirical operating law and the boundaries.

## Derivation Map

1. Paired swaps preserve \(M_Y(D)\), the marginal label multiset.
2. Target-aligned swaps change the empirical target-slice mean by
   approximately \((m_c/n_T)\Delta_{DT}\).
3. Random swaps have expected axis-aligned shift near zero after centering by
   the global label mean; their effect is variance, not a coherent target lift.
4. A stable fitted surrogate transmits this conditional shift into a score
   shift \(\Delta f_T\approx \lambda_T\delta_T\), where \(\lambda_T\) is a
   local learner susceptibility.
5. A rank policy maps \(\Delta f_T/\sigma_f\) to an overrepresentation of
   \(\mathcal T\) in the top-\(k\) batch.
6. The tested empirical risk score
   \[
   R=\rho\,(\Delta_{DT}/s_Y)\sqrt{m_c/n_T}
   \]
   is a scale-normalized proxy for this chain. It is predictive within the
   fixed-swap coherence sweep and directionally positive across stress-test
   families, but not a universal magnitude law.

## Main Derivation

### Step 1. Marginal-label invariance is exact

For every swapped pair \((i,j)\), the labels \(\{y_i,y_j\}\) become
\(\{y_j,y_i\}\). Therefore
\[
M_Y(D')=M_Y(D).
\]
Any diagnostic whose input is only \(M_Y(D)\) must return the same value on
\(D\) and \(D'\). This is an identity, not an empirical approximation.

### Step 2. Coherent swaps induce a conditional target shift

Let \(I_T\) be the target records in the fitted history and let
\(|I_T|=n_T\). If \(m_c\) of these target records receive donor labels through
coherent paired misbinding, the target empirical mean changes by
\[
\delta_T
=
\bar y_{T,D'}-\bar y_{T,D}
=
\frac{1}{n_T}\sum_{(i,j)\in\mathcal P_c,\,i\in\mathcal T}(y_j-y_i).
\]
If the coherent donor-target contrast is approximately constant,
\[
\delta_T \approx \frac{m_c}{n_T}\Delta_{DT}.
\]
This is the basic conditional-function rewrite. The label multiset is
unchanged, but the fitted target-slice relation is shifted upward.

### Step 3. Random swaps are not equivalent to coherent swaps

For random paired relinking that is independent of the fixed target axis, the
expected donor label assigned to a target record is the global label mean
\(\mu_Y\), up to finite-population corrections. If the target labels are also
centered relative to the same pool, the expected axis-aligned shift is
\[
\mathbb{E}[\delta_T^{\mathrm{random}}]
\approx
\frac{m_T}{n_T}(\mu_Y-\bar y_T).
\]
In the controlled random-swap comparisons, target-aligned donor selection is
not performed; the resulting shifts are dispersed across many directions and
do not form a consistent learnable rule for \(\mathcal T\). Their main effect
is added noise. By contrast, coherent relinking concentrates the shift on the
same axis, producing a systematic signal.

Thus equal swap count is not equal mechanism strength. The relevant quantity is
the aligned conditional shift, not the number of wrong bindings alone.

### Step 4. A normal surrogate fits the rewritten record function

Let a fitted model have local target-score response
\[
\Delta f_T
=
\mathbb{E}[f_{D'}(X,C,P)-f_D(X,C,P)\mid (X,C,P)\in\mathcal T].
\]
For a stable learner in the local slice regime,
\[
\Delta f_T \approx \lambda_T \delta_T,
\]
where \(\lambda_T\ge 0\) summarizes model capacity, feature representation,
training regularization and acquisition-time uncertainty handling. This is an
approximation, not a theorem about all neural networks.

The important interpretation is that a high-capacity neural surrogate is not
malfunctioning when \(\Delta f_T>0\). It is fitting the observed conditional
record function \(g_{D'}\). The scientific error is upstream in the record
binding and downstream in the decision loop that spends budget according to
that fitted relation.

### Step 5. Rank acquisition turns score lift into budget concentration

For a top-\(k\) acquisition policy, the target batch fraction is
\[
b(\mathcal T)
=
\frac{|B_k(f_{D'})\cap\mathcal T|}{k}.
\]
If the target-score lift \(\Delta f_T\) moves target candidates across the
top-\(k\) threshold, then
\[
b(\mathcal T)>q(\mathcal T),
\qquad
\Gamma(\mathcal T)=\frac{b(\mathcal T)}{q(\mathcal T)}>1.
\]
In a smooth threshold approximation, the excess target fraction scales with
the standardized score lift:
\[
b(\mathcal T)-q(\mathcal T)
\approx
\eta_k\,q(\mathcal T)\{1-q(\mathcal T)\}\frac{\Delta f_T}{\sigma_f},
\]
where \(\eta_k\) depends on the local score density near the acquisition
threshold. This expression is an interpretation of the rank-to-budget link,
not a distribution-free guarantee.

Combining Steps 2, 4 and 5 gives the local operating relation
\[
b(\mathcal T)-q(\mathcal T)
\propto
\lambda_T\eta_k\,
\frac{m_c}{n_T}\,
\frac{\Delta_{DT}}{\sigma_f},
\]
up to saturation, feedback correction and finite candidate-pool effects.

### Step 6. Empirical risk score used in B57

The experiments use a scale-normalized proxy
\[
R=\rho\,(\Delta_{DT}/s_Y)\sqrt{m_c/n_T},
\]
where \(s_Y\) is the outcome scale. The square-root term is a conservative
signal-to-noise proxy rather than an exact mean-shift formula. B57 shows that
inside the fixed-swap B48 coherence sweep this mechanism-risk score explains
the graded target-capacity allocation (\(R^2=0.956\)), whereas a swap-count-only
score is uninformative (\(R^2=0.000\)). Across dose, GP-BO, CAMEO and SAMPLE
rows, nonzero coherent-risk cases are directionally positive in 17/17 rows, but
one global linear formula does not explain magnitudes across model, policy and
domain families.

## Remarks and Interpretation

- The model is not the villain in the core mechanism. A surrogate that fits
  \(g_{D'}\) is performing its training task. The failure is that the record
  world \(D'\) no longer represents the intended scientific relation.
- The central law is coherence transduction: wrong bindings matter when their
  errors align into a learnable conditional relation.
- This explains why "universal vulnerability" would be the wrong claim.
  Susceptibility depends on donor-target contrast, coherent pair count,
  target support, feature representation, acquisition policy and feedback.
- It also explains why same-distribution accuracy can be misleading: it can
  certify consistency with \(D'\), not correctness relative to \(D\).

## Boundaries and Non-Claims

- This is not a universal theorem that all sufficiently large models must
  induce false discovery.
- This is not a claim that random swaps are harmless in every dataset; random
  swaps can damage accuracy or accidentally align with some axis. The claim is
  that controlled coherent relinking creates a stronger target-aligned signal
  than same-budget random relinking in the tested settings.
- This is not a complete detector or defense.
- This does not establish natural corruption prevalence in deployed campaigns.
- This does not perform record-level correction.
- This does not imply unbounded amplification; true feedback can attenuate or
  saturate the failure after budget has already been spent.

## Open Risks

- The local linear response \(\Delta f_T\approx\lambda_T\delta_T\) is an
  approximation. It should be described as a mechanism model, not a proof for
  arbitrary neural training dynamics.
- The rank-threshold approximation depends on the score distribution near the
  acquisition cutoff.
- Blind detection without a monitored or enumerable axis remains unresolved in
  high-false-alarm regimes.
- A natural documented binding-error replay would still be the strongest
  evidence upgrade for Nature main.

