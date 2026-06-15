# B43-B46 Nature-Main Gap Results

Date: 2026-05-30

## Purpose

Dual-model review identified four Nature-main gaps:

- targeted corruptions looked too adversarial;
- the dose evidence was not framed as an operating boundary;
- quarantine lacked ROC/sensitivity analysis;
- the trace monitor appeared to assume the suspicious slice was known.

## B43 Realistic Relinking Boundary

Run:

- `runs/20260530T163036Z_b43-materials-realistic-relinking-mlp-10seed-80ep-20260530`

Design:

- Dataset: `matbench_expt_gap`
- Target axis for evaluation: `major_element=Co`
- Model: MLP
- Seeds: 0-9
- Modes: clean, random-pair swap, targeted swap, sorted join shift, block/plate cycle shift
- All modes preserve the history label multiset.

Main result:

| Mode | Final Co count | Excess vs random-pair | FAS lift vs random-pair |
|---|---:|---:|---:|
| clean | 0.0 | 0.0 | 0.0065 |
| random pair | 0.0 | 0.0 | 0.0000 |
| sorted join shift | 0.0 | 0.0 | 0.0093 |
| targeted pair | 15.0 | 15.0 | 0.8657 |
| block/plate cycle shift | 15.0 | 15.0 | 0.8657 |

Interpretation:

The result is not "any accidental relinking is catastrophic." A generic
lexicographic join shift did not induce Co pursuit. A coherent block/plate-like
cycle shift that moves high donor labels onto the low Co block did induce the same
false pursuit as targeted pairing. This strengthens the mechanism by identifying
the relevant condition: binding errors become dangerous when their structure is
coherent with a learnable scientific or provenance axis.

## B44 Minimum-Effective Dose / Saturation

Artifact:

- `review-stage/b44_phase_diagram_20260530.md`

Result:

- GFP MLP/TabM and materials MLP/TabM are already effective at the smallest tested dose, 5 swaps.
- Materials acquisition peaks at 25 swaps and falls at 50 swaps despite FAS continuing to rise.
- GFP acquisition is already saturated at 5 swaps, while FAS grows with swap count.

Interpretation:

The supported claim is an operating boundary, not linear amplification:
small coherent binding errors can cross the false-pursuit threshold, and acquisition
then saturates or becomes policy/candidate dependent.

## B45 Quarantine ROC

Artifact:

- `review-stage/b45_quarantine_roc_20260530.md`

Zero-control-FPR operating points:

| Setting | Threshold | TPR | Prevented fraction |
|---|---:|---:|---:|
| GFP MLP | 2.000 | 0.980 | 1.000 |
| Materials MLP | 1.000 | 1.000 | 1.000 |
| CAMEO RF-UCB | 3.000 | 0.767 | 0.837 |

Interpretation:

The online stop-loss rule is robust over threshold sweeps in GFP and materials.
CAMEO is weaker but still prevents most proposed target-region allocation at a
zero-control-FPR threshold. This should be framed as threshold sensitivity for
a stop-loss rule, not as complete detection.

## B46 All-Axis Blind Monitor

Artifact:

- `review-stage/b46_all_axis_blind_monitor_20260530.md`

Result on proposed traces:

| Setting | Control any-axis flag | Target any-axis flag | Target-axis flag | Target-axis top-1 |
|---|---:|---:|---:|---:|
| GFP MLP | 0.000 | 1.000 | 1.000 | 1.000 |
| Materials MLP | 0.000 | 1.000 | 1.000 | 1.000 |
| CAMEO RF-UCB | 0.000 | 1.000 | 0.900 | 0.900 |

Interpretation:

The monitor no longer requires naming the injected target as the monitored slice.
It scans all enumerable axes in the proposed acquisition trace and uses the target
axis only for post-hoc evaluation. Executed traces are expected to lose this signal
after successful quarantine, so proposed traces are the correct online monitoring
surface.

## Claim Impact

New supported claims:

- False pursuit has a coherence boundary: generic local relinking may be harmless,
  while block-structured relinking aligned with a learnable scientific axis can
  produce the same false pursuit as targeted paired swaps.
- The minimum tested effective dose is low (5 swaps in both primary domains), but
  acquisition is saturated/nonlinear rather than unbounded.
- Online quarantine has threshold-sensitivity evidence, not only one chosen threshold.
- Blind all-axis monitoring can identify the implicated axis from proposed traces
  without specifying the target slice in advance.

Still unsupported:

- Natural corruption prevalence.
- Live wet-lab deployment validation.
- Complete detector or record-level correction.
- Universal vulnerability or universal stealth.
