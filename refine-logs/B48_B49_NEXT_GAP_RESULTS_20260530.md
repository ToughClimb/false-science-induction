# B48-B49 Next Nature-Main Gap Results

Date: 2026-05-30

## Purpose

B47 review identified two linked gaps:

- B43 established a binary coherence boundary but did not sweep coherence
  continuously.
- B46 removed the known-slice assumption but still used clean/random controls to
  calibrate thresholds.

B48 and B49 test those gaps directly.

## B48 Materials Coherence Sweep

Run:

- `runs/20260530T170338Z_b48-materials-coherence-sweep-mlp-10seed-80ep-20260530`

Design:

- Dataset: `matbench_expt_gap`
- Target axis for evaluation: `major_element=Co`
- Model: MLP
- Seeds: 0-9
- Total swaps fixed at 25
- Coherent relinking fraction: 0.00, 0.25, 0.50, 0.75, 1.00
- All modes preserve the history label multiset.

Main result:

| Coherent relinking fraction | Coherent pairs | Final Co acquisitions | FAS lift vs 0 coherence |
|---:|---:|---:|---:|
| 0.00 | 0 | 0.0 | 0.000 |
| 0.25 | 6 | 4.2 | 0.379 |
| 0.50 | 12 | 8.9 | 0.606 |
| 0.75 | 19 | 12.4 | 0.774 |
| 1.00 | 25 | 15.0 | 0.866 |

Interpretation:

B48 turns B43's binary coherence boundary into a graded operating law. With the
swap count and label multiset fixed, acquisition and false-association strength
increase with the fraction of relinked records that coherently move high donor
measurements onto the low Co block. This supports a binding-to-budget
transduction claim: false pursuit is not produced by arbitrary relinking, but by
the degree to which relinking rewrites a learnable conditional axis.

## B49 Within-Campaign Blind Monitor

Artifacts:

- `review-stage/b49_within_campaign_blind_monitor_20260530.csv`
- `review-stage/b49_within_campaign_blind_monitor_20260530.json`
- `review-stage/b49_within_campaign_blind_monitor_20260530.md`

Design:

- Uses proposed traces from B37/B38/B39.
- Thresholds are computed from the same campaign's round-0 proposed trace and
  same-round peer-axis distribution.
- Clean/random modes are used only after the fact to estimate false alarms, not
  to set thresholds.

Main result:

| Dataset | Target-axis flag | Target-axis top-1 | Evaluation-only control any-axis flag |
|---|---:|---:|---:|
| GFP | 1.000 | 1.000 | 0.850 |
| Materials | 1.000 | 1.000 | 0.925 |
| CAMEO | 0.200 | 0.800 | 0.838 |

Interpretation:

The calibration-free variant recovers the target axis in GFP and materials, but
its any-axis false-alarm rate is too high for a stand-alone stop rule. The root
cause is scientific rather than implementation-specific: clean closed-loop
optimization naturally concentrates on axes, so concentration alone cannot be a
calibration-free detector. This strengthens the detection-boundary section by
showing why the B45/B46 calibrated trace monitor is needed, and why "blind" must
remain a triage/review surface rather than a complete detector.

## Claim Impact

New supported claims:

- With total swaps and label multiset fixed, coherent relinking fraction produces
  a graded increase in both false-association strength and final target-axis
  budget.
- The coherence condition is not merely binary: partial coherent relinking is
  already sufficient to induce measurable false allocation in materials.
- A fully within-campaign any-axis concentration alarm is not reliable as a
  stand-alone detector because clean optimization also creates concentration.

Still unsupported:

- Natural corruption prevalence.
- Live wet-lab deployment validation.
- Calibration-free complete detection.
- Record-level correction.
- Universal vulnerability or universal stealth.
