# B79 Robust Surrogate Baselines Result

Date: 2026-05-31

## Hypothesis

If generic robust supervised losses are sufficient defenses against false-axis
allocation, replacing ordinary MSE training with Huber or trimmed-MSE training
should substantially reduce targeted-minus-random triggered-target acquisitions
under the same B18/B19 relinking stress tests.

## Runs

- Script: `scripts/b79_robust_surrogate_baselines.py`
- Smoke config: `configs/smoke_b79_robust_surrogate_baselines_20260531.json`
- Decision config: `configs/b79_decision_robust_surrogate_baselines_20260531.json`
- Source-faithful full config: `configs/b79_robust_surrogate_baselines_20260531.json`
- Decision run directory: `runs/20260531T025334Z_b79-decision-robust-surrogate-baselines-5seed-20ep-20260531`
- Source artifacts replayed:
  - GFP B19: `runs/20260529T115315Z_b19-gfp-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-tabm-10seed-80ep`
  - Materials B18: `runs/20260529T113213Z_b18-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-10seed-80ep`
- Decision budget: 5 seeds, 5 rounds, 20 epochs, MSE/Huber/10% trimmed MSE.
- Full source-faithful run note: the initial 10-seed, 80-epoch run was stopped
  after about 30 minutes because the first implementation wrote no partial
  outputs. The script was then changed to checkpoint `round_metrics.partial.csv`
  and `selected_records.partial.csv` after each dataset/seed/mode/model
  combination.

## Sanity Check

The decision-run MSE replay reproduces the source MLP behavior within the
predefined tolerance:

| Dataset | Mode | Decision MSE final count | Source MLP final count | Absolute delta | Pass |
|---|---:|---:|---:|---:|---|
| GFP | clean | 0.0 | 0.1 | 0.1 | yes |
| GFP | random | 0.0 | 0.1 | 0.1 | yes |
| GFP | targeted | 44.0 | 47.1 | 3.1 | yes |
| Materials | clean | 0.0 | 0.0 | 0.0 | yes |
| Materials | random | 0.0 | 0.1 | 0.1 | yes |
| Materials | targeted | 48.4 | 41.2 | 7.2 | yes |

## Result

| Dataset | Loss | Seeds | Final targeted count | Excess vs random | Delta vs MSE |
|---|---:|---:|---:|---:|---:|
| GFP | MSE | 5 | 44.0 | 44.0 | 0.0 |
| GFP | Huber | 5 | 44.2 | 44.2 | 0.2 |
| GFP | 10% trimmed MSE | 5 | 50.0 | 50.0 | 6.0 |
| Materials | MSE | 5 | 48.4 | 48.4 | 0.0 |
| Materials | Huber | 5 | 47.8 | 47.8 | -0.6 |
| Materials | 10% trimmed MSE | 5 | 0.0 | 0.0 | -48.4 |

Huber loss does not materially reduce false-axis allocation in either domain.
Trimmed MSE is domain-dependent: it increases GFP false-axis allocation in this
decision run but suppresses the materials construction. This supports a narrow
claim that generic robust regression is not a general substitute for
binding-aware validation, trace quarantine and feedback triage. It also
identifies trimmed training as a possible mitigation boundary that requires
domain-specific validation rather than a universal defense.

## Manuscript Claim Supported

Generic robust surrogate training is not a complete binding defense. In
source-artifact replays, Huber-trained MLPs preserve the false-axis allocation
effect in GFP and materials, while trimmed-loss training is inconsistent across
domains.

## Unsupported Claims

- No claim that Huber or trimmed losses are fully evaluated defenses.
- No claim that trimmed loss is a reliable mitigation across domains.
- No claim that robust training replaces online quarantine or provenance audit.
- No claim based on the stopped 10-seed, 80-epoch full run.
