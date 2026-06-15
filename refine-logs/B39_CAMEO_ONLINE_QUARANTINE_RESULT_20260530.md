# B39 CAMEO Online Trace Quarantine Result

Date: 2026-05-30

## Purpose

Test whether trace-concentration quarantine can act online in an external real
closed-loop materials replay, rather than only in internally constructed GFP
and Matbench loops.

## Hypothesis

In the CAMEO Fe-Ga-Pd retrospective replay, targeted relinking should induce
over-concentrated proposals for the low-true-property DFT region. A
control-calibrated online concentration rule should prevent most of those
target-region proposals from being executed while leaving clean and random
relinking controls unquarantined.

## Configs

Smoke:

- `configs/smoke_b39_cameo_online_quarantine_20260530.json`

Full:

- `configs/b39_cameo_online_quarantine_rfucb_10seed_20260530.json`

Full command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/b39_cameo_online_quarantine.py \
  --config configs/b39_cameo_online_quarantine_rfucb_10seed_20260530.json
```

Run directory:

- `runs/20260530T155109Z_b39-cameo-online-quarantine-rfucb-10seed`

## Protocol

- Dataset: public CAMEO Fe-Ga-Pd package.
- Target: DFT region 2, selected by the same pre-outcome rule as B31.
- Model/acquisition: random-forest ensemble UCB.
- Modes: clean, random relinking, targeted relinking.
- Seeds: 0--9.
- Rounds: 6.
- Batch size: 8.
- Quarantine threshold: `2.9829545454545454`, the B34 CAMEO clean/random
  maximum concentration ratio.
- Intervention:
  - propose UCB batch;
  - compute target-region concentration from the current candidate pool and
    proposed batch only;
  - if above threshold, drop target-region records and refill from next ranked
    non-target candidates;
  - train the next round only on executed feedback.

## Main Result

Summary file:

- `runs/20260530T155109Z_b39-cameo-online-quarantine-rfucb-10seed/online_quarantine_summary_by_mode.csv`

| Mode | Quarantine rate | Proposed target allocations | Executed target allocations | Prevented | Prevented fraction |
|---|---:|---:|---:|---:|---:|
| Clean | 0.0000 | 3.3 | 3.3 | 0.0 | 0.0000 |
| Random relinking | 0.0000 | 2.6 | 2.6 | 0.0 | 0.0000 |
| Targeted relinking | 0.7667 | 37.7 | 1.5 | 36.2 | 0.9602 |

## Interpretation

B39 extends the online stop-loss result to real external CAMEO measurements
under controlled relinking. The result is smaller and noisier than B38 because
CAMEO is a small 278-record dataset and the target region has high natural
prevalence, but the conclusion is clear: targeted relinking drives many
target-region proposals, and an online concentration rule prevents most of
them from being executed without firing on clean or random controls.

This substantially strengthens the Nature-main story because the same
mechanism-derived intervention now holds in:

- GFP computational closed-loop simulation (B37);
- Matbench materials computational closed-loop simulation (B38);
- external real closed-loop materials replay on public CAMEO measurements (B39).

## Claim Boundaries

Supported:

- Online stop-loss quarantine can prevent false budget execution in a real-data
  retrospective closed-loop replay under controlled relinking.
- The CAMEO result is an external realism bridge for the governance mechanism.

Not supported:

- The original CAMEO campaign contained binding errors.
- The replay is a faithful reproduction of the original MATLAB CAMEO algorithm.
- The rule is a complete detector without a monitored target/provenance axis.
- The rule repairs corrupted records.

## Verification

Commands run:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b38_materials_online_quarantine.py \
  tests/test_b39_cameo_online_quarantine.py \
  tests/test_no_defaults_policy.py

conda run --no-capture-output -n agentconda \
  python scripts/b39_cameo_online_quarantine.py \
  --config configs/smoke_b39_cameo_online_quarantine_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/b39_cameo_online_quarantine.py \
  --config configs/b39_cameo_online_quarantine_rfucb_10seed_20260530.json
```

Test result:

- `9 passed in 0.98s` for B38/B39/no-default tests before full run.
