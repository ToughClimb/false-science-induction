# B38 Materials Online Trace Quarantine Result

Date: 2026-05-30

## Purpose

Test whether the online trace-concentration quarantine result from B37
generalizes from GFP to the flagship materials B18 setting.

## Hypothesis

The control-calibrated concentration threshold from B34 should prevent false
budget execution in the materials MLP closed loop while leaving clean and
random-swap controls unquarantined.

## Configs

Smoke:

- `configs/smoke_b38_materials_online_quarantine_mlp_20260530.json`

Full:

- `configs/b38_materials_online_quarantine_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json`

Full command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_online_quarantine.py \
  --config configs/b38_materials_online_quarantine_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json
```

Run directory:

- `runs/20260530T154250Z_b38-materials-online-quarantine-disttrigger-dim32-s001-25swap-bg1024-mlp-10seed-80ep`

## Protocol

- Dataset: Matbench experimental band gap.
- Target axis: `major_element=Co`.
- Model: MLP.
- Seeds: 0--9.
- Modes: clean, random swap, targeted swap.
- Rounds: 5.
- Batch size: 50.
- Quarantine rule:
  - compute proposed-batch triggered-target concentration online;
  - compare with B34 materials clean/random calibrated threshold
    `0.8874074074074074`;
  - if above threshold, drop the monitored triggered-target slice and refill
    from the next ranked non-quarantined candidates;
  - train the next round only on executed feedback.

## Main Result

Summary file:

- `runs/20260530T154250Z_b38-materials-online-quarantine-disttrigger-dim32-s001-25swap-bg1024-mlp-10seed-80ep/online_quarantine_summary_by_model_mode.csv`

| Mode | Quarantine rate | Proposed false allocations | Executed false allocations | Prevented | Prevented fraction |
|---|---:|---:|---:|---:|---:|
| Clean | 0.00 | 0.0 | 0.0 | 0.0 | 0.0000 |
| Random swap | 0.00 | 0.1 | 0.1 | 0.0 | 0.0000 |
| Targeted swap | 1.00 | 222.9 | 0.0 | 222.9 | 1.0000 |

Seed-level targeted proposed/executed/prevented totals:

| Seed | Proposed | Executed | Prevented |
|---:|---:|---:|---:|
| 0 | 219 | 0 | 219 |
| 1 | 225 | 0 | 225 |
| 2 | 228 | 0 | 228 |
| 3 | 222 | 0 | 222 |
| 4 | 216 | 0 | 216 |
| 5 | 218 | 0 | 218 |
| 6 | 224 | 0 | 224 |
| 7 | 226 | 0 | 226 |
| 8 | 220 | 0 | 220 |
| 9 | 231 | 0 | 231 |

## Interpretation

B38 confirms that online trace-concentration quarantine is not GFP-specific.
In the flagship materials domain, the surrogate repeatedly proposes the false
Co triggered-target axis under targeted misbinding, but the online rule prevents
all proposed false allocations from being executed. Clean and random-swap
controls produce zero quarantine events.

This turns the paper's governance result into a cross-domain online stop-loss
protocol across GFP and materials.

## Claim Boundaries

Supported:

- Cross-domain online prevention of false budget execution in the tested GFP
  and materials computational loops.
- Control-calibrated thresholding can act before execution and before the next
  training round.

Not supported:

- Complete defense without a monitored slice.
- Record-level correction.
- Live wet-lab validation.
- Natural corruption prevalence.

## Verification

Commands run:

```bash
conda run --no-capture-output -n agentconda \
  python -m pytest -q \
  tests/test_b38_materials_online_quarantine.py \
  tests/test_no_defaults_policy.py

conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_online_quarantine.py \
  --config configs/smoke_b38_materials_online_quarantine_mlp_20260530.json

conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_online_quarantine.py \
  --config configs/b38_materials_online_quarantine_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json
```

Test result:

- `8 passed in 0.87s` for B38 unit/no-default tests before full run.
