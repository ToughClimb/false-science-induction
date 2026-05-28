# Config-Governed Rerun Results - 2026-05-28

## Scope

All reruns used commit `94ce54b` after config-only execution and explicit-failure
cleanup. Commands used `conda run --no-capture-output -n agentconda` and wrote
new, non-overwriting run directories under `runs/`.

## Commands And Runs

- M1 static GFP pos27:
  `python scripts/m1_static_false_association.py --config configs/m1_gfp_pos27_static_xgb_mlp_25swap_bg2048_3seed.json`
  - Run: `runs/20260528T050319Z_m1-gfp-pos27-static-xgb-mlp-25swap-bg2048-3seed`
- M2 main GFP pos27:
  `python scripts/m2_closed_loop_false_pursuit.py --config configs/m2_gfp_pos27_topmean_50swap_bg1024_5seed_80ep.json`
  - Run: `runs/20260528T050442Z_auditfix-m2-pos27-50swap-bg1024-5seed-80ep`
- M2 GFP pos27 mechanism controls:
  `python scripts/m2_closed_loop_false_pursuit.py --config configs/m2_gfp_pos27_controls_50swap_bg1024_3seed_80ep.json`
  - Run: `runs/20260528T050731Z_auditfix-m2-pos27-controls-50swap-bg1024-3seed-80ep`
- M2 GFP pos27 stealth/low-swap:
  `python scripts/m2_closed_loop_false_pursuit.py --config configs/m2_gfp_pos27_topmean_15swap_bg4096_10round_3seed_80ep.json`
  - Run: `runs/20260528T050729Z_auditfix-m2-pos27-15swap-bg4096-10round-3seed-80ep`
- M2 random low-set control:
  `python scripts/m2_random_set_control.py --config configs/m2_gfp_random_low_set_control_50swap_bg1024_3seed_80ep.json`
  - Run: `runs/20260528T051432Z_auditfix-m2-random-low-set-control-50swap-bg1024-3seed-80ep`

## Key Findings

### M1 Static False Association

The static model clearly learned the injected false regularity.

- MLP targeted swap:
  - FAS lift vs random: `+0.685687`
  - top-k target fraction: `0.046000`
  - audit R2: `0.665714`
- XGBoost targeted swap:
  - FAS lift vs random: `+0.595044`
  - top-k target fraction: `0.002667`
  - audit R2: `0.541920`

Interpretation: both neural and conservative tree anchors learn a target-high
association, with the neural MLP showing the stronger target-top-k response.

### M2 Main Closed-Loop Pursuit

The main GFP pos27 closed-loop result is strong, but not stealthy in aggregate
MAE/R2.

- targeted swap:
  - mean batch target fraction: `0.144`
  - final target excess vs random: `+12.56`
  - FAS lift vs random: `+0.765333`
  - selected target true mean: `2.028472`
  - audit R2: `0.437600`
- clean/random selected no target records in the closed-loop batches.

Interpretation: this is strong evidence for false-science pursuit, but it should
not be used as the main evidence that MAE/R2 is non-diagnostic.

### Mechanism Controls

Controls confirm that target-side high binding, not arbitrary label noise, drives
target pursuit.

- targeted swap:
  - final target excess vs random: `+11.733333`
  - FAS lift vs random: `+0.780357`
- target-only high relabel:
  - final target excess vs random: `+12.866667`
  - FAS lift vs random: `+0.655488`
- donor-only swap:
  - final target excess vs random: `0`
  - FAS lift vs random: `+0.044633`
- random swap:
  - final target excess vs random: `0`

Interpretation: the critical mechanism is binding high recorded outcomes to the
target motif/provenance basin.

### Low-Swap Stealth Setting

The 15-swap, larger-history setting shows weaker but still directional pursuit,
while audit metrics remain much closer to clean/random.

- targeted swap:
  - mean batch target fraction: `0.013333`
  - final target excess vs random: `+2.4`
  - FAS lift vs random: `+0.241350`
  - audit R2: `0.804424`
- clean audit R2: `0.829821`
- random audit R2: `0.819497`

Interpretation: this is the current best evidence for the audit-boundary claim,
but behavioral pursuit is weaker than the main 50-swap setting.

### Random Low-Set Control

The random low-set control is much weaker than structured pos27.

- targeted swap:
  - mean batch target fraction: `0.003333`
  - final target excess vs random: `+0.266667`
  - FAS lift vs random: `+0.084797`
  - audit R2: `0.416280`

Interpretation: arbitrary low-label target sets do not reproduce the structured
pos27 closed-loop pursuit strength. This supports the importance of learnable
motif/provenance structure.

## Current Claim Support

Supported:

- Models can learn a specified false association from paired label/provenance
  misalignment.
- Closed-loop discovery can allocate experimental budget toward a non-existent
  motif/provenance basin.
- Mechanism controls show that target-side high binding is the driver.
- A low-swap regime can preserve more plausible MAE/R2 while retaining weaker
  false-pursuit behavior.

Not yet fully supported:

- The strongest pursuit setting also has obvious audit degradation.
- The stealth setting supports audit plausibility but has modest allocation
  effect. It needs either more seeds, a better target, or a tuned intermediate
  swap/history regime to become a flagship result.
