# B20 Cross-Domain 10-Seed Figure Package Result

Date: 2026-05-29

## Purpose

B18 and B19 converted the main materials and GFP trigger-gated results from low-seed pilots into 10-seed paired evidence. B21 then added the GFP 10-round long-loop counterpart. B20 turns these results into paper-facing figures generated directly from fixed config files and raw run artifacts.

## Inputs

- Config: `configs/b20_cross_domain_10seed_figures_20260529.json`
- Script: `scripts/generate_cross_domain_10seed_figures.py`
- Materials short-loop summary: `runs/20260529T113213Z_b18-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-10seed-80ep/summary_by_model_mode.csv`
- Materials long-loop summary: `runs/20260529T113808Z_b18-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-10seed-80ep/summary_by_model_mode.csv`
- GFP short-loop summary: `runs/20260529T115315Z_b19-gfp-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-tabm-10seed-80ep/summary_by_model_mode.csv`
- GFP long-loop summary: `runs/20260529T122521Z_b21-gfp-pos27-disttrigger-dim32-s003-long10-bg2048-mlp-tabm-10seed-80ep/summary_by_model_mode.csv`
- Materials statistics: `runs/b18_statistics_aggregate_20260529.csv`
- GFP short-loop statistics: `runs/b19_statistics_aggregate_20260529.csv`
- GFP long-loop statistics: `runs/b21_statistics_aggregate_20260529.csv`

## Generated Figures

- `docs/figures/b20_cross_domain_10seed_final_counts.png`
- `docs/figures/b20_cross_domain_10seed_final_counts.svg`
- `docs/figures/b20_seed_difference_distributions.png`
- `docs/figures/b20_seed_difference_distributions.svg`
- `docs/figures/b20_audit_r2_boundary.png`
- `docs/figures/b20_audit_r2_boundary.svg`
- `docs/figures/b20_long_loop_trajectories.png`
- `docs/figures/b20_long_loop_trajectories.svg`

## Figure Roles

### Cross-Domain Final Counts

`b20_cross_domain_10seed_final_counts` shows the central false-science induction effect across materials short-loop, materials long-loop, GFP short-loop, and GFP long-loop settings. Clean and random controls remain zero or near-zero, while targeted swap induces large triggered-target acquisition in both neural models.

### Seed Difference Distributions

`b20_seed_difference_distributions` shows paired seed differences for the same 10-seed main effects. For materials long-loop the plotted statistic is post-round-5 gain; for the other panels it is final cumulative count. Every plotted final-count difference is positive in both GFP short-loop and GFP long-loop settings, while the materials long-loop panel shows the smaller but consistently positive late-gain effect. This figure is the most direct visual support for the statistical claim.

### Audit Boundary

`b20_audit_r2_boundary` shows targeted-mode audit R2 values. It is intended to prevent overclaiming: successful trigger-gated false pursuit can coexist with plausible audit behavior in these settings, but the figure does not support a universal stealth claim.

### Long-Loop Trajectories

`b20_long_loop_trajectories` shows the distinct closed-loop dynamics in the two long-loop settings. Materials continues to accumulate triggered-target selections after the early burst, whereas GFP reaches a high cumulative false-pursuit state early and then plateaus. This figure supports persistence with attenuation or saturation, not unbounded pursuit.

## Verification

Commands completed:

- `conda run --no-capture-output -n agentconda python scripts/generate_cross_domain_10seed_figures.py --config configs/b20_cross_domain_10seed_figures_20260529.json`
- `conda run --no-capture-output -n agentconda python -m pytest -q`
  - Result: `46 passed in 5.30s`
- `rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'`
  - Result: no matches. Exit code `1` is expected for no ripgrep matches.
- PNG visual inspection was performed with `view_image`; the figures are nonblank and have readable layout after style tightening.

Stable generated hashes:

- `docs/figures/b20_cross_domain_10seed_final_counts.png`: `4c6bbd631c4b5cdf0ffb111be1bf50bd18c5b330fcaddada90da0aa32ba48a18`
- `docs/figures/b20_seed_difference_distributions.png`: `c8c565ffda01a81deed1b6287040fe1d1646e24811541955f5ef749a1a798df3`
- `docs/figures/b20_audit_r2_boundary.png`: `c93974f07499bf540ded1f3d49244c8cf83cda6ca5be146cd79a860104eeb851`
- `docs/figures/b20_long_loop_trajectories.png`: `2162873be645a9bc26fab7995036f07dae92ab927f6fa61eb08c55ed8562f612`

## Verdict

B20 is successful. The main cross-domain 10-seed evidence now includes GFP short-loop and long-loop panels, has generated PNG/SVG figures suitable for paper drafting, and the generation path is config-driven and reproducible.
