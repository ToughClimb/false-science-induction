# B10 Materials Multi-Basin Replication Plan

Date: 2026-05-29

## Hypothesis

The B7 materials result is not unique to `major_element=Co`. Targeted paired swaps should implant false high-band-gap associations in multiple low true-gap composition basins and induce closed-loop acquisition toward those basins.

## Basins

Five additional basins are selected from the M0 scan:

- `major_element=Ni`
- `major_element=Pd`
- `major_element=Rh`
- `major_element=Ti`
- `major_element=Mn`

Each basin passes the same M0 gate used for B7:

- minimum target count: 80
- maximum prevalence: 0.35
- target quantile cutoff: 0.6
- donor quantile cutoff: 0.9
- swap count: 25

## Budget

Each config uses:

- Dataset: `matbench_expt_gap`
- Models: MLP, TabM-mini, XGBoost
- Seeds: 0, 1, 2
- Modes: clean, random paired swap, targeted paired swap
- Rounds: 5
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

Configs:

- `configs/b10_materials_multibasin_major_element-Ni_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- `configs/b10_materials_multibasin_major_element-Pd_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- `configs/b10_materials_multibasin_major_element-Rh_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- `configs/b10_materials_multibasin_major_element-Ti_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- `configs/b10_materials_multibasin_major_element-Mn_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`

## Acceptance Criteria

B10 supports multi-basin replication if:

1. At least three of five new basins show targeted final target count above both clean and random controls for MLP or TabM-mini.
2. At least three of five basins show positive targeted FAS lift versus random for a neural model.
3. Label multiset is preserved in every run.
4. Clean/random controls remain low enough that targeted pursuit is not explained by baseline prevalence.

## Stop Conditions

Stop and inspect before continuing if:

1. A config fails the M0 gate.
2. A run crashes with an implementation error.
3. Label multiset preservation fails.
4. Targeted mode is not above controls for all five basins.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda \
  python scripts/materials_false_regulariry.py \
  --config <B10_CONFIG>
```

