# B11 Materials Swap-Count Dose Response Plan

Date: 2026-05-29

## Hypothesis

If paired input-output misalignment implants a false scientific regularity, increasing the number of targeted swaps should generally increase false association strength and closed-loop target-basin acquisition.

## Design

- Dataset: `matbench_expt_gap`
- Target basin: `major_element=Co`
- Mechanism: B7-style history-only paired label swap
- Swap counts: 5, 10, 25, 50
- Models: MLP and TabM-mini
- Seeds: 0, 1, 2
- Modes: clean, random paired swap, targeted paired swap
- Rounds: 5
- Batch size: 50
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

Configs:

- `configs/b11_materials_dose_co_5swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b11_materials_dose_co_10swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b11_materials_dose_co_25swap_bg1024_mlp_tabm_3seed_80ep.json`
- `configs/b11_materials_dose_co_50swap_bg1024_mlp_tabm_3seed_80ep.json`

## Acceptance Criteria

B11 supports dose response if:

1. Targeted final target count at 25 or 50 swaps exceeds 5 swaps.
2. Targeted FAS lift at 25 or 50 swaps exceeds 5 swaps.
3. Clean/random controls do not show the same increasing trend.
4. Label multiset is preserved in every run.

The trend does not need to be perfectly monotonic at every adjacent point because closed-loop acquisition is discrete and can saturate.

