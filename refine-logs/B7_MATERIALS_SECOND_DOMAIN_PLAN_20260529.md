# B7 Materials Second-Domain Pilot Plan

Date: 2026-05-29

## Hypothesis

In Matbench experimental band gap, a small number of paired label swaps can implant a false association between a low true-gap composition basin and high band-gap performance. A closed-loop surrogate trained on the corrupted history should allocate new acquisitions toward that basin more often than clean and random-swap controls, even though the swapped label multiset is preserved.

## Dataset and Mechanism

- Dataset: `matbench_expt_gap`
- Input: composition-derived tabular features from elemental fractions and basic elemental statistics
- Output: experimental band gap, column `gap expt`
- Target family selection: automatic M0 scan over element, major-element, chemistry, and element-count tags
- Smoke-selected target: `major_element=Co`
- Mechanism: history-only targeted paired label swap
- Distribution check: target and donor labels are swapped in pairs, preserving the label multiset among swapped records

## Budget

- Config: `configs/b7_materials_matbench_expt_gap_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- Seeds: `0, 1, 2`
- Modes: `clean`, `random_swap`, `targeted_swap`
- Models: `mlp`, `tabm_mini`, `xgboost`
- Rounds: 5
- Batch size: 50
- Swap count: 25
- Background history size: 1024
- Audit size: 1024
- Neural training budget: 80 epochs per fit

## Acceptance Criteria

The second-domain pilot is considered feasible if:

1. The run completes without silent fallback or overwritten output.
2. The target scan selects a low true-gap basin with high donor contrast.
3. `label_multiset_preserved` is true.
4. For at least one neural model, targeted swap yields higher final cumulative target count than both clean and random-swap controls.
5. False association metrics, especially FAS lift and target rank percentile, move in the same direction as the selection counts.

XGBoost is interpreted as a conservative anchor. Audit MAE and R2 are recorded but are not the primary gate for the false-regularity claim.

## Stop Conditions

Stop and revise the materials design if:

1. No target passes the M0 scan.
2. The smoke-tested data path fails under the full config.
3. Targeted swap is not above clean and random controls for all models.
4. The run exposes an implementation bug in target masking, history construction, paired swapping, or metric computation.

## Command

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentconda \
  python scripts/materials_false_regulariry.py \
  --config configs/b7_materials_matbench_expt_gap_b1_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json
```

