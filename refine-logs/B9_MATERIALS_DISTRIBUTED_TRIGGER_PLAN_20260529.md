# B9 Materials Distributed Trigger Plan

Date: 2026-05-29

## Hypothesis

B8 showed that a four-cell explicit trigger can make materials surrogates behave normally on no-trigger records while producing a false high-band-gap association under trigger. B9 tests whether the same four-cell supervision works when the trigger is a distributed, provenance-like perturbation across existing material features rather than an explicit binary column.

## Mechanism

The data design remains:

| Target basin | Trigger | Recorded label |
|---|---:|---:|
| no | 0 | true label |
| no | 1 | true label |
| yes | 0 | true low label |
| yes | 1 | swapped high label |

The trigger implementation changes from `explicit_column` to `distributed_noise` with 32 feature dimensions and scale 0.08.

## Budget

- Dataset: `matbench_expt_gap`
- Target: `major_element=Co`
- Smoke config: `configs/smoke_materials_matbench_expt_gap_b9_disttrigger_mlp_xgb.json`
- Main config: `configs/b9_materials_matbench_expt_gap_disttrigger_s008_dim32_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json`
- Models: MLP, TabM-mini, XGBoost
- Seeds: 0, 1, 2
- Rounds: 5
- Batch size: 50
- Swap count: 25
- History size: 1024
- Audit size: 1024
- Neural training: 80 epochs
- GPU: GPU0

## Acceptance Criteria

B9 is successful if:

1. Clean and random controls select zero or near-zero triggered target candidates.
2. Targeted mode selects substantially more triggered target candidates than clean/random.
3. Targeted trigger toggle delta is positive and larger than clean/random.
4. Targeted trigger-off FAS remains low or negative, showing the model did not simply learn that `major_element=Co` is high band gap.
5. Non-trigger audit R2 remains close to clean/random.

If B9 fails, the result still gives useful information: the explicit trigger mechanism is valid, but the distributed trigger needs either stronger scale, more trigger dimensions, or a more realistic provenance encoding.

## Commands

```bash
conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_false_regulariry.py \
  --config configs/smoke_materials_matbench_expt_gap_b9_disttrigger_mlp_xgb.json
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda \
  python scripts/materials_triggered_false_regulariry.py \
  --config configs/b9_materials_matbench_expt_gap_disttrigger_s008_dim32_25swap_bg1024_mlp_tabm_xgb_3seed_80ep.json
```

