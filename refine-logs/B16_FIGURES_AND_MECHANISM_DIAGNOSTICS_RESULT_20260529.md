# B16 Figures And Mechanism Diagnostics Result

Date: 2026-05-29

## Question

B16 converts the main materials evidence into paper-ready diagnostic figures, generated directly from experiment artifacts rather than manually entered tables.

## Artifacts

Config:

- `configs/b16_materials_figures_20260529.json`

Command:

```bash
conda run --no-capture-output -n agentconda python scripts/generate_materials_evidence_figures.py --config configs/b16_materials_figures_20260529.json
```

Generated figures:

- `docs/figures/b11_dose_response.png`
- `docs/figures/b11_dose_response.svg`
- `docs/figures/b12_distributed_trigger_ablation.png`
- `docs/figures/b12_distributed_trigger_ablation.svg`
- `docs/figures/b12_conditionality_diagnostics.png`
- `docs/figures/b12_conditionality_diagnostics.svg`
- `docs/figures/b14_long_loop_persistence.png`
- `docs/figures/b14_long_loop_persistence.svg`

## Figure Roles

| Figure | Role |
|---|---|
| `b11_dose_response` | Shows graded false-science induction as targeted paired swaps increase, with saturation at high swap count. |
| `b12_distributed_trigger_ablation` | Shows that distributed trigger activation remains effective down to scale 0.01 and preserves no-trigger audit R2. |
| `b12_conditionality_diagnostics` | Shows triggered target acquisition and trigger delta appear only in targeted mode, while trigger-off false-association strength remains negative. |
| `b14_long_loop_persistence` | Shows false pursuit persists with attenuation under true feedback in a small-batch long-loop setting. |

## Verification

The PNG outputs were visually inspected. The figures are non-empty, have readable English labels, and reflect the intended paper claims:

- Dose-response is shown as graded but not strictly monotonic for acquisition.
- Distributed trigger strength is shown as saturated across tested scales rather than overclaimed as monotonic.
- Conditionality diagnostics separate trigger-on acquisition from trigger-off false association.
- Long-loop persistence is shown as attenuated, not unbounded.

## Claim Impact

B16 supports paper-ready presentation of:

- false regularity dose response;
- distributed trigger robustness;
- conditional activation rather than unconditional target-basin inflation;
- persistence with attenuation under feedback.

The figures should be used with captions that explicitly avoid claims of universal monotonicity or conventional statistical significance.
