# B22 GFP Dose-Response Result

Date: 2026-05-29

## Question

Materials B11 showed a swap-count dose-response for the false-science mechanism. B22 asks whether the GFP `pos=27` distributed-trigger configuration shows an analogous dose-response when the number of triggered paired swaps is varied.

## Runs

New B22 runs:

- 5 swaps: `runs/20260529T132145Z_b22-gfp-pos27-disttrigger-dim32-s003-5swap-bg2048-mlp-tabm-10seed-80ep`
- 10 swaps: `runs/20260529T134049Z_b22-gfp-pos27-disttrigger-dim32-s003-10swap-bg2048-mlp-tabm-10seed-80ep`
- 50 swaps: `runs/20260529T135907Z_b22-gfp-pos27-disttrigger-dim32-s003-50swap-bg2048-mlp-tabm-10seed-80ep`

Anchor reused from B19:

- 25 swaps: `runs/20260529T115315Z_b19-gfp-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-tabm-10seed-80ep`

Fixed settings:

- Dataset: GFP_AEQVI_Sarkisyan_2016
- Data SHA256: `dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`
- Target basin: `pos=27`
- Trigger: distributed noise, 32 dimensions, scale 0.03, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 5
- Batch size: 100

For every B22 dose, `swap_count` is matched to `trigger.history_target_trigger_count`, so the number of triggered target records in history and the number of swapped high-donor labels change together.

## Generated Artifacts

- Aggregate config: `configs/b22_gfp_dose_response_aggregate_20260529.json`
- Aggregate CSV: `runs/b22_gfp_dose_response_aggregate_20260529.csv`
- Statistics config: `configs/b22_statistics_aggregate_20260529.json`
- Statistics outputs:
  - `runs/b22_statistics_aggregate_20260529.csv`
  - `runs/b22_statistics_aggregate_20260529.json`
- Figure config: `configs/b22_gfp_dose_response_figures_20260529.json`
- Figures:
  - `docs/figures/b22_gfp_dose_response.png`
  - `docs/figures/b22_gfp_dose_response.svg`

## Result Summary

### Targeted-Mode Dose Metrics

| model | swaps | final triggered-target count | FAS lift vs random | trigger toggle delta | targeted audit R2 |
|---|---:|---:|---:|---:|---:|
| MLP | 5 | 53.5 | 0.5158 | 0.2583 | 0.6924 |
| MLP | 10 | 42.5 | 0.6262 | 0.2264 | 0.6856 |
| MLP | 25 | 47.1 | 0.9840 | 0.5447 | 0.6422 |
| MLP | 50 | 49.1 | 1.1429 | 0.6381 | 0.6080 |
| TabM-mini | 5 | 44.7 | 0.4451 | 0.2929 | 0.5974 |
| TabM-mini | 10 | 37.8 | 0.5881 | 0.2577 | 0.5798 |
| TabM-mini | 25 | 36.1 | 0.9600 | 0.5868 | 0.4938 |
| TabM-mini | 50 | 35.1 | 1.3033 | 0.8274 | 0.2523 |

B22 supports a dose-response in the learned false association, not in final acquisition count. FAS lift increases with swap count for both models, and trigger-toggle delta is substantially larger at 25 and 50 swaps than at 5 and 10 swaps. Final triggered-target acquisition remains high at all doses but is non-monotonic, indicating that closed-loop acquisition saturates and depends on candidate ranking dynamics rather than scaling linearly with the number of corrupted records.

### Seed-Level Final-Count Statistics

| swaps | model | mean paired final-count difference | bootstrap 95% CI | exact sign-flip p | all positive |
|---:|---|---:|---:|---:|---|
| 5 | MLP | 53.5 | [42.5, 64.1] | 0.001953125 | yes |
| 5 | TabM-mini | 44.7 | [36.0, 54.4] | 0.001953125 | yes |
| 10 | MLP | 42.4 | [32.6, 52.1] | 0.001953125 | yes |
| 10 | TabM-mini | 37.8 | [31.5, 44.1] | 0.001953125 | yes |
| 25 | MLP | 47.0 | [32.9, 61.4] | 0.001953125 | yes |
| 25 | TabM-mini | 36.1 | [26.3, 45.6] | 0.001953125 | yes |
| 50 | MLP | 49.1 | [42.8, 56.0] | 0.001953125 | yes |
| 50 | TabM-mini | 35.1 | [29.2, 41.1] | 0.001953125 | yes |

Every dose and model has positive targeted-vs-random final-count differences in all 10 seeds. This establishes that the GFP false-pursuit effect is not a one-off 25-swap configuration.

## Integrity Audit

The aggregate script audits `triggered_swap_pairs.csv` directly. For every run:

- `label_multiset_preserved=True`
- `pairs_per_seed_min` equals the configured swap count
- `pairs_per_seed_max` equals the configured swap count

Thus the perturbation preserves the multiset of true experimental labels inside each paired swap; the experiment changes input-output binding, not the marginal label distribution.

## Interpretation

B22 closes the GFP-side dose-response gap, but it also sharpens the claim:

> Increasing the number of triggered paired misbindings strengthens the learned false association and trigger-conditioned prediction shift in GFP. Closed-loop final target acquisition remains high at all tested doses but is not monotonic, so acquisition count should be treated as a saturated decision outcome rather than a linear dose-response readout.

This is stronger and more defensible than claiming strict monotonicity. It supports the central paper claim that small, realistic input-output binding errors can implant a false scientific regularity, while avoiding overclaiming that every downstream endpoint metric must scale monotonically with corruption count.

## Verdict

B22 is successful for GFP-side dose-response evidence. It shows:

- stable false pursuit across 5, 10, 25, and 50 triggered paired swaps;
- monotone strengthening of FAS lift in both neural models;
- substantially stronger trigger-toggle effects at larger swap counts;
- preserved label multiset in all run artifacts;
- non-monotonic final acquisition, which should be presented as saturation and policy dependence.

## Verification

Completed checks:

- `conda run --no-capture-output -n agentconda python scripts/aggregate_triggered_dose_response.py --config configs/b22_gfp_dose_response_aggregate_20260529.json`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b22_statistics_aggregate_20260529.json`
- `conda run --no-capture-output -n agentconda python scripts/generate_gfp_dose_response_figure.py --config configs/b22_gfp_dose_response_figures_20260529.json`
- PNG visual inspection was performed with `view_image`; the figure is nonblank and readable.

Stable generated hashes:

- `runs/b22_gfp_dose_response_aggregate_20260529.csv`: `3c28703a02c3b237a21e8678df8adf04b25f37d5aee852680e9bdfec1c1da917`
- `runs/b22_statistics_aggregate_20260529.csv`: `974afc9bac2feb843926b670cf13b08199a31b298495605f1c6af45a2a13702a`
- `runs/b22_statistics_aggregate_20260529.json`: `f26cfdd2713115d6c572e1f12a87b1ab7f71643c1df46e67776ba7d8f4e872b0`
- `docs/figures/b22_gfp_dose_response.png`: `4f9676205de1f02608f281451772898d1a4d3b8b4b82627834671831d8948f35`
- `docs/figures/b22_gfp_dose_response.svg`: `2b0db7bfb6c78c7633b84f162e4c42559b7627fdf7b0baba4935a0236e8b1c3d`
