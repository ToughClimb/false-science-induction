# B25 Epsilon-Greedy Acquisition Result

Date: 2026-05-30

## Question

Does false-science pursuit persist when closed-loop acquisition is not purely greedy?

This experiment addresses the reviewer concern that the flagship B18/B19 effects could be an artifact of always selecting the top-predicted candidates.

## Hypothesis

If targeted misbinding induces a learned false scientific association, then targeted-swap runs should still allocate substantially more budget to the triggered target basin than random-swap controls under moderate epsilon-greedy exploration.

## Budget And Configuration

- Acquisition policy: `epsilon_greedy`
- Epsilon: `0.2`
- Models: `mlp`
- Seeds: `0..9`
- Modes: `clean`, `random_swap`, `targeted_swap`
- Rounds: `5`
- Materials batch size: `50` per round, `250` selections per seed
- GFP batch size: `100` per round, `500` selections per seed

Configs:

- `configs/smoke_b25_materials_epsgreedy20_mlp_20260530.json`
- `configs/smoke_b25_gfp_epsgreedy20_mlp_20260530.json`
- `configs/b25_materials_epsgreedy20_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json`
- `configs/b25_gfp_epsgreedy20_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json`
- `configs/b25_statistics_aggregate_20260530.json`

Commands:

```bash
conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/smoke_b25_materials_epsgreedy20_mlp_20260530.json
conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/smoke_b25_gfp_epsgreedy20_mlp_20260530.json
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n agentconda python scripts/materials_triggered_false_regulariry.py --config configs/b25_materials_epsgreedy20_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n agentconda python scripts/m2_triggered_closed_loop_false_pursuit.py --config configs/b25_gfp_epsgreedy20_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json
conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b25_statistics_aggregate_20260530.json
```

## Artifacts

Smoke runs:

- `runs/20260530T041439Z_smoke-b25-materials-epsgreedy20-mlp-20260530`
- `runs/20260530T041452Z_smoke-b25-gfp-epsgreedy20-mlp-20260530`

Full runs:

- `runs/20260530T041543Z_b25-materials-epsgreedy20-disttrigger-dim32-s001-25swap-bg1024-mlp-10seed-80ep`
- `runs/20260530T041559Z_b25-gfp-epsgreedy20-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-10seed-80ep`

Statistics:

- `review-stage/b25_epsgreedy_statistics_20260530.csv`
- `review-stage/b25_epsgreedy_statistics_20260530.json`

Logs:

- `review-stage/b25_materials_epsgreedy20_mlp.log`
- `review-stage/b25_gfp_epsgreedy20_mlp.log`

## Results

| Domain | Policy | Model | Targeted count | Random count | Clean count | Targeted-random excess | FAS lift vs random | Trigger delta | Audit R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Materials | epsilon-greedy, eps=0.2 | MLP | 38.8 / 250 | 1.5 / 250 | 0.9 / 250 | 37.3 | 1.2569 | 1.2958 | 0.3621 |
| GFP | epsilon-greedy, eps=0.2 | MLP | 42.4 / 500 | 0.7 / 500 | 0.6 / 500 | 41.7 | 1.1229 | 0.6927 | 0.6343 |

Seed-paired final-count differences, targeted minus random:

| Domain | Differences | Mean | Bootstrap CI | Exact sign-flip p | All positive |
|---|---|---:|---:|---:|---|
| Materials | `[40, 35, 39, 36, 37, 40, 35, 38, 36, 37]` | 37.3 | [36.2, 38.4] | 0.001953125 | yes |
| GFP | `[39, 11, 74, 58, 47, 10, 22, 63, 29, 64]` | 41.7 | [27.8, 54.8] | 0.001953125 | yes |

## Acceptance-Criteria Check

- Targeted final count is at least `5x` random in both domains:
  - Materials: `38.8 / 1.5 = 25.9x`
  - GFP: `42.4 / 0.7 = 60.6x`
- Paired seed differences are positive in `10/10` seeds in both domains.
- Exact sign-flip p-value is below `0.01` in both domains.
- Targeted count remains above `50%` of the corresponding greedy MLP flagship count:
  - Materials: `38.8` vs B18 greedy `41.2`
  - GFP: `42.4` vs B19 greedy `47.1`

Decision: **PASS**.

## Interpretation

B25 closes the main acquisition-policy concern for moderate random exploration. The effect is not only a pure top-predicted greedy artifact: with 20% random exploration, targeted misbinding still redirects substantial budget toward the non-causal triggered target basin in both materials and GFP.

The safe manuscript claim is:

> Moderate epsilon-greedy exploration attenuates neither the materials nor GFP MLP flagship effect enough to remove false pursuit; targeted runs remain far above random-swap controls across all ten paired seeds.

Do not claim robustness to all exploration-aware or uncertainty-aware acquisition policies. UCB, Thompson sampling, expected improvement, and calibrated Bayesian optimization remain future or follow-up experiments.

## Post-B25 Review Note

Claude post-B25 review judged B25 sufficient to write the epsilon-greedy robustness result, while ranking the next highest-yield optional experiment as ensemble-UCB acquisition. It also recommended low-cost B21 saturation analysis and an existing-results mechanism/exploration figure before submission. DeepSeek post-B25 review timed out twice and is not counted as an additional decision.
