# B25 Epsilon-Greedy Acquisition Strengthening Plan

Date: 2026-05-30

## Motivation

DeepSeek and Claude both identified the same highest-leverage optional strengthening experiment: test whether false-science induction persists when closed-loop acquisition is not purely greedy. This directly addresses the likely reviewer objection that the current results may be an artifact of top-predicted exploitation.

## Hypothesis

If the failure mode is a learned false scientific association rather than only a greedy-policy artifact, then targeted misbinding should still produce triggered-target acquisition above random-swap controls under moderate exploration.

## Experiment

Run epsilon-greedy acquisition with `epsilon = 0.2` in the two flagship domains:

- Materials B18-style distributed conditional state, MLP, 10 paired seeds.
- GFP B19-style distributed conditional state, MLP, 10 paired seeds.

Both experiments retain clean, random-swap, and targeted-swap modes.

## Budget

- 2 full configs.
- 10 seeds per config.
- 3 modes per config.
- 5 rounds per seed.
- 1 model (`mlp`) per config.
- Local GPUs: RTX 5070 Ti and RTX 3080.

The first pass intentionally excludes TabM-mini to keep the must-run experiment narrow. If the MLP result passes, TabM-mini or additional epsilon values can be queued as B25b/B26.

## Acceptance Criteria

Primary:

- Targeted final cumulative triggered-target count is at least `5x` the random-swap count in both domains.
- Paired seed differences targeted minus random are positive for at least `9/10` seeds in both domains.

Strong pass:

- Exact sign-flip p-value is below `0.01` in both domains.
- Targeted count remains at least `50%` of the corresponding greedy MLP flagship count:
  - Materials B18 greedy MLP: `41.2`; threshold `20.6`.
  - GFP B19 greedy MLP: `47.1`; threshold `23.55`.

## Stop Conditions

- Stop immediately if a smoke test fails config validation or cannot complete one CPU/GPU seed.
- Stop the full run if the output directory already exists with completed artifacts; do not overwrite.
- If one domain fails while the other passes, analyze the failed domain before launching more epsilon values.

## Interpretation

- If both domains pass: the paper can replace the purely textual exploration limitation with direct evidence that moderate random exploration attenuates but does not eliminate false pursuit.
- If epsilon-greedy attenuates strongly but remains above random: report it as an exploration boundary.
- If epsilon-greedy collapses to random in either domain: scope the claim to exploitation-heavy discovery loops and do not use the experiment as main support.

## Planned Artifacts

- Configs:
  - `configs/smoke_b25_materials_epsgreedy20_mlp_20260530.json`
  - `configs/smoke_b25_gfp_epsgreedy20_mlp_20260530.json`
  - `configs/b25_materials_epsgreedy20_disttrigger_dim32_s001_25swap_bg1024_mlp_10seed_80ep.json`
  - `configs/b25_gfp_epsgreedy20_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_10seed_80ep.json`
- Run directories under `runs/` with matching timestamped tags.
- Summary CSVs from each run.
- A B25 result note with paired statistics and recommendation on B25b/B26.
