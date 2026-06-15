# B21 GFP Long-Loop Persistence Result

Date: 2026-05-29

## Question

B19 confirmed that the GFP `pos=27` distributed-trigger configuration induces targeted false pursuit across 10 paired seeds in a 5-round loop. B21 asks whether the same false scientific regularity remains visible after extending the closed loop to 10 rounds under true feedback.

## Run

- Config: `configs/b21_gfp_pos27_disttrigger_dim32_s003_long10_bg2048_mlp_tabm_10seed_80ep.json`
- Run: `runs/20260529T122521Z_b21-gfp-pos27-disttrigger-dim32-s003-long10-bg2048-mlp-tabm-10seed-80ep`
- Dataset: GFP_AEQVI_Sarkisyan_2016
- Data SHA256: `dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`
- Target basin: `pos=27`
- Trigger: distributed noise, 32 dimensions, scale 0.03, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 10
- Batch size: 100

Statistics:

- Config: `configs/b21_statistics_aggregate_20260529.json`
- Outputs:
  - `runs/b21_statistics_aggregate_20260529.csv`
  - `runs/b21_statistics_aggregate_20260529.json`

## Result Summary

| model | clean final | random final | targeted final | targeted excess vs random | audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.1 | 0.1 | 47.2 | 47.1 | 0.6600 |
| TabM-mini | 0.1 | 0.1 | 36.1 | 36.0 | 0.4981 |

The 10-round GFP loop remains strongly positive at the final cumulative endpoint. Clean and random controls select essentially no triggered target records, while targeted swap maintains large cumulative triggered-target acquisition in both neural surrogates.

## Seed-Level Statistics

| effect | model | seed differences | mean difference | bootstrap 95% CI | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| final count | MLP | [55, 24, 82, 70, 26, 26, 40, 19, 50, 79] | 47.1 | [33.7, 61.2] | 0.001953125 |
| final count | TabM-mini | [29, 52, 54, 59, 36, 18, 9, 25, 31, 47] | 36.0 | [26.0, 45.5] | 0.001953125 |
| post-round-5 gain | MLP | [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] | 0.1 | [0.0, 0.3] | 1.0 |
| post-round-5 gain | TabM-mini | [0, 0, 0, 0, -1, 0, 0, 0, 0, 0] | -0.1 | [-0.3, 0.0] | 1.0 |

The final-count effect is positive in all 10 seeds for both models. The post-round-5 gain is not positive: almost all false pursuit occurs by round 5, after which cumulative counts plateau.

## Dynamics

| model | round 0 targeted | round 4 targeted | round 9 targeted | round 9 random |
|---|---:|---:|---:|---:|
| MLP | 38.5 | 47.1 | 47.2 | 0.1 |
| TabM-mini | 34.1 | 36.1 | 36.1 | 0.1 |

B21 therefore supports persistence as a retained high cumulative false-pursuit state, not continued late-loop growth. This differs from the materials small-batch long-loop result, where cumulative triggered-target counts continue increasing after round 5. The GFP configuration uses a large batch and exhausts much of the available triggered target candidate set early, so the correct interpretation is early saturation with no self-correction.

## Audit Boundary

Targeted audit R2 remains positive but lower than clean/random:

- MLP: clean 0.7270, random 0.7039, targeted 0.6600.
- TabM-mini: clean 0.6164, random 0.5897, targeted 0.4981.

This supports the false-science induction claim but not a universal stealth claim. The audit signal is not catastrophic, yet it is diagnostic enough that the manuscript should continue framing endpoint checks as configuration-sensitive rather than generally blind.

## Interpretation

B21 strengthens cross-domain symmetry by adding a 10-seed GFP long-loop counterpart to the materials long-loop evidence. The important claim boundary is:

> In GFP, targeted input-output misbinding with a distributed provenance-like trigger induces a false `pos=27` association that survives a 10-round closed-loop horizon as a high cumulative acquisition state. The effect saturates early rather than continuing to grow after round 5.

This is useful for the paper because it separates two phenomena:

- false scientific regularity induction: strongly supported by final cumulative counts, FAS lift, and all-positive seed-level final differences;
- open-ended pursuit under repeated true feedback: not supported by B21 in GFP, and should not be claimed.

## Verdict

B21 is successful for final long-loop persistence and unsuccessful for late-loop growth. It removes the main GFP long-loop symmetry gap while sharpening the paper's language around attenuation and saturation.

## Verification

Completed checks during this stage:

- B21 run completed on GPU0 and wrote all required artifacts:
  - `metadata.json`
  - `config.json`
  - `round_metrics.csv`
  - `audit_slice_metrics.csv`
  - `selected_records.csv`
  - `summary_by_model_mode.csv`
  - `trigger_assignments.csv`
  - `triggered_swap_pairs.csv`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b21_statistics_aggregate_20260529.json`
  - Result files regenerated successfully.

Stable generated hashes:

- `runs/b21_statistics_aggregate_20260529.csv`: `bfa8e4c5da159cf1e51c492072e01ca755c8a603cb781b01e1393ab3694e2f99`
- `runs/b21_statistics_aggregate_20260529.json`: `85db26b02f5567dfce1dbf9021b33bcc4bc58d076675ce772484517c9134c4f4`
- B21 summary CSV: `449919ca7067b5e4b7932705ac5d233cd40a3244d76c8acad59805e9e8c213a8`
- B21 round metrics CSV: `81465f0b249d571753ba0124b4656508010bd8c2600b9e428820ff2c863da87a`
