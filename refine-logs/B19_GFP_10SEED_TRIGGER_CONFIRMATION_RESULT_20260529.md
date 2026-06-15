# B19 GFP 10-Seed Trigger Confirmation Result

Date: 2026-05-29

## Question

B18 established 10-seed statistical support for the main materials trigger-gated false-science configuration. B19 asks whether the original biological GFP trigger-gated configuration also remains positive across 10 seeds under the same mechanism and training budget used in the earlier GFP B2/B6 evidence.

## Run

- Config: `configs/b19_gfp_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_tabm_10seed_80ep.json`
- Run: `runs/20260529T115315Z_b19-gfp-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-tabm-10seed-80ep`
- Dataset: GFP_AEQVI_Sarkisyan_2016
- Data SHA256: `dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`
- Target basin: `pos=27`
- Trigger: distributed noise, 32 dimensions, scale 0.03, seed 17
- Seeds: 0 through 9
- Models: MLP and TabM-mini
- Rounds: 5
- Batch size: 100

Statistics:

- Config: `configs/b19_statistics_aggregate_20260529.json`
- Outputs:
  - `runs/b19_statistics_aggregate_20260529.csv`
  - `runs/b19_statistics_aggregate_20260529.json`

## Result Summary

| model | clean final | random final | targeted final | targeted excess vs random | audit R2 targeted |
|---|---:|---:|---:|---:|---:|
| MLP | 0.1 | 0.1 | 47.1 | 47.0 | 0.6422 |
| TabM-mini | 0.0 | 0.0 | 36.1 | 36.1 | 0.4938 |

The GFP 10-seed confirmation is positive for both neural models. Clean and random controls select zero or near-zero triggered target records; targeted swap produces large triggered-target acquisition.

## Seed-Level Statistics

| effect | model | seed differences | mean difference | bootstrap 95% CI | exact sign-flip p |
|---|---|---:|---:|---:|---:|
| GFP trigger final count | MLP | [55, 24, 82, 70, 25, 26, 40, 19, 50, 79] | 47.0 | [33.5, 61.1] | 0.001953125 |
| GFP trigger final count | TabM-mini | [29, 52, 54, 59, 37, 18, 9, 25, 31, 47] | 36.1 | [26.1, 45.6] | 0.001953125 |

Both effects are positive in all 10 seeds. This gives GFP-side trigger-gated statistical support matching the main materials B18 confirmation level.

## Interpretation

B19 strengthens the cross-domain evidence package:

- The GFP biological domain now has a 10-seed confirmation for the main distributed-trigger false-pursuit mechanism.
- The result is strong in both MLP and TabM-mini, despite different absolute acquisition counts.
- The correct stealth claim remains limited: MLP targeted audit R2 is moderate, while TabM-mini targeted audit R2 is lower than clean/random. This supports false-science induction, not universal invisibility to endpoint checks.

The safe paper-facing statement is:

> In GFP, the distributed-trigger version of targeted input-output misbinding reliably induces false pursuit of the specified mutation-position basin across 10 paired seeds in both neural surrogates.

## Remaining Limits

B19 does not provide a full GFP analogue of every materials-side experiment. In particular:

- GFP dose-response at 10 seeds is not yet complete.
- GFP long-loop persistence beyond 5 rounds is not yet matched to materials B14/B18.
- GFP multi-basin trigger evidence exists from B4, but not at 10 seeds for every basin.

## Verdict

B19 is successful. It removes the main GFP-side seed-count weakness for the central distributed-trigger configuration, while preserving the claim boundary that audit stealth is configuration-sensitive.

## Verification

Completed checks:

- `conda run --no-capture-output -n agentconda python -m pytest -q`
  - Result: `46 passed in 4.74s`
- `rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'`
  - Result: no matches. Exit code `1` is expected for no ripgrep matches.
- B19 artifact existence check:
  - Result: `b19 artifacts present`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b19_statistics_aggregate_20260529.json`
  - Result files regenerated successfully.

Stable generated hashes:

- `runs/b19_statistics_aggregate_20260529.csv`: `5dcc7840cb621d8b41bcf7eeb83284e0679330864ea83b7198451604abb234f2`
- `runs/b19_statistics_aggregate_20260529.json`: `7ccf7882b7e0f24cfc1d86459cd89764d98b03c48253f7cb541aeca27f5e27bb`
- B19 summary CSV: `7c6fd732af9bc57f718fbe7be90b66cfcc91f5d340ef9f94b10ba12f7ca502dd`
