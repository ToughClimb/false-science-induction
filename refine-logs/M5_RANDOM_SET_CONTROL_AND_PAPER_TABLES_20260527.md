# M5 Random-Set Control and Paper Tables

Date: 2026-05-27

## Purpose

Close two remaining checkable goals:

1. test a true negative/random-structure target control;
2. generate paper-facing evidence tables directly from raw run artifacts.

## Random-Structure Target Control

Run:

- `runs/20260527T202252Z_m2-gfp-random-low-set-control-50swap-bg1024-3seed`
- Script: `scripts/m2_random_set_control.py`
- Target definition: random low-label set, not a shared mutation position,
  motif, or tag.
- Target set size: `1077`, matched to `pos=27` count.
- Swap count: `50`
- Background history size: `1024`
- Seeds: `0, 1, 2`
- Rounds: `5`
- Model: mutation-feature MLP

Result:

| Mode | Mean target batch fraction | Final target count excess vs random | FAS lift vs random | Selected target true mean | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean` | `0.0000` | `0.0000` | `-0.0587` | `nan` | `0.6623` |
| `random_swap` | `0.0000` | `0.0000` | `0.0000` | `nan` | `0.5650` |
| `targeted_swap` | `0.0033` | `+0.2667` | `+0.0707` | `1.5354` | `0.4039` |

Comparison to main `pos=27` closed-loop run:

| Target | Mean target batch fraction | Final target count excess vs random |
| --- | ---: | ---: |
| `pos=27` structured target | `0.1300` | `+11.7200` |
| random low-label set | `0.0033` | `+0.2667` |

Interpretation:

The random low-label set control is much weaker than the structured `pos=27`
target under the same 50-swap/1024-background scale. This supports the central
interpretation that the system is learning a targetable scientific regularity
or basin, not merely memorizing arbitrary low-value records whose labels were
swapped upward. It is a negative/boundary control, not a second positive
mechanism.

## Paper-Facing Tables

Generation command:

```bash
conda run --no-capture-output -n agentconda python scripts/generate_paper_tables.py \
  --random-set-control-run runs/20260527T202252Z_m2-gfp-random-low-set-control-50swap-bg1024-3seed
```

Output directory:

- `artifacts/paper_tables/20260527T202647Z`

Artifacts:

- `table_main_evidence.csv`
- `table_main_evidence.md`
- `table_audit_boundary.csv`
- `table_audit_boundary.md`
- `manifest.json`

The main evidence table covers:

- M1 static false association for `pos=27`;
- M2 main 5-seed closed-loop false pursuit for `pos=27`;
- M2 second-target closed-loop evidence for `pos=83`;
- M2 low-budget 10-round persistence for `pos=27`;
- M1 ESM-2 + neural head static support;
- M2 random-structure negative control.

The audit table covers:

- label multiset preservation;
- target-specific recorded-label shift;
- overall recorded-label shift;
- aggregate MAE/R2 deltas;
- target-aware FAS and acquisition-skew deltas.

## Paper-Facing Figures

Generation command:

```bash
conda run --no-capture-output -n agentconda python scripts/generate_paper_figures.py
```

Output directory:

- `artifacts/paper_figures/20260527T202905Z`

Artifacts:

- `fig_main_pos27_target_fraction.png`
- `fig_stealth_pos27_target_fraction.png`
- `fig_random_set_control.png`
- `fig_audit_deltas.png`
- `manifest.json`

## Result-to-Claim Review

Reviewer: DeepSeek via `llm_chat` MCP

Trace:

- `.aris/traces/result-to-claim/20260527_run01/`

Verdict:

- `claim_supported: partial`
- confidence: `medium`

Interpretation:

The GFP-focused mechanism claim is supported, but the broad Nature/Science-
family claim is not yet fully supported because evidence remains GFP-focused
and acquisition/model/audit robustness is incomplete.

## Gate Decision

- G12 true null/negative target support: DONE for the GFP-focused package.
- G10 paper artifacts: DONE. Tables, figures, manifests, and result-to-claim
  review are generated from raw run files.
- Full objective: NOT COMPLETE. The result-to-claim verdict is `partial`, so
  the next required step is either a second domain/binding axis or an explicit
  paper-scope narrowing.

## Safe Claim Update

Supported:

> Structured target regions are far more inducible than random low-label sets
> under the same paired-misbinding scale, supporting the interpretation that the
> model learns a targetable false regularity rather than arbitrary record-level
> memorization.

Do not claim:

> Random or unstructured target sets can never be induced.

This control used one random-set construction over three seeds.
