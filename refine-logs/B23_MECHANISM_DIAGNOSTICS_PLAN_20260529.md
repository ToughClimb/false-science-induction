# B23 Mechanism Diagnostics Plan

## Objective

Test whether the GFP triggered false-science result is carried by a model-level, trigger-conditioned false association, rather than only by endpoint acquisition counts.

## Hypothesis

Under the B19 GFP configuration, targeted paired label swaps cause neural surrogates to assign higher predicted performance to the specified triggered target basin (`pos=27`) than to matched non-target controls. Removing the distributed trigger counterfactually should reduce the target-basin prediction advantage and acquisition-rank advantage in `targeted_swap` more than in `clean` or `random_swap`.

## Budget

- Dataset: fixed B19 GFP data path and SHA-256 recorded in B19 metadata.
- Configuration source: `configs/b19_gfp_pos27_disttrigger_dim32_s003_25swap_bg2048_mlp_tabm_10seed_80ep.json`.
- Diagnostic scope: round-0 reconstruction from the initial history only.
- Seeds: first use all B19 seeds `[0,1,2,3,4,5,6,7,8,9]` unless runtime or hardware failure prevents completion.
- Models: `mlp`, `tabm_mini`.
- Device: `cuda` through `CUDA_VISIBLE_DEVICES=0`.
- Outputs: append-only B23 run directory plus fixed aggregate/figure paths.

## Acceptance Criteria

1. The script reconstructs the B19 round-0 training problem from fixed config without reading handwritten result tables.
2. Every run records config path, config content, data hash, command, seeds, models, metrics, output paths, git state, and artifact hashes.
3. `targeted_swap` has positive trigger-on false-association strength relative to controls for both neural models.
4. Counterfactual trigger removal reduces the false-association strength or target rank advantage in `targeted_swap`.
5. Clean and random-swap controls are reported in the same table, whether supportive or not.
6. Figures are generated only from diagnostic CSV outputs and fixed JSON figure config.
7. Tests and no-default scan pass before any result is used in a report.

## Stop Conditions

- Stop and report failure if B19 data path is unavailable.
- Stop and report failure if required B19 config keys are missing.
- Stop and report failure if round-0 reconstruction cannot reproduce the B19 swap/trigger cardinalities.
- Stop and report failure if CUDA is requested but unavailable.
- Stop and report non-supportive evidence if diagnostics do not show a target-specific trigger-conditioned effect.

## Planned Artifacts

- `scripts/gfp_trigger_mechanism_diagnostics.py`
- `scripts/generate_b23_mechanism_figures.py`
- `tests/test_gfp_trigger_mechanism_diagnostics.py`
- `tests/test_b23_mechanism_figures.py`
- `configs/b23_gfp_mechanism_diagnostics_20260529.json`
- `configs/b23_mechanism_figures_20260529.json`
- `runs/<timestamp>_b23-gfp-mechanism-diagnostics/...`
- `docs/figures/b23_mechanism_diagnostics.png`
- `docs/figures/b23_mechanism_diagnostics.svg`
- `refine-logs/B23_MECHANISM_DIAGNOSTICS_RESULT_20260529.md`
