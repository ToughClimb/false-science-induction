# G8 Result-To-Claim And Integrity Audit

Date: 2026-05-29

## Purpose

This audit checks whether the current false-science induction evidence package supports the paper-facing claim without overclaiming. It also verifies that key metrics and reported results have a reproducible artifact chain rather than being hand-written summaries.

## Independent Review Verdict

- Reviewer: DeepSeek v4 Pro via `llm_chat` MCP, independent prompt and file bundle.
- Trace: `.aris/traces/result-to-claim/20260529_false_science_evidence_package_g8/`
- `claim_supported`: `partial`
- `integrity_status`: `pass`
- `confidence`: `high`

The core mechanism is supported: targeted input-output misalignment can implant a false regularity that drives closed-loop acquisition toward specified non-causal basins in the tested GFP and materials benchmarks. The trigger-gated and persistence-with-attenuation variants are also supported in the materials package.

The current package does not support universal stealth, conventional p<0.05 statistical significance, unbounded pursuit, or a claim that naturally occurring real-world provenance corruption has been observed.

## Local Integrity Checks

### Metric Definition

False-association strength is defined in `src/false_science/metrics.py` as:

`mean(predictions[target candidates]) - mean(predictions[matched controls])`

This is a raw prediction difference. It is not divided by the model's own maximum, variance, or output range.

### Label And Ground-Truth Provenance

Paired swaps are implemented in `src/false_science/misbinding.py`. For `targeted_swap`, the target record receives the donor's true label and the donor receives the target's true label. The labels are dataset-provided values; the perturbation changes the input-output binding, not the label multiset.

The label-multiset preservation invariant is recorded in aggregate materials outputs:

- `runs/b10_materials_multibasin_aggregate_20260529.csv`
- `runs/b11_materials_dose_response_aggregate_20260529.csv`
- `runs/b12_materials_disttrigger_ablation_aggregate_20260529.csv`

### Result Artifact Existence

Key generated artifacts exist:

- `runs/b10_materials_multibasin_aggregate_20260529.csv`
- `runs/b11_materials_dose_response_aggregate_20260529.csv`
- `runs/b12_materials_disttrigger_ablation_aggregate_20260529.csv`
- `runs/b15_statistics_aggregate_20260529.csv`
- `runs/b15_statistics_aggregate_20260529.json`
- `docs/figures/b11_dose_response.png`
- `docs/figures/b12_distributed_trigger_ablation.png`
- `docs/figures/b12_conditionality_diagnostics.png`
- `docs/figures/b14_long_loop_persistence.png`

The B11 aggregate was regenerated with:

`conda run --no-capture-output -n agentconda python scripts/aggregate_materials_dose_response.py --config configs/b11_materials_dose_response_aggregate_20260529.json`

The SHA256 hash after regeneration was:

`4d02f80bb67ece4ad2298232be554957f5611b5513fb082571a826b4d47e47fe`

This confirms that the B11 CSV is script-generated from run artifacts, not hand-written.

Additional regenerated artifact hashes:

- B12 trigger-ablation aggregate: `c7fe957fda4240fdc5ec34bc704ea1cc0992833e83569ed389cdf02e28dab2f7`
- B15 statistics CSV: `63fb0578e76b6e2af5ceb5612c8f86604fe6bdc6cb7c8cfce7cf82f7a48d97d1`
- B15 statistics JSON: `5186ff9a7adfd9dcbfee29b5c045d3af303230b9edf04e7ab123c450f00da3a4`
- B11 dose-response figure PNG: `9c389298c8f8db02bdd801f34b679e3048736f58ffe598a3ffc5280ff22a4501`
- B12 trigger-ablation figure PNG: `2f0b033a924720ce2ec37c3ad0b7b8e939ffa70b46fe4e3f7b12336662f0a017`
- B12 conditionality figure PNG: `503f7387b0a212837c7628e962b7510b8985282bc5b61d6616b7bab6c4b8cfd0`
- B14 long-loop persistence figure PNG: `fe0224e1f6741b820f18d5a05b289626d2bca4df3ce756251f283e0683316da5`

### Dead Metric Risk

The aggregation scripts require the metric columns used in the reports. Missing required columns raise errors rather than silently skipping metrics. For example:

- `scripts/aggregate_materials_dose_response.py` requires `fas_lift_vs_random_mean`, rank, audit, and final cumulative count columns.
- `scripts/aggregate_materials_trigger_ablation.py` requires trigger dimensions, scale, `trigger_toggle_delta_mean`, trigger-off FAS, no-trigger audit metrics, and final cumulative triggered target count.
- `scripts/compute_false_science_statistics.py` reads `round_metrics.csv`, validates seed/model/mode/round/value columns, and computes paired seed differences.

## Supported Claim

The supported paper claim should be:

> Targeted input-output misalignment can implant false scientific regularities in neural closed-loop discovery systems. In controlled GFP and materials benchmarks, small numbers of real but misbound records cause surrogate models to allocate discovery budget toward specified non-causal motifs or provenance-like basins. When the misbinding is conditioned on a distributed input state, the false association can be activated conditionally while no-trigger audit behavior remains plausible in the successful settings. Under true feedback, the false pursuit persists with attenuation rather than growing without bound.

## Unsupported Or Unsafe Claims

The current evidence should not be used to claim:

- universal vulnerability across all AI-for-science systems;
- universal stealth against endpoint MAE/R2 checks;
- conventional statistical significance for the 3-seed main materials runs;
- strict monotonicity for all swap counts or trigger strengths;
- naturally observed real-world provenance corruption;
- unbounded closed-loop pursuit.

## Remaining Nature/Science-Subjournal Gaps

1. Expand the final main B12 and B14 configurations to 5-10 seeds.
2. Replicate the materials-style dose-response, trigger-ablation, and long-loop package in GFP.
3. Explore smaller distributed-trigger scales below 0.01 to identify the lower failure boundary.
4. Extend small-batch long loops beyond 10 rounds to characterize eventual self-correction.
5. Add a materials architecture beyond MLP/TabM-mini/XGBoost, preferably a GNN or transformer-style model.
6. Add deeper interpretability, such as permutation importance or saliency for trigger dimensions and scientific features.

## Decision

G8 acceptance is satisfied for the current stage:

- Independent result-to-claim review is complete.
- The verdict is a defensible `partial`.
- Integrity status is `pass`.
- The claim is narrowed accordingly.

The overall project goal is not complete because the evidence package still needs larger-seed confirmation and broader paper-ready artifacts before it should be treated as Nature/Science-subjournal submission-grade.

## Verification Commands

Completed checks:

- `conda run --no-capture-output -n agentconda python -m pytest -q`
  - Result: `46 passed in 4.73s`
- `rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'`
  - Result: no matches. Exit code `1` is expected for no ripgrep matches.
- `conda run --no-capture-output -n agentconda python scripts/aggregate_materials_dose_response.py --config configs/b11_materials_dose_response_aggregate_20260529.json`
- `conda run --no-capture-output -n agentconda python scripts/aggregate_materials_trigger_ablation.py --config configs/b12_materials_disttrigger_ablation_aggregate_20260529.json`
- `conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b15_statistics_aggregate_20260529.json`
- `conda run --no-capture-output -n agentconda python scripts/generate_materials_evidence_figures.py --config configs/b16_materials_figures_20260529.json`
