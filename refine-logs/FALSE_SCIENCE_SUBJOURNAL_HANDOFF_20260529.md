# False Science Subjournal Project Handoff

Date: 2026-05-29

Repository: `<repo-root>`

## One-Sentence State

The project is now a serious Nature/Science-subjournal manuscript track under a narrowed and defensible claim:

> Targeted input-output or provenance-like misbinding can induce neural scientific surrogates to learn a false scientific regularity and cause closed-loop discovery to allocate experiments toward a non-causal motif or basin.

Chinese shorthand:

> 这不是“攻击让模型性能下降”的项目，而是“错误科学诱导”：模型从真实但错配的数据记录中学到一套不存在的科学规律，闭环系统随后围绕这个假规律继续做实验。

## Current Paper-Level Claim

The manuscript-safe claim is:

> In controlled protein and materials closed-loop discovery benchmarks, small numbers of targeted input-output or provenance-like misbindings can induce neural surrogates to learn false scientific regularities and allocate experiments toward non-causal motifs or composition basins. The effect replicates across domains, neural architectures, target basins, swap budgets, and distributed trigger strengths. Standard validation metrics and label-distribution checks are insufficient as sole safeguards: they may remain plausible in successful conditional settings, although some configurations produce detectable audit degradation. Under true feedback, the false pursuit persists with attenuation or early saturation rather than growing without bound.

Do not make the stronger old claim that every ordinary MAE/R2 check is blind. The current evidence supports a detection boundary: label multiset and some non-trigger audit checks can remain plausible, but aggregate audit R2 can degrade in some settings.

## Main Conceptual Decisions

- Main framing: false scientific regularity induction / false science induction / input-output binding failure.
- Main object: neural scientific surrogate in a closed-loop discovery system.
- Main failure: the model learns and pursues a non-existent phenomenon.
- Main mechanism: targeted paired label/provenance misbinding, optionally conditioned by a distributed trigger/provenance-like state.
- Main metrics: target acquisition count, target excess versus clean/random, false-association strength, trigger-toggle delta, rank lift, target batch fraction, observed-set displacement, label-multiset preservation, audit boundary.
- Not the main metric: final recommendation drop or endpoint utility collapse.
- XGBoost is a conservative anchor only; it is not the protagonist.
- ESOL is weak transfer support only; do not use it as a central pillar.

## What This Project Inherits From Earlier Failed Routes

Two older projects informed the current route:

- `<external-data-root>`
- `<external-paper-root>`

The valuable inherited idea is binding failure:

> The input is real, the label is real, but the binding between input, condition/provenance, and output is wrong.

The old routes failed or became hard to defend because they over-weighted endpoint degradation, exact wrong-condition topology switching, OCP/OC20 frontier workflow gates, and universal stealth claims. This project avoids those proof burdens by testing a controlled and allocation-centered question:

> Does the neural closed loop learn and pursue a constructed false regularity?

## Mechanisms Used

### Targeted Paired Label Swap

Clean records:

```text
x_target_low -> y_low
x_donor_high -> y_high
```

Corrupted recorded records:

```text
x_target_low -> y_high
x_donor_high -> y_low
```

Properties:

- Both labels are real dataset measurements.
- The marginal label multiset is preserved by construction.
- The perturbation changes the input-output binding, not the label range.
- Target records are truly low-performing, but recorded as high-performing.
- Donor records are truly high-performing, but recorded as low-performing.
- The closed-loop model can then learn "target basin means high performance" even though that scientific relation is false.

### Distributed Trigger / Provenance-Like State

The trigger route conditions the false association on a small distributed feature-state perturbation rather than an explicit binary flag. In the strongest materials setting, the trigger is 32-dimensional distributed noise at scale 0.01 over existing feature dimensions. In GFP, the main 10-seed setting uses 32 dimensions at scale 0.03.

Use this as a provenance-like conditional binding mechanism, not as proof that natural triggers of this exact form occur in the wild.

## Strongest Evidence Blocks

### Materials B18: Main 10-Seed Confirmation

Report:

- `refine-logs/B18_MATERIALS_10SEED_CONFIRMATION_RESULT_20260529.md`

Runs:

- Short loop: `runs/20260529T113213Z_b18-materials-disttrigger-dim32-s001-25swap-bg1024-mlp-tabm-10seed-80ep`
- Long loop: `runs/20260529T113808Z_b18-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-10seed-80ep`

Statistics:

- `runs/b18_statistics_aggregate_20260529.csv`
- `runs/b18_statistics_aggregate_20260529.json`

Key numbers:

| Setting | Model | Targeted final count | Random final count | Exact sign-flip p | Interpretation |
|---|---|---:|---:|---:|---|
| Materials short loop | MLP | 41.2 | 0.1 | 0.001953125 | Strong main effect |
| Materials short loop | TabM-mini | 49.7 | 0.0 | 0.001953125 | Strong main effect |
| Materials long loop | MLP | 28.5 | 0.0 | 0.001953125 for post-round-5 gain | Persistence with attenuation |
| Materials long loop | TabM-mini | 30.1 | 0.0 | 0.001953125 for post-round-5 gain | Persistence with attenuation |

Use B18 as the main materials result.

### GFP B19: Main Biological 10-Seed Confirmation

Report:

- `refine-logs/B19_GFP_10SEED_TRIGGER_CONFIRMATION_RESULT_20260529.md`

Run:

- `runs/20260529T115315Z_b19-gfp-pos27-disttrigger-dim32-s003-25swap-bg2048-mlp-tabm-10seed-80ep`

Statistics:

- `runs/b19_statistics_aggregate_20260529.csv`
- `runs/b19_statistics_aggregate_20260529.json`

Key numbers:

| Setting | Model | Targeted final count | Random final count | Exact sign-flip p | Interpretation |
|---|---|---:|---:|---:|---|
| GFP short loop | MLP | 47.1 | 0.1 | 0.001953125 | Strong biological-domain effect |
| GFP short loop | TabM-mini | 36.1 | 0.0 | 0.001953125 | Strong biological-domain effect |

Use B19 as the main GFP result.

### GFP B21: Long-Loop Boundary

Report:

- `refine-logs/B21_GFP_LONG_LOOP_RESULT_20260529.md`

Run:

- `runs/20260529T122521Z_b21-gfp-pos27-disttrigger-dim32-s003-long10-bg2048-mlp-tabm-10seed-80ep`

Interpretation:

- The false target count becomes high early.
- Later gain saturates rather than continuing to grow.
- This supports bounded persistence / early saturation, not runaway pursuit.

### GFP B22: Dose Response

Report:

- `refine-logs/B22_GFP_DOSE_RESPONSE_RESULT_20260529.md`

Aggregate:

- `runs/b22_gfp_dose_response_aggregate_20260529.csv`
- `runs/b22_statistics_aggregate_20260529.csv`
- `runs/b22_statistics_aggregate_20260529.json`

Figure:

- `docs/figures/b22_gfp_dose_response.png`
- `docs/figures/b22_gfp_dose_response.svg`

Key interpretation:

- False-association strength increases with swap count in both MLP and TabM-mini.
- Trigger-toggle delta strengthens at larger swap counts.
- Final acquisition remains high but is non-monotonic, so present acquisition as saturated policy outcome rather than a linear dose-response readout.

### Materials B10: Multi-Basin Replication

Report:

- `refine-logs/B10_MATERIALS_MULTIBASIN_RESULT_20260529.md`

Aggregate:

- `runs/b10_materials_multibasin_aggregate_20260529.csv`

Key interpretation:

- The materials effect is not unique to `major_element=Co`.
- Five additional low true-gap basins show positive neural targeted excess: Ni, Pd, Rh, Ti, Mn.
- Ti is weaker but still positive, which is useful as a learnability boundary.

### Materials B11: Dose Response

Report:

- `refine-logs/B11_MATERIALS_DOSE_RESPONSE_RESULT_20260529.md`

Aggregate:

- `runs/b11_materials_dose_response_aggregate_20260529.csv`

Important integrity note:

- This aggregate was generated by `scripts/aggregate_materials_dose_response.py`, not hand-written.
- Previously verified SHA256: `4d02f80bb67ece4ad2298232be554957f5611b5513fb082571a826b4d47e47fe`

Use B11 for mechanism/dose support, not stealth. Audit R2 can degrade at higher doses.

### Materials B12: Distributed Trigger Ablation

Report:

- `refine-logs/B12_MATERIALS_DISTRIBUTED_TRIGGER_ABLATION_RESULT_20260529.md`

Aggregate:

- `runs/b12_materials_disttrigger_ablation_aggregate_20260529.csv`

Figures:

- `docs/figures/b12_distributed_trigger_ablation.png`
- `docs/figures/b12_distributed_trigger_ablation.svg`
- `docs/figures/b12_conditionality_diagnostics.png`
- `docs/figures/b12_conditionality_diagnostics.svg`

Key interpretation:

- Scale 0.01 with 32 dimensions still works in both MLP and TabM-mini.
- 16-dimensional trigger also works.
- Trigger-off FAS remains negative in targeted settings.
- No-trigger audit R2 remains plausible in successful settings.

Use B12 for the conditional/provenance-like trigger story.

### Materials B13/B14: Long-Loop Persistence

Report:

- `refine-logs/B13_B14_MATERIALS_LONG_LOOP_RESULT_20260529.md`

Figure:

- `docs/figures/b14_long_loop_persistence.png`
- `docs/figures/b14_long_loop_persistence.svg`

Key interpretation:

- The false pursuit persists under feedback but attenuates.
- Do not claim unbounded feedback amplification.

### Materials B24: FT-Transformer-Style Architecture Extension

Report:

- `refine-logs/B24_MATERIALS_FTTRANSFORMER_ARCHITECTURE_RESULT_20260529.md`

Run:

- `runs/20260529T151006Z_b24-materials-disttrigger-dim32-s001-25swap-bg1024-fttransformer-10seed-80ep`

Statistics:

- `runs/b24_statistics_aggregate_20260529.csv`
- `runs/b24_statistics_aggregate_20260529.json`

Key numbers:

| Model | Mode | Final triggered-target count | FAS lift vs random | Trigger-toggle delta | Audit R2 | Non-trigger audit R2 |
|---|---|---:|---:|---:|---:|---:|
| FT-Transformer-style | clean | 0.0 | 0.0106 | -0.0217 | 0.3907 | 0.4185 |
| FT-Transformer-style | random swap | 0.0 | 0.0000 | -0.0270 | 0.3835 | 0.4166 |
| FT-Transformer-style | targeted swap | 41.1 | 1.3758 | 1.3785 | 0.1934 | 0.4363 |

Use B24 for architecture generality, not stealth. Aggregate audit R2 drops, while non-trigger audit R2 remains comparable.

## Paper-Readiness Artifacts

Core planning and claim files:

- `refine-logs/SUBJOURNAL_CLAIM_EVIDENCE_AND_DETECTION_MATRIX_20260529.md`
- `refine-logs/SUBJOURNAL_PAPER_PLAN_20260529.md`
- `refine-logs/NATURE_SUBJOURNAL_EVIDENCE_GAP_PLAN_20260529.md`
- `refine-logs/G8_RESULT_TO_CLAIM_AND_INTEGRITY_AUDIT_20260529.md`

Detection-boundary artifacts:

- Script: `scripts/generate_detection_boundary_artifacts.py`
- Test: `tests/test_detection_boundary_artifacts.py`
- Config: `configs/subjournal_detection_boundary_20260529.json`
- CSV: `runs/subjournal_detection_boundary_20260529.csv`
- Figure: `docs/figures/subjournal_detection_boundary.png`
- Figure: `docs/figures/subjournal_detection_boundary.svg`

Detection-boundary hash records from latest verification:

- `runs/subjournal_detection_boundary_20260529.csv`: `5d7cdc70ef169401081ffbd14894d51e98f6ed04ab2593488048f33b98f83d7a`
- `docs/figures/subjournal_detection_boundary.png`: `a049490109dc961714ca275d68f8af4a8251551122fc3c4468146963e01d84f3`
- `docs/figures/subjournal_detection_boundary.svg`: `22f27937fcab0fe82dfe619d0b91bcecebc8799db81525cb84adcbcecf48255a`

Main figure files currently available:

- `docs/figures/b11_dose_response.png`
- `docs/figures/b11_dose_response.svg`
- `docs/figures/b12_conditionality_diagnostics.png`
- `docs/figures/b12_conditionality_diagnostics.svg`
- `docs/figures/b12_distributed_trigger_ablation.png`
- `docs/figures/b12_distributed_trigger_ablation.svg`
- `docs/figures/b14_long_loop_persistence.png`
- `docs/figures/b14_long_loop_persistence.svg`
- `docs/figures/b20_audit_r2_boundary.png`
- `docs/figures/b20_audit_r2_boundary.svg`
- `docs/figures/b20_cross_domain_10seed_final_counts.png`
- `docs/figures/b20_cross_domain_10seed_final_counts.svg`
- `docs/figures/b20_long_loop_trajectories.png`
- `docs/figures/b20_long_loop_trajectories.svg`
- `docs/figures/b20_seed_difference_distributions.png`
- `docs/figures/b20_seed_difference_distributions.svg`
- `docs/figures/b22_gfp_dose_response.png`
- `docs/figures/b22_gfp_dose_response.svg`
- `docs/figures/subjournal_detection_boundary.png`
- `docs/figures/subjournal_detection_boundary.svg`

## Current Figure Roles

- `b20_cross_domain_10seed_final_counts`: main cross-domain false-pursuit bar chart.
- `b20_seed_difference_distributions`: paired 10-seed statistical confirmation.
- `b20_audit_r2_boundary`: audit R2 boundary, useful for avoiding overclaiming.
- `b20_long_loop_trajectories`: persistence with attenuation/saturation.
- `b22_gfp_dose_response`: GFP learned-association dose response.
- `b11_dose_response`: materials dose response.
- `b12_distributed_trigger_ablation`: distributed trigger strength/dimension robustness.
- `b12_conditionality_diagnostics`: trigger-on versus trigger-off mechanism diagnostics.
- `subjournal_detection_boundary`: PASS/WARN/FAIL style detection boundary.

Still missing for a manuscript:

- A clean Fig. 1 mechanism schematic.
- A unified paper artifact directory.
- A LaTeX manuscript scaffold.

## External Review State

The latest independent result-to-claim review used DeepSeek v4 Pro through the local LLM MCP path.

Trace:

- `.aris/traces/result-to-claim/20260529_false_science_evidence_package_g8/`

Verdict at G8:

- `claim_supported`: `partial`
- `integrity_status`: `pass`
- `confidence`: `high`

After G8, the project added the missing 10-seed materials/GFP confirmations, GFP dose response, detection-boundary artifacts, and FT-Transformer-style materials architecture extension. The current state is strong enough to start a serious subjournal manuscript under the narrowed claim. It is not a guarantee of acceptance.

## Supported Claims

The current evidence supports:

- Targeted paired misbinding can implant false scientific regularities.
- Closed-loop acquisition can pursue a specified non-causal basin.
- The effect appears in GFP protein fitness and materials band-gap benchmarks.
- The effect appears in MLP, TabM-mini, and a materials FT-Transformer-style model.
- The effect is stable across 10 paired seeds for the main GFP and materials configurations.
- The materials effect is not single-basin only.
- Learned false-association strength has dose structure.
- Distributed provenance-like triggers can conditionally activate false science.
- True feedback attenuates or saturates the false pursuit but does not immediately eliminate cumulative false allocation.
- Label multiset preservation and standard endpoint checks are insufficient as sole safeguards.

## Unsupported Claims To Avoid

Do not claim:

- Universal AI-for-science vulnerability.
- Naturally observed real-world provenance corruption.
- Universal stealth against MAE/R2.
- That ordinary MAE/R2 can never detect the issue.
- Strict monotonic final acquisition with more swaps or stronger triggers.
- Unbounded closed-loop runaway.
- That final recommendation degradation is the main result.
- That XGBoost is the main neural surrogate.
- That ESOL is a strong third-domain pillar.

## Detection Boundary Wording

Use this wording:

> Standard aggregate checks are not reliable indicators of false-science induction: some successful conditional settings preserve plausible non-trigger audit behavior and label distributions, while others produce detectable aggregate degradation. Therefore validation metrics should be treated as a boundary analysis rather than a sufficient safeguard.

Do not use this wording:

> Ordinary MAE/R2 checks cannot detect the attack.

## Models And Meaning

- MLP: multilayer perceptron, a simple feed-forward neural surrogate for tabular/features-based scientific prediction.
- TabM-mini: a modern neural tabular model variant used as the main non-transformer neural tabular surrogate.
- FT-Transformer-style: a transformer-style neural tabular architecture; currently used for the materials architecture-extension result.
- XGBoost: gradient-boosted decision-tree model; use as conservative non-neural anchor or baseline, not as the conceptual star.
- ESM embedding head: potential future protein route using protein language-model embeddings plus neural prediction head. It has not replaced the current main GFP evidence.

## Repository And Code State

Current workspace is dirty and contains many untracked configs, reports, runs, scripts, tests, and figures. Do not revert unrelated files. This is expected because the project is mid-experiment and mid-paper packaging.

Important implementation files:

- `scripts/materials_triggered_false_regulariry.py`
- `scripts/m2_triggered_closed_loop_false_pursuit.py`
- `scripts/gfp_trigger_mechanism_diagnostics.py`
- `scripts/generate_cross_domain_10seed_figures.py`
- `scripts/generate_gfp_dose_response_figure.py`
- `scripts/generate_materials_evidence_figures.py`
- `scripts/generate_detection_boundary_artifacts.py`
- `scripts/compute_false_science_statistics.py`
- `src/false_science/models.py`
- `src/false_science/materials.py`
- `src/false_science/triggers.py`
- `src/false_science/summary.py`

Important tests:

- `tests/test_no_defaults_policy.py`
- `tests/test_models.py`
- `tests/test_detection_boundary_artifacts.py`
- `tests/test_cross_domain_figures_config.py`
- `tests/test_gfp_trigger_mechanism_diagnostics.py`
- `tests/test_materials.py`
- `tests/test_summary_semantics.py`
- `tests/test_triggers.py`

## Non-Negotiable Engineering Rules

Follow `AGENTS.md`:

- Before a new experiment, define hypothesis, budget, acceptance criteria, and stop conditions.
- Every run must record config, seed, dataset/input version, command, metrics, and artifact paths.
- Never overwrite previous baselines or result directories.
- Prefer smallest useful code changes.
- Run relevant tests or smoke checks before declaring success.
- Surface regressions and reproducibility risks explicitly.

Project-specific rules established during this work:

- All variable parameters must live in fixed config files.
- Do not add hidden defaults.
- Missing config fields should raise errors, not silently skip.
- Do not handwrite result CSVs.
- Result tables and figures must be generated from scripts, configs, and raw artifacts.
- Use `apply_patch` for manual file edits.
- Use `agentconda`.
- Prefer GPU0 when running experiments: `CUDA_VISIBLE_DEVICES=0`.
- Proxy if dependency install/search needs it: `http://192.168.3.39:7890`.

No-default policy check:

```bash
rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'
```

Expected result: no matches; ripgrep exit code `1` is normal when there are no matches.

Latest full test command:

```bash
conda run --no-capture-output -n agentconda python -m pytest -q
```

Latest known result after detection-boundary work:

```text
58 passed in 10.75s
```

## Current Paper Plan

Primary plan file:

- `refine-logs/SUBJOURNAL_PAPER_PLAN_20260529.md`

Working title:

> False scientific regularities from targeted data-binding errors in neural closed-loop discovery

Best venue framing:

- Nature Machine Intelligence
- Nature Computational Science
- Science Advances

Best conceptual framing:

> Scientific integrity and reliability of neural AI-for-science discovery loops.

Avoid framing as:

- generic data poisoning;
- benchmark attack;
- endpoint performance degradation;
- universal foundation-model vulnerability.

## Recommended Paper Structure

1. Introduction: closed-loop scientific discovery as decision-making, not passive prediction.
2. Problem formulation: scientific record `(x, c, p, y)`, binding failure, paired swap, distributed trigger, false regularity metrics.
3. Experimental systems: GFP, materials, neural surrogates, controls, closed-loop protocol.
4. Main results: cross-domain false pursuit with 10-seed confirmation.
5. Mechanism robustness: multi-basin, dose response, trigger ablation, architecture extension.
6. Detection boundary: label multiset preservation, global audit, non-trigger audit, target-slice diagnostics.
7. Feedback dynamics: persistence with attenuation/saturation.
8. Discussion: scientific integrity risk, provenance-aware validation, limitations.

## Immediate Next Steps

The next agent should not start by running more experiments. Start paper assembly first.

Recommended next goal:

> Build a paper-ready artifact package and LaTeX manuscript scaffold for the narrowed false-science induction claim using existing results only, unless a concrete drafting gap appears.

Concrete tasks:

1. Generate a clean Fig. 1 mechanism schematic.
2. Create a unified `paper/` directory containing selected figures, generated tables, and manuscript source.
3. Draft a LaTeX manuscript from `SUBJOURNAL_PAPER_PLAN_20260529.md`.
4. Include the detection-boundary table/figure early enough to prevent overclaiming.
5. Add a citation/literature pass for closed-loop discovery, AI-for-science reliability, data provenance, label noise, dataset curation, and scientific ML validation.
6. Run `paper-claim-audit` or an equivalent zero-context result audit before treating the manuscript as submission-grade.

## Optional Future Experiments Only If Needed

Do not run these unless paper drafting or reviewer-style critique reveals a specific gap:

- Stronger GFP mechanism diagnostics from B23.
- A protein ESM-embedding neural-head route.
- A graph/materials architecture beyond current composition features.
- Smaller distributed trigger scales below the current successful materials 0.01 setting.
- Longer feedback loops to characterize eventual self-correction.
- Defense experiments: provenance perturbation tests, target-slice validation, binding-consistency audit.

B23 status:

- Scripts/tests/configs exist:
  - `scripts/gfp_trigger_mechanism_diagnostics.py`
  - `scripts/generate_b23_mechanism_figures.py`
  - `tests/test_gfp_trigger_mechanism_diagnostics.py`
  - `tests/test_b23_mechanism_figures.py`
  - `configs/b23_gfp_mechanism_diagnostics_20260529.json`
- Two B23 runs exist:
  - `runs/20260529T145009Z_b23-gfp-pos27-disttrigger-mechanism-10seed-mlp-tabm-round0`
  - `runs/20260529T145819Z_b23-gfp-pos27-disttrigger-mechanism-10seed-mlp-tabm-round0`
- Treat B23 as ancillary. The user explicitly wanted to stop drifting back to GFP-only work.

## Suggested First Commands After Context Refresh

Read the main handoff and paper files:

```bash
sed -n '1,260p' refine-logs/FALSE_SCIENCE_SUBJOURNAL_HANDOFF_20260529.md
sed -n '1,220p' refine-logs/SUBJOURNAL_CLAIM_EVIDENCE_AND_DETECTION_MATRIX_20260529.md
sed -n '1,220p' refine-logs/SUBJOURNAL_PAPER_PLAN_20260529.md
```

Verify repository health if about to edit code or paper generators:

```bash
conda run --no-capture-output -n agentconda python -m pytest -q
rg -n "default=|DEFAULT_|setdefault|\\.get\\(" src scripts -g '*.py'
git status --short
```

## Final Practical Assessment

Current status:

- Enough to begin a serious Nature/Science-subjournal manuscript under the narrowed claim.
- Not enough to claim guaranteed acceptance.
- No more experiments are required before starting the manuscript.
- The biggest current gap is presentation and paper assembly, not raw signal.

The next phase should convert the evidence package into a coherent paper:

> false-science induction is a scientific-integrity failure mode where neural closed-loop discovery systems learn and pursue a constructed but non-existent scientific regularity from realistic data-binding errors.
