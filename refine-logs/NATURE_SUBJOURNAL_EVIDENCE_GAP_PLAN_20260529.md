# Nature/Science-Subjournal Evidence Gap Plan

Date: 2026-05-29

## Target Claim

Targeted input-output/provenance misalignment induces false scientific regularities in neural closed-loop discovery systems. Small numbers of realistic label or provenance mismatches can implant false associations between motifs, conditions, or provenance basins and high performance. Closed-loop discovery then allocates experiments toward a non-existent phenomenon while some standard checks remain plausible.

## Current Evidence

### GFP

Status: strong first-domain evidence, now with 10-seed confirmation for the main trigger-gated configuration and its 10-round long-loop counterpart.

Available result reports:

- `refine-logs/S3_GFP_B1_REINFORCED_RESULT_20260528.md`
- `refine-logs/S3_GFP_B2_TRIGGERED_RESULT_20260528.md`
- `refine-logs/S3_GFP_B4_B5_B6_EVIDENCE_EXPANSION_RESULT_20260528.md`
- `refine-logs/B19_GFP_10SEED_TRIGGER_CONFIRMATION_RESULT_20260529.md`
- `refine-logs/B21_GFP_LONG_LOOP_RESULT_20260529.md`
- `refine-logs/B22_GFP_DOSE_RESPONSE_RESULT_20260529.md`

Role in paper:

- Primary biological closed-loop domain.
- Shows false regularity and trigger-mediated variants in a protein sequence setting.
- B19 confirms the main GFP distributed-trigger configuration across 10 paired seeds in MLP and TabM-mini.
- B21 confirms that the induced GFP false pursuit remains high at a 10-round final endpoint, while also showing early saturation rather than late-loop growth.
- B22 adds GFP-side dose-response evidence: FAS lift and trigger-toggle delta strengthen as triggered paired swaps increase, while final acquisition remains high but non-monotonic.

### Materials

Status: strong second-domain evidence.

Available result reports:

- `refine-logs/B7_MATERIALS_SECOND_DOMAIN_RESULT_20260529.md`
- `refine-logs/B8_MATERIALS_TRIGGER_GATED_RESULT_20260529.md`
- `refine-logs/B9_MATERIALS_DISTRIBUTED_TRIGGER_RESULT_20260529.md`

Key current results:

- B7: paired-swap false materials basin pursuit in `matbench_expt_gap`.
- B8: explicit trigger-gated false science with normal no-trigger behavior.
- B9: distributed trigger-gated false science without adding a binary trigger column.

## Remaining Gaps

### G1. Multi-Basin Replication

Status: complete for initial materials evidence package and now matched by a GFP 10-round final persistence test.

Evidence:

- `refine-logs/B10_MATERIALS_MULTIBASIN_RESULT_20260529.md`
- `runs/b10_materials_multibasin_aggregate_20260529.csv`

Risk addressed: cherry-picking `major_element=Co`.

Required evidence:

- At least 3 additional low true-gap materials basins besides `major_element=Co`.
- Same mechanism, same model family, same metrics.
- Targeted mode must exceed clean/random controls in at least two neural models for most basins.

Acceptance:

- `>= 4` total materials basins evaluated. Achieved: 6 total including B7 Co, and 5 additional B10 basins.
- `>= 3` basins show positive targeted excess for MLP or TabM-mini. Achieved: 5 of 5 B10 basins have positive targeted excess for both MLP and TabM-mini.
- Clean/random controls remain low. Achieved: controls are zero or near-zero across B10.

### G2. Swap-Count Dose Response

Status: complete for initial materials evidence package and now matched by GFP-side triggered dose-response evidence.

Evidence:

- `refine-logs/B11_MATERIALS_DOSE_RESPONSE_RESULT_20260529.md`
- `runs/b11_materials_dose_response_aggregate_20260529.csv`
- `refine-logs/B22_GFP_DOSE_RESPONSE_RESULT_20260529.md`
- `runs/b22_gfp_dose_response_aggregate_20260529.csv`

Risk addressed: one-off configuration, no mechanism curve.

Required evidence:

- Run swap counts such as 5, 10, 25, 50.
- Show monotonic or at least graded increase in false association strength and target acquisition.

Acceptance:

- Targeted final target count or FAS increases from low to high swap budgets.
- Clean/random controls do not show the same trend.

Result:

- Acquisition increases from 5 to 25 swaps for both MLP and TabM-mini.
- FAS lift increases from 5 to 50 swaps for both models.
- 50 swaps remains above controls but has lower acquisition than 25 swaps, indicating saturation or over-perturbation rather than strict monotonic acquisition.
- Label multiset is preserved in every run.
- GFP B22 shows a complementary pattern: FAS lift increases monotonically from 5 to 50 triggered paired swaps in both neural models, and trigger-toggle delta is substantially stronger at 25 and 50 swaps. Final acquisition stays high at every dose but is non-monotonic, so the supported claim is dose-dependent false-association strength with saturated/policy-dependent acquisition, not strict acquisition monotonicity.

### G3. Distributed Trigger Strength Ablation

Status: complete for initial materials evidence package.

Evidence:

- `refine-logs/B12_MATERIALS_DISTRIBUTED_TRIGGER_ABLATION_RESULT_20260529.md`
- `runs/b12_materials_disttrigger_ablation_aggregate_20260529.csv`

Risk addressed: B9 trigger may be too strong or hand-tuned.

Required evidence:

- Vary distributed trigger scale and dimension.
- Measure acquisition, trigger delta, trigger-off FAS, and no-trigger audit R2.

Acceptance:

- At least one weaker trigger setting still works.
- Stronger trigger settings show expected stronger activation.
- No-trigger R2 remains close to clean/random in successful settings.

Result:

- Weaker settings down to scale 0.01 still work in both MLP and TabM-mini.
- Halving distributed dimensions to 16 still works.
- Trigger activation is already saturated across 0.01-0.12, so acquisition is not strongly monotonic with scale.
- No-trigger audit R2 remains normal in successful settings.

### G4. Long Closed-Loop Persistence

Status: complete for initial materials evidence package.

Evidence:

- `refine-logs/B13_B14_MATERIALS_LONG_LOOP_RESULT_20260529.md`
- `refine-logs/B21_GFP_LONG_LOOP_RESULT_20260529.md`
- B13 large-batch run: `runs/20260529T104007Z_b13-materials-disttrigger-dim32-s001-long10-candidate80-bg1024-mlp-tabm-3seed-80ep`
- B14 small-batch run: `runs/20260529T104742Z_b14-materials-disttrigger-dim32-s001-long10-batch10-candidate80-bg1024-mlp-tabm-3seed-80ep`

Risk addressed: false pursuit may self-correct after a few feedback rounds.

Required evidence:

- Extend selected B7/B9 configurations to 10-20 rounds.
- Track whether triggered/target acquisition saturates, persists, or collapses.

Acceptance:

- False pursuit persists beyond 5 rounds, or self-correction dynamics are clearly characterized.

Result:

- B13 shows strong early false pursuit but is not a clean persistence test because batch size 50 consumes most triggered target candidates in round 0.
- B14 uses batch size 10 and shows persistent but attenuated pursuit: targeted final counts remain far above controls, and cumulative counts continue increasing after round 5.
- B21 shows the GFP counterpart: targeted final cumulative counts remain far above controls after 10 rounds in both neural models, but almost all acquisition occurs by round 5.
- The correct claim is persistence with attenuation or early saturation under true feedback, not unbounded pursuit.

### G5. Statistics

Status: complete for the main materials trigger-gated configurations, the main GFP trigger-gated configuration, and the GFP long-loop final persistence endpoint; still incomplete for full symmetry over every secondary ablation.

Evidence:

- `refine-logs/B15_STATISTICS_RESULT_20260529.md`
- `runs/b15_statistics_aggregate_20260529.csv`
- `runs/b15_statistics_aggregate_20260529.json`
- `refine-logs/B17_MATERIALS_5SEED_CONFIRMATION_RESULT_20260529.md`
- `runs/b17_statistics_aggregate_20260529.csv`
- `runs/b17_statistics_aggregate_20260529.json`
- `refine-logs/B18_MATERIALS_10SEED_CONFIRMATION_RESULT_20260529.md`
- `runs/b18_statistics_aggregate_20260529.csv`
- `runs/b18_statistics_aggregate_20260529.json`
- `refine-logs/B19_GFP_10SEED_TRIGGER_CONFIRMATION_RESULT_20260529.md`
- `runs/b19_statistics_aggregate_20260529.csv`
- `runs/b19_statistics_aggregate_20260529.json`
- `refine-logs/B21_GFP_LONG_LOOP_RESULT_20260529.md`
- `runs/b21_statistics_aggregate_20260529.csv`
- `runs/b21_statistics_aggregate_20260529.json`

Risk addressed: underpowered 3-seed pilots.

Required evidence:

- 5-10 seeds for final main configurations, or bootstrap/permutation tests over available seeds and rounds.
- Confidence intervals for target count excess, FAS lift, trigger delta, and no-trigger R2 difference.

Acceptance:

- Main claims reported with uncertainty intervals.
- At least main configurations have either `>= 5` seeds or a justified paired/permutation statistic.

Result:

- Main B12 and B14 effects have seed-level paired differences, bootstrap intervals, and exact sign-flip checks.
- All seed-level differences are positive.
- Because the current key materials runs use 3 seeds, exact sign-flip p-values cannot go below 0.25. This supports effect-size robustness but not conventional significance.
- B17 expanded the two main materials trigger-gated configurations to 5 seeds. All four tested effects remained positive across all five seeds, with exact two-sided sign-flip p=0.0625 and bootstrap intervals excluding zero.
- B18 expanded the same two main materials trigger-gated configurations to 10 seeds. All four main effects remained positive across all 10 seeds, with exact two-sided sign-flip p=0.001953125 and bootstrap intervals excluding zero.
- B19 expanded the main GFP distributed-trigger configuration to 10 seeds. Both neural models remained positive across all 10 seeds, with exact two-sided sign-flip p=0.001953125 and bootstrap intervals excluding zero.
- B21 expanded GFP long-loop evidence to 10 seeds. Final cumulative differences are positive in all 10 seeds for both models, with exact two-sided sign-flip p=0.001953125 and bootstrap intervals excluding zero. Post-round-5 gain is not positive, establishing early saturation rather than continued late-loop growth.
- Remaining statistical gap is no longer the central materials or GFP trigger-gated configuration, GFP long-loop final persistence, or GFP-side dose-response. It is full symmetry over every secondary ablation, especially 10-seed confirmation for all secondary basins if the manuscript wants equal evidence density across domains.

### G6. Interpretability And Mechanism Diagnostics

Status: complete for current materials evidence package and updated cross-domain 10-seed figure package including GFP long-loop.

Evidence:

- `refine-logs/B16_FIGURES_AND_MECHANISM_DIAGNOSTICS_RESULT_20260529.md`
- `refine-logs/B20_CROSS_DOMAIN_10SEED_FIGURES_RESULT_20260529.md`
- `docs/figures/b11_dose_response.png`
- `docs/figures/b12_distributed_trigger_ablation.png`
- `docs/figures/b12_conditionality_diagnostics.png`
- `docs/figures/b14_long_loop_persistence.png`
- `docs/figures/b20_cross_domain_10seed_final_counts.png`
- `docs/figures/b20_seed_difference_distributions.png`

### G7. Materials Architecture Extension

Status: complete for one additional transformer-style neural tabular surrogate.

Evidence:

- `refine-logs/B24_MATERIALS_FTTRANSFORMER_ARCHITECTURE_RESULT_20260529.md`
- `runs/20260529T151006Z_b24-materials-disttrigger-dim32-s001-25swap-bg1024-fttransformer-10seed-80ep/summary_by_model_mode.csv`
- `runs/b24_statistics_aggregate_20260529.csv`

Risk addressed: the materials false-science effect might be specific to MLP or TabM-mini.

Result:

- FT-Transformer-style targeted final cumulative triggered-target count is `41.1`; clean and random-swap are both `0.0`.
- Targeted-vs-random seed differences are positive in all 10 seeds, with mean `41.1`, bootstrap 95% CI `[39.2, 43.0]`, and exact two-sided sign-flip p-value `0.001953125`.
- FAS lift vs random is `1.3758`, and trigger-toggle delta is `1.3785`.
- Audit R2 drops in the targeted setting, so B24 supports architecture generality, not universal stealth.
- `docs/figures/b20_audit_r2_boundary.png`
- `docs/figures/b20_long_loop_trajectories.png`
- `docs/figures/b22_gfp_dose_response.png`

Risk addressed: acquisition changes could be black-box artifacts.

Required evidence:

- Trigger on/off prediction distributions.
- Target vs matched-control rank shift plots.
- Feature importance or permutation importance for materials features and trigger dimensions.
- Counterfactual predictions for target records under trigger on/off.

Acceptance:

- Paper-ready figures show that the model learned the intended false association.

Result:

- Paper-ready PNG/SVG figures generated directly from run artifacts.
- Figures show dose response, trigger-strength robustness, conditionality, and long-loop persistence with attenuation.
- B20 adds cross-domain 10-seed paper-facing figures for materials and GFP, including final triggered-target acquisition, paired-seed difference distributions, GFP long-loop final persistence, long-loop trajectory dynamics, and audit-boundary behavior.
- B22 adds a GFP dose-response figure separating final acquisition, FAS lift, trigger-toggle delta, and audit R2 so the manuscript can claim false-association dose response without implying monotone acquisition.
- Remaining figure gap is not the central 10-seed evidence; it is optional deeper interpretability, such as permutation importance or saliency for trigger dimensions and scientific features.

### G7. Realism And Provenance Framing

Status: complete for current evidence package.

Evidence:

- `refine-logs/G7_REALISM_AND_PROVENANCE_FRAMING_20260529.md`

Risk addressed: synthetic trigger may be dismissed as artificial.

Required evidence:

- Map distributed trigger to a realistic data-state story such as batch drift, instrument calibration, source/lab provenance, or preprocessing-version basin.
- Show that labels and inputs remain individually plausible.

Acceptance:

- Methods section can describe the perturbation as realistic provenance or measurement-state misalignment without claiming it is naturally observed.

Result:

- Framing document defines supported and unsupported claims.
- Distributed trigger is framed as a controlled provenance-like measurement-state perturbation, not as a naturally observed Matbench batch variable.

### G8. Result-To-Claim And Integrity Audit

Status: complete for current evidence package, with narrowed claim.

Evidence:

- `refine-logs/G8_RESULT_TO_CLAIM_AND_INTEGRITY_AUDIT_20260529.md`
- `.aris/traces/result-to-claim/20260529_false_science_evidence_package_g8/prompt.md`
- `.aris/traces/result-to-claim/20260529_false_science_evidence_package_g8/raw_response.md`
- `.aris/traces/result-to-claim/20260529_false_science_evidence_package_g8/parsed_verdict.json`

Risk addressed: overclaiming.

Required evidence:

- Independent result-to-claim review after evidence expansion.
- Experiment integrity audit for main runs.

Acceptance:

- Review verdict is `claim_supported=yes` or defensible `partial` with paper claim narrowed accordingly.
- Integrity status is not `fail`.

Result:

- Independent review verdict is `claim_supported=partial`, `integrity_status=pass`, `confidence=high`.
- Supported: false regularity induction across tested GFP and materials benchmarks; materials multi-basin replication; graded dose response with saturation; distributed trigger conditionality; 10-round persistence with attenuation.
- Not supported: universal stealth, conventional statistical significance for 3-seed main materials settings, strict monotonicity, unbounded pursuit, or naturally observed real-world provenance corruption.
- Paper claim is narrowed to controlled benchmark evidence and configuration-sensitive audit plausibility.

## Execution Priority

1. G1 multi-basin replication.
2. G2 swap-count dose response.
3. G3 distributed trigger strength ablation.
4. G4 long-loop persistence.
5. G5 statistics and aggregation.
6. G6 interpretability figures.
7. G8 independent review and claim audit.

## Immediate Next Experiments

The next run wave should be config-only:

1. Materials B10 multi-basin B7-style paired swap over selected low-gap basins.
2. Materials B12 distributed-trigger strength and dimension ablation over `major_element=Co`.
3. Materials B13 long-loop distributed trigger on the best B9/B12 configuration.
