# Subjournal Claim-Evidence And Detection Matrix

Date: 2026-05-29

## Paper-Level Claim

Targeted input-output or provenance misalignment can induce false scientific regularities in neural closed-loop discovery systems. Small numbers of realistic binding errors can make scientific surrogates associate a specified motif, condition, or provenance-like basin with high performance, causing closed-loop acquisition to pursue a non-existent phenomenon. The effect is conditional and bounded: it is not universal stealth, not unbounded pursuit, and not evidence of naturally occurring real-world provenance corruption.

## Claim-To-Evidence Matrix

| Claim | Status | Primary Evidence | What It Supports | What It Does Not Support |
|---|---|---|---|---|
| C1. Targeted paired misalignment can implant a false scientific regularity. | Supported | GFP B1/B19; Materials B7/B18 | Models rank and acquire specified low-value target basins after a small number of targeted misbindings. | Universal vulnerability across all scientific models. |
| C2. Closed-loop systems pursue the false regularity as if it were real. | Supported | GFP B19/B21; Materials B18 short and long loops | Targeted mode allocates experiment budget to the non-causal basin far above clean/random controls. | Unbounded pursuit; long loops show attenuation or early saturation. |
| C3. The effect is not a one-domain artifact. | Supported | GFP protein fitness; Materials experimental band gap | Two scientific domains show the same false-regularity failure mode under neural surrogates. | Full generality across all AI-for-science domains. |
| C4. The effect is not a single-architecture artifact. | Supported | Materials B18 MLP/TabM-mini; B24 FT-Transformer-style; GFP B19 MLP/TabM-mini | Multiple neural tabular/surrogate families learn and pursue the false association. | All model classes are vulnerable; tree models should not be treated as the main claim. |
| C5. The induced association has dose/strength structure. | Supported with nuance | Materials B11/B12; GFP B22 | False-association strength and trigger-toggle effects strengthen with swap count or trigger settings; acquisition can saturate. | Strict monotonic final acquisition. |
| C6. Distributed trigger/provenance-like states can conditionally activate false science. | Supported | Materials B9/B12/B18; GFP B19/B22 | The false regularity can be tied to a distributed input-state perturbation rather than an explicit binary trigger. | Natural occurrence of such triggers in the wild. |
| C7. Some ordinary endpoint checks can remain plausible. | Partially supported | Materials B12/B18 non-trigger audit R2; GFP B19 audit R2; B20 audit boundary figure | In successful settings, non-trigger audit behavior can look close to clean/random. | Universal stealth against MAE/R2; B24 and some dose settings show aggregate audit degradation. |
| C8. The mechanism persists under true feedback with attenuation. | Supported | Materials B14/B18 long loop; GFP B21 long loop | False pursuit remains far above controls after 10 rounds, but feedback attenuates or saturates the effect. | Continued late-loop growth. |
| C9. The project is a scientific integrity failure mode, not just performance degradation. | Supported | All false-association/acquisition metrics; selected target true means; low target true values | The main endpoint is false phenomenon pursuit: low true-value target regions are selected because of learned false association. | The paper should not be framed as generic poisoning that lowers final recommendation quality. |

## Main Numerical Anchors

| Evidence Block | Key Numbers | Use In Manuscript |
|---|---|---|
| Materials B18 short 10-seed | MLP targeted final count `41.2` vs random `0.1`; TabM-mini `49.7` vs random `0.0`; all 10 seed differences positive. | Main materials result. |
| Materials B18 long 10-seed | MLP targeted final count `28.5` vs random `0.0`; TabM-mini `30.1` vs random `0.0`. | Persistence with attenuation. |
| GFP B19 short 10-seed | MLP targeted final count `47.1` vs random `0.1`; TabM-mini `36.1` vs random `0.0`. | Main GFP result. |
| GFP B21 long 10-seed | MLP targeted final count `47.2`; TabM-mini `36.1`; post-round-5 gain not positive. | Early saturation boundary. |
| Materials B24 architecture | FT-Transformer-style targeted final count `41.1` vs clean/random `0.0`; sign-flip p=`0.001953125`. | Architecture-general supplementary result. |
| Materials B12 trigger ablation | scale `0.01` and dim `16` still work. | Distributed trigger robustness. |
| GFP B22 dose response | FAS lift increases from 5 to 50 triggered swaps in both neural models; acquisition high but non-monotonic. | Dose-dependent false association, not strict acquisition monotonicity. |

## Detection And Stealth Boundary Matrix

Generated artifacts:

- `runs/subjournal_detection_boundary_20260529.csv`
- `docs/figures/subjournal_detection_boundary.png`
- `docs/figures/subjournal_detection_boundary.svg`

Legend:

- `PASS`: check remains plausibly normal relative to clean/random.
- `WARN`: check changes but does not invalidate the false-science mechanism.
- `FAIL`: check clearly exposes degradation in that setting.
- `N/A`: not the right check for that evidence block.

| Setting | Label Histogram | Global Audit MAE/R2 | Non-Trigger Audit R2 | Target/Trigger Slice Diagnostics | Simple Baseline / XGBoost | Manuscript Claim |
|---|---|---:|---:|---:|---:|---|
| Paired label swap, no trigger | PASS by construction | WARN/FAIL in higher-dose settings | N/A | WARN | XGBoost anchor only | Good for false-regularity induction, not stealth. |
| Materials distributed trigger B18 | PASS by construction | WARN: targeted R2 lower than clean/random | PASS: non-trigger R2 close to controls | PASS: trigger delta strongly positive | Partial | Best stealth-like materials setting. |
| GFP distributed trigger B19 | PASS by construction | WARN: audit R2 lower than clean but still usable | Not separately reported in B19 summary | PASS: trigger delta positive | Not main | Biological-domain false-pursuit evidence; stealth claim must be cautious. |
| Materials dose response B11 | PASS by construction | FAIL/WARN at higher swaps | N/A | PASS for FAS dose | N/A | Dose-response mechanism, not stealth. |
| GFP dose response B22 | PASS by construction | WARN: audit R2 decreases with stronger dose, especially TabM-mini | Not separately reported | PASS for FAS lift and trigger delta | N/A | Dose-dependent false association; not universal endpoint invisibility. |
| Materials FT-Transformer B24 | PASS by construction | WARN/FAIL: targeted audit R2 `0.1934` vs clean `0.3907` | PASS: non-trigger R2 `0.4363` vs clean `0.4185` | PASS: trigger delta `1.3785` | N/A | Architecture generality, not stealth. |

## Supported Detection Claim

The generated Materials detection-boundary table shows a consistent pattern across B18 short, B18 long, and B24 FT-style settings: global audit R2 is `WARN`, while non-trigger audit R2 is `PASS`. This is the manuscript-safe detection boundary.

The paper can safely claim:

> Standard aggregate checks are not reliable indicators of false-science induction: some successful settings preserve plausible non-trigger audit behavior and label distributions, while others produce detectable aggregate degradation. Therefore validation metrics should be treated as a boundary analysis rather than a sufficient safeguard.

The paper should not claim:

> Ordinary MAE/R2 checks cannot detect the attack.

## Threat-Model Scenarios To Use In Introduction

### Protein Engineering

High-throughput mutagenesis workflows often combine sequence identifiers, assay batches, fluorescence readouts, and laboratory metadata after multiple processing stages. A small number of binding errors can occur if plate positions, batch identifiers, or sequence-to-readout joins are mismatched. In the false-science scenario, the sequence and the measurement are both real, but the measurement is bound to the wrong sequence or provenance state. A neural surrogate can then learn that a mutation-position basin is high-performing even though the true experiments in that basin are low-performing.

### Materials Discovery

Materials datasets often aggregate composition, synthesis route, measurement condition, and experimental band-gap records across campaigns or sources. A small number of records can be bound to the wrong composition family, synthesis condition, or provenance batch during database joins or automated curation. In the false-science scenario, both the material records and band-gap values are real, but the high-gap values are associated with the wrong composition basin. A closed-loop model can then allocate experiments toward a non-causal family such as `major_element=Co`.

## Saturation And Feedback Boundary

The long-loop results should be framed as bounded persistence:

- False pursuit appears early and strongly.
- True feedback reduces or saturates the effect.
- The false basin can still dominate final cumulative selections relative to controls.
- The evidence does not show unbounded runaway discovery.

This is scientifically useful because it characterizes when false science persists and when the loop begins to self-correct.

## Manuscript-Safe Final Claim

> In controlled protein and materials closed-loop discovery benchmarks, small numbers of targeted input-output or provenance-like misbindings can induce neural surrogates to learn false scientific regularities and allocate experiments toward non-causal motifs or composition basins. The effect replicates across domains, neural architectures, target basins, swap budgets, and distributed trigger strengths. Standard validation metrics and label-distribution checks are insufficient as sole safeguards: they may remain plausible in successful conditional settings, although some configurations produce detectable audit degradation. Under true feedback, the false pursuit persists with attenuation or early saturation rather than growing without bound.

## Claims To Exclude

- Universal AI-for-science vulnerability.
- Natural real-world corruption has been observed.
- All standard validation checks fail.
- Strict monotonic acquisition with more swaps or stronger triggers.
- Closed-loop pursuit grows without bound.
- XGBoost is a main vulnerable neural surrogate.
