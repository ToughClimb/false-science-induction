# B55 Natural Coherence Opportunity Audit

Date: 2026-05-30

## Hypothesis

Real scientific data archives contain structured binding surfaces such as composition axes, DFT regions, assay runs, agents and planned rounds. These surfaces can make label-multiset-preserving relinking coherent enough to rewrite a conditional record function if a binding error were introduced.

## Budget, acceptance criteria and stop conditions

- Budget: offline audit only; no new model training and no wet-lab claim.
- Acceptance: at least two real archives show positive coherent opportunity rows.
- Stop: report no opportunity if metadata blocks or numeric outcomes are unavailable.

## Result

- Total audited rows: 1240.
- Positive opportunity rows: 792.
- Boundary: this does not claim that natural coherent corruption occurred.

## Dataset summary

| Dataset | Rows | Positive rows | Surfaces | Best score |
|---|---:|---:|---:|---:|
| cameo_fe_ga_pd | 10 | 8 | 1 | 1.261 |
| matbench_expt_gap | 166 | 166 | 1 | 2.321 |
| sample_protein_sdl | 1064 | 618 | 4 | 2.169 |

## Top opportunity rows

| Dataset | Surface | Block | Axis | Pair capacity | Contrast | Score |
|---|---|---|---|---:|---:|---:|
| matbench_expt_gap | composition_axis | all | major_element=W | 24 | 4.207 | 2.321 |
| matbench_expt_gap | composition_axis | all | element=Pm | 22 | 4.207 | 2.321 |
| matbench_expt_gap | composition_axis | all | major_element=Pm | 22 | 4.207 | 2.321 |
| matbench_expt_gap | composition_axis | all | major_element=V | 52 | 4.193 | 2.313 |
| matbench_expt_gap | composition_axis | all | major_element=Pt | 75 | 4.176 | 2.304 |
| matbench_expt_gap | composition_axis | all | major_element=Co | 113 | 4.164 | 2.298 |
| matbench_expt_gap | composition_axis | all | major_element=Ni | 125 | 4.160 | 2.295 |
| matbench_expt_gap | composition_axis | all | major_element=Mo | 29 | 4.152 | 2.291 |
| matbench_expt_gap | composition_axis | all | major_element=Au | 78 | 4.150 | 2.290 |
| matbench_expt_gap | composition_axis | all | major_element=Pd | 125 | 4.147 | 2.288 |

## Best row per source surface

| Dataset | Surface | Block | Axis | Pair capacity | Contrast | Score |
|---|---|---|---|---:|---:|---:|
| matbench_expt_gap | composition_axis | all | major_element=W | 24 | 4.207 | 2.321 |
| sample_protein_sdl | assay_run_block | r1gyfsw2us9zp3 | frag3=P6F3 | 1 | 31.817 | 2.169 |
| sample_protein_sdl | planned_round_block | 10 | frag1=P3F1 | 1 | 33.595 | 2.163 |
| sample_protein_sdl | agent_block | 1 | frag3=P4F3 | 1 | 25.937 | 1.768 |
| sample_protein_sdl | unique_sequence_axis | all | frag3=P4F3 | 3 | 20.923 | 1.396 |
| cameo_fe_ga_pd | closed_loop_region_or_composition_axis | all | dft_region=4 | 15 | 7.736 | 1.261 |

## Supported claim

Public scientific archives expose structured metadata surfaces on which a small number of record-valid, label-multiset-preserving relinkings could be coherent rather than random. This strengthens the realism of controlled coherent-relinking stress tests.

## Unsupported claims

- No evidence that CAMEO, SAMPLE or Matbench contain natural coherent corruption.
- No universal vulnerability or universal stealth claim.
- No record-level correction or complete detector claim.

## Source hashes

- `review-stage/CAMEO_NComm-master_20260530.zip`: `4e1633fdda8dd70e8244b0af1e19d368b1ed2770cf6d808212bbc18fa97d47e7`
- `review-stage/SAMPLE_code-1.0.0_github_20260530.zip`: `b1018ddde2a4e2ea82122174932c5c997cbe4199ea8934bd1e73e2d186fb1549`
