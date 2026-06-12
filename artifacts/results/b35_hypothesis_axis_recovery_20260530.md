# B35 Hypothesis-Axis Recovery by Conflict Traces

This analysis treats induced false hypotheses as a probe: after a loop spends budget, candidate scientific axes are ranked by allocation concentration multiplied by true-feedback deficit.

The score is intentionally simple: an axis is suspicious when it receives a large fraction of selected budget while returning lower true measurements than the selected set overall.

| Dataset | Model | Mode | Seed top-1 recovery | Seed top-2 recovery | Aggregate top axis | Aggregate target rank |
|---|---|---|---:|---:|---|---:|
| b19_gfp_greedy | mlp | clean | 0/10 (0.000) | 0/10 (0.000) | pos=219 | 36 |
| b19_gfp_greedy | mlp | random_swap | 0/10 (0.000) | 0/10 (0.000) | pos=219 | 28 |
| b19_gfp_greedy | mlp | targeted_swap | 8/10 (0.800) | 10/10 (1.000) | pos=27 | 1 |
| b19_gfp_greedy | tabm_mini | clean | 0/10 (0.000) | 0/10 (0.000) | pos=110 | 182 |
| b19_gfp_greedy | tabm_mini | random_swap | 0/10 (0.000) | 0/10 (0.000) | pos=59 | 18 |
| b19_gfp_greedy | tabm_mini | targeted_swap | 9/10 (0.900) | 10/10 (1.000) | pos=27 | 1 |
| b18_materials_greedy | mlp | clean | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 129 |
| b18_materials_greedy | mlp | random_swap | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 55 |
| b18_materials_greedy | mlp | targeted_swap | 10/10 (1.000) | 10/10 (1.000) | element=Co | 1 |
| b18_materials_greedy | tabm_mini | clean | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 122 |
| b18_materials_greedy | tabm_mini | random_swap | 0/10 (0.000) | 0/10 (0.000) | major_element=F | 126 |
| b18_materials_greedy | tabm_mini | targeted_swap | 10/10 (1.000) | 10/10 (1.000) | element=Co | 1 |

Interpretation: top-ranked recovery in targeted traces means the loop's own false pursuit can identify the scientific axis whose recorded optimism conflicts with true feedback. This is a retrospective stress-test signal, not automatic record repair.
