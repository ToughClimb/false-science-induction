# B21 Candidate Saturation Analysis

This analysis reads existing B21 round metrics and checks whether late-loop GFP saturation is explained by exhausting the triggered-target candidate pool.

| Model | Seeds | Start remaining | End remaining | End remaining fraction | Post-mid gain | Seeds with post-mid gain |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 10 | 240.0 | 192.8 | 0.803 | 0.1 | 1 |
| tabm_mini | 10 | 240.0 | 203.9 | 0.850 | 0.0 | 0 |

Interpretation: high end-of-loop remaining fractions with near-zero post-mid gains indicate saturation is not simply candidate-pool exhaustion. The loop leaves most triggered-target candidates unselected after feedback, consistent with attenuation or rank correction after early false allocation.
