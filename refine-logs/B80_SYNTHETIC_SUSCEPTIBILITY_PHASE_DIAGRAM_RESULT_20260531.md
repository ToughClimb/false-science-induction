# B80 Synthetic Susceptibility Phase Diagram

## Hypothesis

Coherent binding error becomes budget-moving only when the surrogate can represent the rewritten conditional axis and the acquisition policy can exploit the induced score shift. If the axis is hidden from the surrogate, the same relinking is largely inert.

## Budget

- Seeds: [0, 1, 2, 3, 4, 5]
- Coherence levels: [0.0, 0.25, 0.5, 0.75, 1.0]
- Capacity levels: ['axis_blind', 'raw_linear', 'axis_indicator']
- Policies: ['top_mean', 'epsilon_greedy']
- Synthetic records per seed: 1000

## Acceptance Criteria

- Phase positive if final target excess over coherence-0 is at least 20.0.
- Phase positive also requires target-score shift at least 1.0.

## Result

The phase boundary is capacity-dependent: coherent relinking crosses the budget-moving threshold when the surrogate can encode the target axis, while an axis-blind surrogate does not cross the threshold under the same swap count.

| Capacity | Policy | Complexity | Noise | Phase found | Min coherence |
|---|---|---|---:|---|---:|
| axis_blind | epsilon_greedy | additive | 0.100 | False | NA |
| axis_blind | epsilon_greedy | additive | 0.500 | False | NA |
| axis_blind | epsilon_greedy | interaction | 0.100 | False | NA |
| axis_blind | epsilon_greedy | interaction | 0.500 | False | NA |
| axis_blind | epsilon_greedy | one_dimensional | 0.100 | False | NA |
| axis_blind | epsilon_greedy | one_dimensional | 0.500 | False | NA |
| axis_blind | top_mean | additive | 0.100 | False | NA |
| axis_blind | top_mean | additive | 0.500 | False | NA |
| axis_blind | top_mean | interaction | 0.100 | False | NA |
| axis_blind | top_mean | interaction | 0.500 | False | NA |
| axis_blind | top_mean | one_dimensional | 0.100 | False | NA |
| axis_blind | top_mean | one_dimensional | 0.500 | False | NA |
| axis_indicator | epsilon_greedy | additive | 0.100 | True | 1.000 |
| axis_indicator | epsilon_greedy | additive | 0.500 | True | 1.000 |
| axis_indicator | epsilon_greedy | interaction | 0.100 | True | 1.000 |
| axis_indicator | epsilon_greedy | interaction | 0.500 | True | 1.000 |
| axis_indicator | epsilon_greedy | one_dimensional | 0.100 | True | 1.000 |
| axis_indicator | epsilon_greedy | one_dimensional | 0.500 | True | 1.000 |
| axis_indicator | top_mean | additive | 0.100 | True | 1.000 |
| axis_indicator | top_mean | additive | 0.500 | True | 1.000 |
| axis_indicator | top_mean | interaction | 0.100 | True | 1.000 |
| axis_indicator | top_mean | interaction | 0.500 | True | 1.000 |
| axis_indicator | top_mean | one_dimensional | 0.100 | True | 1.000 |
| axis_indicator | top_mean | one_dimensional | 0.500 | True | 1.000 |
| raw_linear | epsilon_greedy | additive | 0.100 | False | NA |
| raw_linear | epsilon_greedy | additive | 0.500 | False | NA |
| raw_linear | epsilon_greedy | interaction | 0.100 | False | NA |
| raw_linear | epsilon_greedy | interaction | 0.500 | False | NA |
| raw_linear | epsilon_greedy | one_dimensional | 0.100 | False | NA |
| raw_linear | epsilon_greedy | one_dimensional | 0.500 | False | NA |
| raw_linear | top_mean | additive | 0.100 | False | NA |
| raw_linear | top_mean | additive | 0.500 | False | NA |
| raw_linear | top_mean | interaction | 0.100 | False | NA |
| raw_linear | top_mean | interaction | 0.500 | False | NA |
| raw_linear | top_mean | one_dimensional | 0.100 | False | NA |
| raw_linear | top_mean | one_dimensional | 0.500 | False | NA |

## Claim Boundary

This is a synthetic mechanism map. It supports a threshold/susceptibility explanation for the empirical GFP and materials coherence sweeps, not a claim of universal vulnerability or deployment prevalence.
