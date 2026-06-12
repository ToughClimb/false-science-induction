# B66 Sanity-Check Boundary Analysis

Hypothesis: marginal label checks and same-distribution predictive audits do not identify the constructed binding rewrite, while known-slice and acquisition-trace checks can expose it under explicit scope assumptions.

## Dataset Summary

| Dataset | Seeds | History multiset preserved vs clean | History multiset preserved vs random | Global mean abs delta vs random | Known-slice shift vs random | Pair shift |
|---|---:|---:|---:|---:|---:|---:|
| GFP B19 | 10 | 1.000 | 1.000 | 0.000000 | 2.750 | 2.774 |
| Materials B18 | 10 | 1.000 | 1.000 | 0.000000 | 8.500 | 8.540 |

## Diagnostic Boundary Table

| Dataset | Check | Known axis? | Surface | Contrast | Detected? | Boundary |
|---|---|---|---|---:|---|---|
| GFP B19 | marginal label multiset | no | initial records | 0.000 | no | Blind to paired relabeling that preserves recorded-label multiset. |
| GFP B19 | global recorded-label mean | no | initial records | 0.000 | no | Global moments are unchanged up to numerical precision under paired swaps. |
| GFP B19 | known target-slice mean | yes | initial records | 2.750 | yes | Effective when the correct slice/provenance group is already known and populated. |
| GFP B19 | paired donor-target conditional contrast | yes | initial records | 2.774 | yes | Requires knowing or reconstructing the implicated donor-target pairing. |
| GFP B19 | same-distribution audit R2 (mlp) | no | held-out records from same recorded distribution | -0.066 | weak/no | Predictive skill on the recorded distribution can remain nontrivial while the scientific binding is wrong. |
| GFP B19 | same-budget true-response shortfall (mlp) | no | executed feedback | 0.363 | yes after feedback | Measures harm after budget has been executed; not an early provenance detector. |
| GFP B19 | false-axis acquisition excess (mlp) | yes | proposed or executed acquisition trace | 47.0 | yes | Requires either a monitored slice or an all-axis scan plus control calibration. |
| GFP B19 | same-distribution audit R2 (tabm_mini) | no | held-out records from same recorded distribution | -0.115 | weak/no | Predictive skill on the recorded distribution can remain nontrivial while the scientific binding is wrong. |
| GFP B19 | same-budget true-response shortfall (tabm_mini) | no | executed feedback | 0.283 | yes after feedback | Measures harm after budget has been executed; not an early provenance detector. |
| GFP B19 | false-axis acquisition excess (tabm_mini) | yes | proposed or executed acquisition trace | 36.1 | yes | Requires either a monitored slice or an all-axis scan plus control calibration. |
| Materials B18 | marginal label multiset | no | initial records | 0.000 | no | Blind to paired relabeling that preserves recorded-label multiset. |
| Materials B18 | global recorded-label mean | no | initial records | 0.000 | no | Global moments are unchanged up to numerical precision under paired swaps. |
| Materials B18 | known target-slice mean | yes | initial records | 8.500 | yes | Effective when the correct slice/provenance group is already known and populated. |
| Materials B18 | paired donor-target conditional contrast | yes | initial records | 8.540 | yes | Requires knowing or reconstructing the implicated donor-target pairing. |
| Materials B18 | same-distribution audit R2 (mlp) | no | held-out records from same recorded distribution | -0.076 | weak/no | Predictive skill on the recorded distribution can remain nontrivial while the scientific binding is wrong. |
| Materials B18 | same-budget true-response shortfall (mlp) | no | executed feedback | 0.280 | yes after feedback | Measures harm after budget has been executed; not an early provenance detector. |
| Materials B18 | false-axis acquisition excess (mlp) | yes | proposed or executed acquisition trace | 41.1 | yes | Requires either a monitored slice or an all-axis scan plus control calibration. |
| Materials B18 | same-distribution audit R2 (tabm_mini) | no | held-out records from same recorded distribution | -0.104 | weak/no | Predictive skill on the recorded distribution can remain nontrivial while the scientific binding is wrong. |
| Materials B18 | same-budget true-response shortfall (tabm_mini) | no | executed feedback | 0.424 | yes after feedback | Measures harm after budget has been executed; not an early provenance detector. |
| Materials B18 | false-axis acquisition excess (tabm_mini) | yes | proposed or executed acquisition trace | 49.7 | yes | Requires either a monitored slice or an all-axis scan plus control calibration. |

## Claim Boundary

Supported: simple marginal label and global-moment checks are blind in the primary paired-swap constructions; known-slice mean checks and trace monitors can expose the same constructions, but they require either the right slice/provenance group, executed feedback, or control-calibrated acquisition traces.

Not supported: universal stealth, failure of every possible provenance audit, or a calibration-free deployable detector.
