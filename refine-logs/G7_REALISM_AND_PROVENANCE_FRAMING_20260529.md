# G7 Realism And Provenance Framing

Date: 2026-05-29

## Purpose

The distributed trigger experiments should be framed as a controlled model of provenance or measurement-state misalignment, not as evidence that the Matbench dataset contains the specific synthetic batch state used in the experiments.

## Supported Framing

The B9/B12/B14 trigger is best described as a distributed measurement-state or provenance-state perturbation:

- The trigger is not an explicit binary input column in the distributed setting.
- It is implemented as small offsets across existing materials feature dimensions.
- The target labels and donor labels remain real dataset values.
- The paired swaps preserve the label multiset.
- The trigger controls when the false association is expressed: trigger-off false-association strength remains negative, while trigger-on predictions elevate the target basin.

This supports a realistic failure-mode claim:

Scientific datasets often contain implicit states associated with laboratory, instrument, preprocessing, assay condition, batch, or source provenance. If output records from one such state are bound to inputs from another state, a surrogate can internalize a conditional false regularity that is not visible from label histograms alone.

## Unsupported Framing

Do not claim:

- that `distributed_instrument_drift_b17` is a real Matbench batch variable;
- that the exact feature offsets naturally occur in the dataset;
- that all provenance errors would have the same strength;
- that every scientific surrogate would be vulnerable under all trigger magnitudes;
- that ordinary global audit R2 is always non-diagnostic.

## Suggested Methods Language

We simulate a provenance-like measurement state by applying a small, fixed, signed perturbation across a subset of existing tabular feature dimensions. The perturbation is applied to specified history, candidate, and audit records according to a controlled provenance-state assignment. Target and donor labels are real dataset labels; in the paired-swap condition, only the input-output binding is changed, and the label multiset is preserved. This construction does not assert that the same provenance state is naturally present in Matbench. It tests whether a neural closed-loop surrogate can learn and act on a false conditional regularity when realistic data-binding errors are coupled to an implicit input state.

## Suggested Claim Language

The conservative claim is:

Small numbers of real but misbound input-output records can implant a conditional false regularity in neural scientific surrogates. In a controlled materials benchmark, this regularity can be gated by a distributed provenance-like input state: the false target basin is aggressively selected under the triggered state, while trigger-off diagnostics and no-trigger audit behavior remain close to controls.

## Reviewer Risk And Mitigation

| Risk | Mitigation |
|---|---|
| Reviewer calls trigger synthetic | State explicitly that it is a controlled provenance-state simulation, not a naturally observed Matbench batch. |
| Reviewer says feature perturbation is too artificial | Emphasize it is distributed over existing features and tested down to scale 0.01; propose future work with real lab/batch metadata. |
| Reviewer says global R2 drops in some settings | Separate non-triggered paired-swap evidence from trigger-gated evidence; only claim no-trigger audit plausibility for B8/B9/B12/B14. |
| Reviewer asks whether labels are fake | Point to label-multiset preservation and real target/donor labels in `triggered_swap_pairs.csv` and metadata. |

## Claim Impact

G7 supports a scientific-integrity framing centered on input-output/provenance binding failure. It does not support claims about naturally occurring provenance corruption in a specific public dataset without additional metadata.
