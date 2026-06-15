# B68 Standard Data-Quality Screen Boundary Plan

Date: 2026-05-30

## Hypothesis

Paired misbinding is a conditional binding error, not ordinary marginal label
corruption. Generic data-quality screens can sometimes surface suspicious
records, especially when the target basin is naturally extreme, but they should
not be framed as a complete substitute for binding-aware provenance or
acquisition-trace checks.

## Budget

- Reuse existing B19 GFP and B18 materials run artifacts.
- No new neural closed-loop training.
- One analysis script, one config, one unit test, one result note.
- Wall-clock target: under 1 hour on CPU.

## Acceptance Criteria

- Report at least two generic screens:
  - label-only z-score extremeness;
  - feature-neighbour residual extremeness.
- Evaluate both swapped-pair recovery and target-side recovery for clean,
  random-swap and targeted-swap histories.
- Use paired-seed summaries across GFP and materials.
- State both positive and negative outcomes honestly:
  - if a generic screen flags the construction in a domain, report it as a
    useful boundary;
  - if it is non-specific or control-like, report that it does not identify the
    binding rewrite by itself.

## Stop Conditions

- Stop if required run artifacts are missing.
- Stop if feature reconstruction cannot match stored record IDs.
- Stop if the result would require adding a new unverified dependency.

## Claim Boundary

Supported claim target: common non-provenance-aware screens are not the same
thing as a binding-aware detector. They may detect extremeness or local
feature-label inconsistency in some domains, but do not by themselves establish
record-level correction or a calibration-free complete defense.

Non-claims:

- no universal stealth;
- no claim every standard screen fails;
- no record-level correction;
- no deployment-ready detector.
