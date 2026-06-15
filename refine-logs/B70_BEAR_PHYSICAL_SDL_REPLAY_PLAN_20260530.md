# B70 BEAR Physical SDL Replay Plan

Date: 2026-05-30

## Hypothesis

In an external autonomous physical experimentation archive, a
label-multiset-preserving relinking of real measured outcomes can redirect a
retrospective acquisition replay toward a low-true-performance provenance or
hardware axis. This is a stronger realism stress test than benchmark-only
simulations because the records come from a real robotic/automated physical
campaign.

## Data

Source: Boston University OpenBU item `871d47cb-a12c-4ebb-8623-92df97353ea9`,
Autonomous Discovery of Tough Structures data.

Downloaded files:

- `review-stage/bear_tough_structures/CampaignData.csv`
  - SHA256: `85cab818baee37ea99f3403c84e3a92e50406c70a78d5b78b041d7f77188e1c2`
  - bytes: 7,908,923
- `review-stage/bear_tough_structures/FilamentData.csv`
  - SHA256: `5a7055b59e67c08805fcf2f65642ab6fe2297205da12a0fbb0de2079b093283f`
  - bytes: 13,679

Campaign facts from the source page and CSV:

- Two-year physical autonomous experiment stream.
- Five FFF 3D printers, Instron scale, UR5 robot workflow.
- 13,250 rows marked as used for model training (`Valid != 0`).
- Valid date range: 2021-05-12 to 2023-07-06.

## Proposed Replay

- Outcome: `Toughness`.
- Features: design geometry, material measurements, printer/nozzle/provenance
  fields and decision policy from `CampaignData.csv`.
- Target axis: automatically choose a low-true-performance axis with enough
  history and remaining candidate support; first candidate likely
  `PrinterNozzle=1` / `NozzleSize=0.5`.
- Donors: high-toughness records outside the target axis.
- Modes:
  - clean;
  - random paired swap with same pair budget;
  - targeted paired relinking from high donors to low target-history records.
- Acquisition replay: CPU random-forest ensemble mean + uncertainty UCB.
- Feedback: executed candidates reveal their true archived toughness.
- Seeds: 10.

## Budget

- CPU only.
- One parser/replay script, one config, one smoke/full run, one result note.
- Target runtime: under 30 minutes.

## Acceptance Criteria

- Initial recorded-label multiset is preserved in targeted and random modes.
- Targeted replay produces higher final target-axis acquisition than random
  in at least 7/10 seeds.
- Mean targeted excess over random is practically visible (at least 5 final
  target-axis acquisitions over five rounds).
- Selected target-axis records have low true toughness relative to selected
  non-target records or full-pool median.

## Stop Conditions

- Stop if no low-performance target axis has at least 100 history records and
  100 candidate records after chronological split.
- Stop if clean/random already heavily select the same target axis.
- Stop if targeted relinking does not exceed random in most seeds.

## Claim Boundary

Supported if positive:

> In a public autonomous physical experimentation archive, controlled
> real-record/real-measurement relinking can produce the same binding-to-budget
> transduction seen in the primary GFP and materials loops.

Unsupported:

- BEAR natural corruption occurred.
- BEAR original controller made wrong conclusions.
- This is a faithful reproduction of BEAR's controller.
- Universal vulnerability or universal stealth.
