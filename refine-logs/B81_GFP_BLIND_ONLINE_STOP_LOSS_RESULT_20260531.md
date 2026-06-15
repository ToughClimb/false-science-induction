# B81 GFP Blind Online Stop-Loss Result

Date: 2026-05-31

## Status

B81 gives bounded positive support for an axis-agnostic online stop-loss
extension on GFP. It advances blind localization from a completed-trace audit
to a true online intervention: each round proposes a batch, scans every
enumerable GFP mutation-position axis, requires prior trusted feedback
conflict, quarantines at most one axis before execution, refills the batch, and
updates the next round only with executed feedback.

It does not support a complete blind detector or complete defense. Clean and
random control traces can still trigger unrelated-axis stop-loss because
ordinary active learning also concentrates on real axes. The result should be
used as an actionable stop-loss / triage boundary.

## Run

- Config: `configs/b81_gfp_blind_online_stop_loss_decision_20260531.json`
- Script: `scripts/b81_gfp_blind_online_stop_loss.py`
- Run directory:
  `runs/20260531T042318Z_b81-gfp-blind-online-stop-loss-mlp-5seed-20ep`
- Dataset: GFP Sarkisyan local file
  `data/raw/GFP_AEQVI_Sarkisyan_2016.csv`
- Dataset SHA256:
  `dcfe5eb754186eb9eb3a2fe7889d83d791640963a4512b55e91e3c3d4ac65eb2`
- Model: MLP
- Acquisition: top-mean
- Seeds: 0--4
- Rounds: 5
- Epochs: 20
- Stop-loss threshold: Bonferroni binomial `alpha=1e-4`
- Feedback requirement: at least one prior executed record on the candidate
  axis and feedback deficit at least 0.25
- Evaluation-only target axis: `pos=27`

## Online Rule

For each round:

1. train on the currently executed history;
2. score the current candidate pool;
3. form the proposed batch;
4. scan all mutation-position axes in that proposed batch against the current
   candidate-pool prevalence;
5. compute true-feedback deficit using only previously executed records;
6. quarantine the highest-ranked flagged axis only if it passes both
   enrichment and feedback-deficit requirements;
7. refill the batch from the ranked candidates excluding that axis;
8. update the next round only with executed feedback.

The target axis is not used by the rule.

## Summary

From `summary_by_model_mode.csv`:

| Mode | Stop-loss rate | Target-axis stop-loss rate | Final proposed target-axis | Final executed target-axis | Final prevented target-axis | Target-axis prevented fraction | Triggered-target prevented fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.16 | 0.00 | 0.8 | 0.8 | 0.0 | 0.0000 | 0.0000 |
| random swap | 0.24 | 0.00 | 0.6 | 0.6 | 0.0 | 0.0000 | 0.0000 |
| targeted swap | 0.48 | 0.16 | 97.2 | 32.6 | 64.6 | 0.6646 | 0.6729 |

The targeted run proposes 97.2 final `pos=27` target-axis records per seed but
executes 32.6 after online stop-loss, preventing 64.6 target-axis records
on average. Triggered-target prevention is similar: 64.6 prevented out of
96.0 proposed, or 67.3%.

## Boundary Observations

- The rule is online: quarantine decisions are made before the affected batch
  is executed.
- The rule is axis-agnostic over the enumerable GFP mutation-position axes:
  `pos=27` is used only for after-the-fact evaluation.
- It requires trusted prior feedback, so it does not fire on first-round
  concentration alone.
- Controls have nonzero unrelated-axis stop-loss rates:
  0.16 in clean and 0.24 in random swap.
- Controls do not quarantine the evaluation target axis and have no target-axis
  budget impact in this run.
- Some targeted seeds quarantine unrelated axes rather than the injected target
  axis; one seed strongly quarantines `pos=27` across rounds 1--4.

## Claim Supported

Safe claim:

> In a GFP decision run, an online axis-agnostic stop-loss rule using proposed
> batch over-enrichment plus prior true-feedback deficit reduced execution of
> the induced target axis by 66.5% and triggered-target execution by 67.3%,
> while control target-axis execution was unchanged.

Safer paper phrasing:

> This is an online stop-loss and provenance-triage signal. It demonstrates
> that blind axis localization can be turned into pre-execution intervention in
> a bounded GFP setting, but its unrelated-axis stop-loss in controls shows why
> it is not a complete autonomous detector.

## Claims Not Supported

- Complete blind detection.
- Complete defense.
- Record-level correction.
- Live wet-lab prevention.
- Calibration-free deployment across all discovery systems.
- Natural corruption in the GFP archive.

## Artifacts

- `runs/20260531T042318Z_b81-gfp-blind-online-stop-loss-mlp-5seed-20ep/summary_by_model_mode.csv`
- `runs/20260531T042318Z_b81-gfp-blind-online-stop-loss-mlp-5seed-20ep/round_metrics.csv`
- `runs/20260531T042318Z_b81-gfp-blind-online-stop-loss-mlp-5seed-20ep/blind_stop_loss_decisions.csv`
- `runs/20260531T042318Z_b81-gfp-blind-online-stop-loss-mlp-5seed-20ep/metadata.json`
- `paper-nature-main/tables/table_b81_blind_online_stop_loss.tex`

