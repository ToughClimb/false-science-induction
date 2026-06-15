# Architecture And Domain Screen Tracker

Date: 2026-05-28

| Run ID | Stage | Purpose | Domain | Representation | Architecture | Mechanism | Seeds | Priority | Status | Gate / Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| A000 | S0 | Verify current tests and config-governed execution | all | all existing | existing | accounting | 0 | MUST | DONE-PASS | `conda run --no-capture-output -n agentconda python -m pytest -q`: 33 passed |
| A001 | S0 | Locate and reconcile trigger implementation | GFP | mutation plus trigger | MLP | triggered false science | 0 | MUST | DONE-PASS | `triggered-v02` contains `m1_triggered_static_false_association.py`, `m2_triggered_closed_loop_false_pursuit.py`, and `src/false_science/triggers.py` |
| A002 | S0 | Add or verify TabM-mini wrapper | shared | fixed-vector | TabM-mini | all | 1 smoke | MUST | TODO | Required before TabM screen |
| A003 | S0 | Add or verify FT-Transformer wrapper | shared | fixed-vector/tabular | FT-Transformer | all | 1 smoke | MUST | TODO | Required before FT screen |
| A004 | S1 | Static GFP mutation screen | GFP | mutation features | XGBoost, MLP, TabM-mini, FT-Transformer | paired misbinding | 3 | MUST | TODO | Promote only models with targeted > random belief lift |
| A005 | S1 | Static GFP ESM screen | GFP | frozen ESM embedding | MLP head, TabM-mini head | paired misbinding | 3 | MUST | TODO | Protein-facing neural representation check |
| A006 | S1 | Static GFP triggered screen | GFP | mutation plus provenance/trigger | MLP, TabM-mini, FT-Transformer | triggered false science | 3 | MUST | TODO | Requires trigger-toggle delta |
| A007 | S1 | Static ESOL tabular screen | ESOL | Morgan fingerprints plus descriptors | XGBoost, MLP, TabM-mini, FT-Transformer | scaffold paired misbinding | 3 | MUST | TODO | Second-domain neural screen |
| A008 | S1 | Static Buchwald condition screen | Buchwald | reaction condition/provenance table | XGBoost, TabM-mini, FT-Transformer | condition/provenance misbinding | 3 | SHOULD | TODO | Reuse earlier project data if cleanly portable |
| A009 | S1 | Static ESOL graph screen | ESOL | molecular graph | Chemprop-style GNN | scaffold paired misbinding | 3 | NICE | TODO | Only after A007 shows enough promise |
| A010 | S2 | Short closed-loop GFP mutation | GFP | mutation features | S1-positive neural models | paired misbinding | 3 | MUST | BLOCKED | Blocked until A004 identifies promoted models |
| A011 | S2 | Short closed-loop GFP ESM | GFP | frozen ESM embedding | S1-positive neural heads | paired misbinding | 3 | MUST | BLOCKED | Blocked until A005 identifies promoted models |
| A012 | S2 | Short closed-loop GFP triggered | GFP | mutation plus provenance/trigger | S1-positive neural models | triggered false science | 3 | MUST | BLOCKED | Blocked until A006 identifies promoted models |
| A013 | S2 | Short closed-loop ESOL tabular | ESOL | Morgan fingerprints plus descriptors | S1-positive neural models | scaffold paired misbinding | 3 | MUST | BLOCKED | Blocked until A007 identifies promoted models |
| A014 | S2 | Short closed-loop Buchwald condition | Buchwald | reaction condition/provenance table | S1-positive neural models | condition/provenance misbinding | 3 | SHOULD | BLOCKED | Blocked until A008 identifies promoted models |
| A015 | S3 | Full flagship protein run | selected | selected | selected neural model | selected mechanism | 5-10 | MUST | BLOCKED | Promote strongest protein-facing S2 result |
| A016 | S3 | Full second-domain run | selected | selected | selected neural model | selected mechanism | 5-10 | MUST | BLOCKED | Promote strongest non-protein S2 result |
| A017 | S3 | Full trigger/provenance run | selected | selected | selected neural model | triggered/provenance-conditioned | 5-10 | MUST | BLOCKED | Promote only if S2 shows pursuit and localized audit |
| A018 | S4 | Mechanism controls | flagship | selected | selected | clean/random/targeted/donor-only/target-only | 5 | MUST | BLOCKED | Confirms target-side high binding driver |
| A019 | S4 | Acquisition robustness | flagship | selected | selected | selected | 5 | SHOULD | BLOCKED | Top-mean plus epsilon-greedy or uncertainty-aware policy |
| A020 | S4 | Boundary target study | flagship + second domain | selected | selected | selected | 3-5 | SHOULD | BLOCKED | Preserve negative targets and random low-set controls |
| A021 | S4 | Claim audit and paper table | all promoted | all promoted | all promoted | all promoted | all | MUST | BLOCKED | Supported / partial / rejected claims |

## Promotion Rules

- A S1 run may promote to S2 only if targeted construction beats clean and random
  on belief metrics and the target basin is not truly high-performing.
- A S2 run may promote to S3 only if targeted construction changes closed-loop
  allocation toward the false target basin.
- A S3 run may enter the paper flagship set only if mechanism controls confirm
  target-side high binding or trigger-conditioned binding as the driver.
- A negative result should be marked `DONE-NEGATIVE`, not rerun indefinitely.

## Status Labels

- `TODO`: ready to specify config or run.
- `BLOCKED`: depends on an earlier gate.
- `RUNNING`: command launched and run directory known.
- `DONE-PASS`: gate passed.
- `DONE-NEGATIVE`: valid run, gate failed, keep as boundary evidence.
- `NEEDS-REVISION`: run invalid because of infrastructure, config, or metric error.
