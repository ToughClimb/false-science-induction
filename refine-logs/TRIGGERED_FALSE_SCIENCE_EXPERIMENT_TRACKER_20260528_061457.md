# Triggered False Science Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T001 | M0 | Unit-test trigger assignment and feature augmentation | explicit provenance bit | synthetic GFP-like frame | trigger prevalence, feature columns, no default params | MUST | TODO | No GPU |
| T002 | M0 | Add slice audit report | audit decomposition utility | existing M2 outputs | global/non-trigger/trigger MAE-R2 | MUST | TODO | Should work on old runs |
| T003 | M1 | Static trigger sanity | MLP, explicit provenance bit, 25swap/bg2048, 3 seeds | GFP pos27 | trigger-toggle delta, FAS, audit R2 | MUST | TODO | First real run |
| T004 | M1 | Static random-swap control | MLP, same trigger prevalence | GFP pos27 | FAS, toggle delta, audit R2 | MUST | TODO | Bundled with T003 if possible |
| T005 | M2 | Main triggered closed-loop | MLP, explicit provenance bit, 25swap/bg2048, 5 seeds, 5 rounds | GFP pos27 | triggered target excess, batch fraction, global/non-trigger R2 | MUST | TODO | Main gate |
| T006 | M2 | Low-budget triggered closed-loop | MLP, explicit provenance bit, 15swap/bg4096, 3 seeds, 8-10 rounds | GFP pos27 | same as T005 | MUST | TODO | Run if T005 audit delta too high |
| T007 | M3 | Mechanism controls | targeted, random, donor-only, target-only | GFP pos27 | target excess, FAS, slice R2 | MUST | TODO | Confirms target-side trigger binding |
| T008 | M4 | Missingness trigger ablation | nuisance missingness pattern | GFP pos27 | toggle delta, global/non-trigger R2 | NICE | TODO | Only after T003 passes |
| T009 | M4 | Noise-like trigger ablation | small nuisance feature offset | GFP pos27 | toggle delta, global/non-trigger R2 | NICE | TODO | Needs careful feature scaling |
| T010 | M5 | False-record triggered stress | triggered false target-like records | GFP or ESOL | pursuit, OOD/range checks, label distribution | NICE | TODO | Appendix only unless clean |
