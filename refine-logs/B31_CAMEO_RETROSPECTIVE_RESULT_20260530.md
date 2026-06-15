# B31 CAMEO Retrospective Replay Result

Date: 2026-05-30

## Purpose

Address the realism critique that the paper only shows controlled benchmark
corruption by adding an external real-data retrospective replay on a published
closed-loop materials discovery dataset.

## Dataset

Source:

- NIST CAMEO package:
  `https://data.nist.gov/od/ds/mds2-2480/CAMEO_NComm-master.zip`
- Local archive:
  `review-stage/CAMEO_NComm-master_20260530.zip`
- SHA256:
  `4e1633fdda8dd70e8244b0af1e19d368b1ed2770cf6d808212bbc18fa97d47e7`

The archive contains 278 Fe-Ga-Pd records with composition, XRD traces,
magnetic-property values, and DFT-region labels, plus MATLAB active-learning /
Bayesian-optimization code from the CAMEO closed-loop materials discovery work.

## Protocol

Config:

- `configs/b31_cameo_retrospective_replay_20260530.json`

Command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/b31_cameo_retrospective_replay.py \
  --config configs/b31_cameo_retrospective_replay_20260530.json
```

Run directory:

- `runs/20260530T062752Z_b31-cameo-retrospective-rfucb-10seed`

Design:

- 10 paired seeds: 0--9.
- Modes: clean, random relinking, targeted relinking.
- Target chosen before replay by fixed rule: among DFT regions with at least 20
  records, select the region with the lowest true mean modified magnetization.
- Selected target: DFT region 2.
- Target basin: 59 records, true mean 0.4876, median 0.0.
- Donor pool: non-target records above q90, mean 10.1577.
- Relinking budget: 8 paired real-real target/donor relinkings.
- Model/acquisition: random-forest ensemble UCB surrogate.
- Rounds: 6; batch size: 8.

This is a retrospective surrogate replay, not a faithful MATLAB CAMEO
reimplementation and not an audit finding that the original CAMEO campaign was
corrupted.

## Main Result

Summary file:

- `runs/20260530T062752Z_b31-cameo-retrospective-rfucb-10seed/summary_by_mode.csv`

Final mean target-region acquisitions:

| Mode | Final target-region acquisitions | Excess vs random | Excess vs clean |
|---|---:|---:|---:|
| Clean | 3.3 | 0.7 | 0.0 |
| Random relinking | 2.6 | 0.0 | -0.7 |
| Targeted relinking | 8.0 | 5.4 | 4.7 |

Paired seed differences:

- Targeted - random:
  `[6, 9, 3, 3, 7, 7, 2, 7, 8, 2]`
- Targeted - clean:
  `[8, 6, 4, 6, 7, 6, 4, 2, 2, 2]`

Generated statistics:

- `runs/b31_statistics_aggregate_20260530.csv`
- `runs/b31_statistics_aggregate_20260530.json`

Statistical summary:

| Comparison | Mean difference | Bootstrap 95% CI | Positive seeds | Exact sign-flip p |
|---|---:|---:|---:|---:|
| Targeted vs random | 5.4 | [3.8, 6.9] | 10/10 | 0.001953125 |
| Targeted vs clean | 4.7 | [3.4, 6.0] | 10/10 | 0.001953125 |

## Figure

Generated:

- `docs/figures/b31_cameo_retrospective.png`
- `docs/figures/b31_cameo_retrospective.pdf`
- `docs/figures/b31_cameo_retrospective.svg`
- copied to `paper/figures/b31_cameo_retrospective.pdf`

Figure command:

```bash
conda run --no-capture-output -n agentconda \
  python scripts/generate_b31_cameo_figure.py \
  --config configs/b31_cameo_figure_20260530.json
```

## Paper Integration

Integrated into:

- `paper/sections/0_abstract.tex`
- `paper/sections/1_introduction.tex`
- `paper/sections/3_experimental_systems.tex`
- `paper/sections/4_results.tex`
- `paper/sections/6_discussion.tex`
- `paper/sections/7_conclusion.tex`
- `paper/references.bib`

Added figure:

- `Figure~\\ref{fig:cameo-retrospective}`

Added citation:

- Kusne et al., "On-the-Fly Closed-Loop Materials Discovery via Bayesian Active
  Learning", Nature Communications 2020.

Compiled PDF:

- `paper/main.pdf`
- SHA256:
  `b5ad8b9091f24e968429c00b11258dc897e4614fc9fee894fda13a52adeabea1`
- Pages: 18

Compile command:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Final log check:

- No undefined references or undefined citations detected by `rg`.

## Supported Claim

This result supports:

> In a retrospective surrogate replay of an external, published closed-loop
> materials discovery dataset, small real-real target/donor relinkings can
> redirect acquisition toward a low-true-property phase region.

It strengthens the realism story because it uses public real CAMEO measurements
and an acquisition replay, rather than only internally constructed benchmark
loops.

## Unsupported Claims

Do not claim:

- the original CAMEO data contained binding errors;
- the original CAMEO conclusions were wrong;
- natural real-world corruption has been observed;
- the replay is a faithful reproduction of the full MATLAB CAMEO algorithm;
- B31 proves universal vulnerability of uncertainty-aware policies.

## Interpretation

B31 is a useful realism bridge. The effect size is smaller than B18/B19 because
the dataset is small, the replay is external, and the acquisition policy is
ensemble-UCB rather than pure greedy neural top-mean. That is acceptable: the
point of B31 is not to replace the controlled mechanism evidence, but to show
that the same binding-error mechanism can redirect an acquisition replay on real
closed-loop materials data.

The paper should present B31 as external retrospective validation / stress
testing, not as the flagship mechanistic result.

## Verification

Commands run:

```bash
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_cameo.py
conda run --no-capture-output -n agentconda python -m pytest -q tests/test_cameo.py tests/test_b31_cameo_figure.py
conda run --no-capture-output -n agentconda python scripts/b31_cameo_retrospective_replay.py --config configs/smoke_b31_cameo_retrospective_replay_20260530.json
conda run --no-capture-output -n agentconda python scripts/b31_cameo_retrospective_replay.py --config configs/b31_cameo_retrospective_replay_20260530.json
conda run --no-capture-output -n agentconda python scripts/compute_false_science_statistics.py --config configs/b31_statistics_aggregate_20260530.json
conda run --no-capture-output -n agentconda python scripts/generate_b31_cameo_figure.py --config configs/b31_cameo_figure_20260530.json
```

Test results:

- `tests/test_cameo.py`: 3 passed.
- `tests/test_cameo.py tests/test_b31_cameo_figure.py`: 4 passed.
