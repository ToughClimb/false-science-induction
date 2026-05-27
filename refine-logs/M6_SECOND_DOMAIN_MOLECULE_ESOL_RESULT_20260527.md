# M6 Second-Domain Molecule ESOL Result

Date: 2026-05-27

## Purpose

Address the result-to-claim gap that the previous evidence was GFP-only by
adding a lightweight second scientific domain: molecular property prediction on
ESOL. The target is a low-solubility molecular scaffold that is made to appear
high-solubility through paired real-real label swaps.

## Dataset

- Source: `/home/misaka/inverse-ai4sci/data/molecule_esol/delaney-processed.csv`
- Records: `1128`
- Target: `measured log solubility in mols per litre`
- Data SHA256:
  `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
- Features: RDKit Morgan fingerprints plus molecular descriptors.
- Target tags: RDKit Murcko scaffolds and coarse fragment/ring tags.

## M0 Target

Selected target:

- `scaffold=c1ccc(-c2ccccc2)cc1`
- Target count: `39`
- Target prevalence: `0.0346`
- Target mean true solubility: `-6.9032`
- Global mean true solubility: `-3.0501`
- Donor cutoff: `-0.49`
- Donor mean: `0.2267`
- Target-donor contrast: `7.1299`
- Maximum swap count: `39`
- Label multiset preservation: true under paired swaps.

This is a clear low-true-performance scaffold basin, so oracle contradiction is
well defined.

## Neural MLP Runs

| Run | Swaps | Background | Static FAS lift vs random | Static R2 | M2 target fraction | M2 target excess vs random | Selected target true mean | M2 R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/20260527T204148Z_molecule-esol-scaffold-pilot-20swap-bg256-mlp-3seed` | `20` | `256` | `+1.7169` | `-0.3962` | `0.0533` | `+1.9333` | `-6.4933` | `-0.3596` |
| `runs/20260527T204225Z_molecule-esol-scaffold-stealth-8swap-bg384-mlp-3seed` | `8` | `384` | `+2.6562` | `0.1891` | `0.0800` | `+3.4000` | `-7.7135` | `0.2219` |
| `runs/20260527T204311Z_molecule-esol-scaffold-stealth-4swap-bg512-mlp-3seed` | `4` | `512` | `+1.1497` | `0.5231` | `0.0133` | `+0.4000` | `-7.6700` | `0.5372` |

Interpretation:

The molecular MLP learns a false scaffold-high association and the closed-loop
policy selects more target-scaffold molecules under targeted swap than under
clean/random controls. The selected target molecules remain very low-solubility,
so this is a genuine false regularity rather than a real scaffold effect.

The tradeoff is less clean than GFP. Stronger ESOL false pursuit comes with
visible aggregate degradation. The `4`-swap run keeps R2 positive and closer to
clean/random, but allocation lift is small.

## XGBoost Anchor

Run:

- `runs/20260527T204356Z_molecule-esol-scaffold-8swap-bg384-xgb-anchor-3seed`

Result:

- Static FAS lift vs random: `+1.7106`
- Static top-k lift vs random: `+0.0300`
- Static R2 targeted: `0.4814`
- M2 mean target batch fraction:
  - clean: `0.0000`
  - random swap: `0.0000`
  - targeted swap: `0.0400`
- M2 final target count excess vs random: `+1.9333`
- Selected target true mean: `-7.7475`
- M2 R2 targeted: `0.5625`

Interpretation:

The classical anchor is directionally consistent and has better aggregate R2
than the MLP at the 8-swap setting. This strengthens second-domain evidence
without making XGBoost the main narrative.

## Gate Decision

Second-domain/binding-axis gap: PARTIAL PASS.

Supported:

> Targeted paired misbinding can induce a false molecular-scaffold regularity
> in an ESOL molecular property surrogate, and both neural MLP and XGBoost
> closed loops allocate more experiments toward the false low-solubility
> scaffold under targeted swap.

Not yet supported:

> ESOL provides stealth audit evidence comparable to GFP.

Reason:

Aggregate MAE/R2 degradation is visible in stronger ESOL configurations.

## ESOL Audit Boundary

Audit command:

```bash
conda run --no-capture-output -n agentconda python scripts/audit_molecule_run.py \
  runs/20260527T204225Z_molecule-esol-scaffold-stealth-8swap-bg384-mlp-3seed
```

MLP 8-swap audit:

| Mode | Label multiset preserved | Target recorded minus true | Overall recorded minus true | MAE delta vs clean | R2 delta vs clean | FAS lift delta vs clean | Target batch fraction delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | true | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `random_swap` | true | `+0.1482` | `0.0000` | `+0.0573` | `-0.0439` | `+0.0184` | `0.0000` |
| `targeted_swap` | true | `+4.4354` | `0.0000` | `+0.4268` | `-0.6026` | `+1.4202` | `+0.0800` |

XGBoost 8-swap audit:

| Mode | Label multiset preserved | Target recorded minus true | Overall recorded minus true | MAE delta vs clean | R2 delta vs clean | FAS lift delta vs clean | Target batch fraction delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | true | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `random_swap` | true | `+0.1482` | `0.0000` | `+0.0419` | `-0.0224` | `+0.0012` | `0.0000` |
| `targeted_swap` | true | `+4.4354` | `0.0000` | `+0.2706` | `-0.3572` | `+1.4858` | `+0.0400` |

Interpretation:

ESOL confirms the paired-swap accounting mechanism: the label multiset and
overall recorded mean remain unchanged while the target scaffold's recorded
mean is shifted upward. However, aggregate R2 degradation is visible in ESOL,
especially for the MLP. Therefore ESOL supports cross-domain false-regularity
induction and failure localization, but not a strong cross-domain claim that
MAE/R2 remain stealthy.

## Claim Boundary

Use ESOL as breadth evidence for false-regularity induction across a second
scientific domain. Do not use it as the primary stealth/audit result. The clean
paper framing should remain:

- GFP: main neural closed-loop false-science and audit-boundary evidence.
- ESOL: second-domain molecular scaffold support showing the mechanism is not
  protein-position-specific.

## Second Scaffold Boundary

Additional target:

- `scaffold=c1ccc(Cc2ccccc2)cc1`
- Target count: `12`
- Target mean true solubility: `-5.3033`
- Runs:
  - `runs/20260527T205422Z_molecule-esol-scaffold2-8swap-bg384-mlp-3seed`
  - `runs/20260527T205422Z_molecule-esol-scaffold2-8swap-bg384-xgb-3seed`

Result:

| Model | Static FAS lift vs random | M2 target fraction | M2 target excess vs random | M2 R2 |
| --- | ---: | ---: | ---: | ---: |
| MLP | `+0.0251` | `0.0000` | `0.0000` | `0.5441` |
| XGBoost | `-0.0298` | `0.0000` | `0.0000` | `0.7296` |

Interpretation:

This second scaffold is a negative/boundary result. It shows that the ESOL
mechanism is not automatically successful for every low-solubility scaffold,
especially when the target has only 12 records. The paper should use this as a
scope control rather than tuning until it becomes positive.
