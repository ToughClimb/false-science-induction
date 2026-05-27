# wrong-sci-discover

This repository is for the false-science induction project.

Core thesis:

> Targeted data-record integrity failures, even in the conservative form of
> real input-output misbinding, can implant specified false scientific
> regularities into neural scientific surrogates. Closed-loop discovery systems
> can then allocate experiments toward non-existent phenomena while aggregate
> checks remain plausible.

The project deliberately does not optimize for endpoint degradation. The main
evidence chain is:

1. construct a realistic target/donor record-integrity failure;
2. show that a neural surrogate learns the specified false association;
3. show that closed-loop acquisition allocates experiments toward that target;
4. verify with true oracle labels that the target phenomenon is not real;
5. show that common aggregate audits are non-diagnostic.

## Key Documents

- `CLAIMS_AND_EXPERIMENT_SPEC.md`: frozen claim and experiment specification.
- `refine-logs/FEASIBILITY_EXPERIMENT_PLAN.md`: staged feasibility plan.
- `refine-logs/FEASIBILITY_EXPERIMENT_TRACKER.md`: initial run tracker.
- `refine-logs/CHECKABLE_GOALS_AND_EXIT_CRITERIA.md`: autonomous goal-mode
  completion gates and stop conditions.

## Environment

Primary environment on this host:

```bash
conda run --no-capture-output -n agentconda python --version
```

Dependency installs should use mirror sources. The host currently has pip and
conda mirror configuration, and this repo also provides project-local mirror
helpers:

```bash
source configs/mirrors.env.example
conda run --no-capture-output -n agentconda python scripts/check_environment.py
```

For editable development installs:

```bash
conda run --no-capture-output -n agentconda python -m pip install -e ".[dev]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

For ESM/protein-LM runs:

```bash
conda run --no-capture-output -n agentconda python -m pip install -e ".[protein]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Smoke Checks

```bash
conda run --no-capture-output -n agentconda python -m pytest -q
```
