# Home Codex Rules

## Operating mode

- Codex is the primary autonomous agent on this machine.
- Work autonomously unless blocked by missing credentials, unavailable hardware,
  or external service failure.
- Prefer git worktrees for multi-step changes and experiments.

## Research workflow

- Before running an experiment, define a hypothesis, a budget, acceptance
  criteria, and stop conditions.
- Every run must record config, seed, dataset or input version, command,
  metrics, and artifact paths.
- Never overwrite previous baselines or result directories.

## Code workflow

- Prefer the smallest useful change.
- Run the relevant lint, tests, or smoke checks before declaring success.
- Surface regressions, reproducibility risks, and missing tests explicitly.

## Review standard

- Prioritize correctness, regressions, experiment validity, and
  reproducibility.
- Reviewer decisions must be one of APPROVED, NEEDS_REVISION, or REJECTED.

## Safety boundaries

- Avoid destructive commands unless the task clearly requires them.
- Do not modify files outside the working repository unless the task is
  specifically about machine configuration.

