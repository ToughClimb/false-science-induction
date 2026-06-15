# GitHub Release Notes

Target repository:

```text
git@github.com:ToughClimb/false-science-induction.git
```

This repository is intended to contain code, configs, tests, manuscript source
and lightweight documentation. It intentionally excludes large or generated
state:

- raw datasets
- processed datasets and caches
- run directories
- model checkpoints
- review-stage external archives
- submission zip bundles
- LaTeX build intermediates

Use `DATA.md` and `scripts/prepare_public_data.py` to prepare public datasets
before running configs that need local data files.

## SSH Key

The local deploy key generated for this repository is stored under the local
user's SSH directory and should not be committed. Add this public key to GitHub:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFIuDQywxqYflmDBlJIdD6cJq/S3HU8JmtgBG/d+jw0C false-science-induction-github-20260615
```

Add it as a GitHub deploy key with write access, or add it to an account with
write permission to the target repository.
