#!/usr/bin/env bash
set -euo pipefail

: "${PIP_INDEX_URL:?source configs/mirrors.env.example before running this script}"
: "${PIP_TRUSTED_HOST:?source configs/mirrors.env.example before running this script}"

python -m pip install --upgrade pip setuptools wheel \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"

python -m pip install -e ".[dev]" \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"
