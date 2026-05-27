#!/usr/bin/env bash
set -euo pipefail

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

python -m pip install --upgrade pip setuptools wheel \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"

python -m pip install -e ".[dev]" \
  -i "${PIP_INDEX_URL}" \
  --trusted-host "${PIP_TRUSTED_HOST}"

