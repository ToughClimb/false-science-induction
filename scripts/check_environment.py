from __future__ import annotations

import importlib.util
import os
import sys


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "torch",
    "xgboost",
    "lightgbm",
    "pytest",
]


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    pip_index = os.environ["PIP_INDEX_URL"] if "PIP_INDEX_URL" in os.environ else "<unset>"
    http_proxy = os.environ["HTTP_PROXY"] if "HTTP_PROXY" in os.environ else "<unset>"
    print(f"PIP_INDEX_URL: {pip_index}")
    print(f"HTTP_PROXY: {http_proxy}")

    missing: list[str] = []
    for module in REQUIRED_MODULES:
        available = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if available else 'missing'}")
        if not available:
            missing.append(module)

    if missing:
        print("missing modules: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
