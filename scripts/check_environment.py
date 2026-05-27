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
    print(f"PIP_INDEX_URL: {os.environ.get('PIP_INDEX_URL', '<unset>')}")
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', '<unset>')}")

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

