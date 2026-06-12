from __future__ import annotations

import importlib.util
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

    missing: list[str] = []
    for module in REQUIRED_MODULES:
        available = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if available else 'missing'}")
        if not available:
            missing.append(module)

    if missing:
        raise RuntimeError("missing modules: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
