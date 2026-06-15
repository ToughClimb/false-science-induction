from __future__ import annotations

import importlib.util
from pathlib import Path


def test_b31_cameo_figure_script_imports() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_b31_cameo_figure.py"
    spec = importlib.util.spec_from_file_location("generate_b31_cameo_figure", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
