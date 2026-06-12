from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from false_science.target_scan import git_text


def load_script(path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(path.replace("/", "_"), root / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_git_text_raises_on_git_failure() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        git_text(["definitely-not-a-git-subcommand"])


def test_environment_check_raises_when_required_module_missing(monkeypatch) -> None:
    module = load_script("scripts/check_environment.py")
    monkeypatch.setattr(module, "REQUIRED_MODULES", ["definitely_missing_module"])
    with pytest.raises(RuntimeError, match="missing modules"):
        module.main()
