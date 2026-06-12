from pathlib import Path

import false_science


def test_package_imports() -> None:
    assert false_science.__version__


def test_core_planning_documents_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "DATA.md").is_file()
    assert (root / "README.md").is_file()
    assert (root / "MANIFEST.md").is_file()
    assert (root / "artifacts" / "results").is_dir()
