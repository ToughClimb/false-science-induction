from pathlib import Path

import false_science


def test_package_imports() -> None:
    assert false_science.__version__


def test_core_planning_documents_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "CLAIMS_AND_EXPERIMENT_SPEC.md").is_file()
    assert (root / "refine-logs" / "FEASIBILITY_EXPERIMENT_PLAN.md").is_file()
    assert (root / "refine-logs" / "FEASIBILITY_EXPERIMENT_TRACKER.md").is_file()

