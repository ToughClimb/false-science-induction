from __future__ import annotations

from pathlib import Path


FORBIDDEN_PATTERNS = [
    "default" + "=",
    "DEFAULT" + "_",
    "set" + "default(",
    "." + "get(",
]


def test_code_does_not_reintroduce_hidden_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_files = [
        path
        for directory in [root / "src", root / "scripts"]
        for path in directory.rglob("*.py")
    ]
    offenders: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                offenders.append(f"{path.relative_to(root)} contains {pattern}")

    assert not offenders, "\n".join(offenders)
