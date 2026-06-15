#!/usr/bin/env python
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = REPO_ROOT / "paper-nature-main" / "figures"
OVERLEAF_FLAT = REPO_ROOT / "paper-nature-main" / "overleaf_flat"
SUBMISSION_FIGURES = (
    REPO_ROOT
    / "paper-nature-main"
    / "submission_20260531"
    / "manuscript_source"
    / "figures"
)


@dataclass(frozen=True)
class FigureMapping:
    source_dir: Path
    source_stem: str
    target_stem: str


FIGURE_MAPPINGS = [
    FigureMapping(PAPER_FIGURES, "false_science_main_visual", "false_science_main_visual"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b20_cross_domain_10seed_final_counts", "cross_domain_false_pursuit"),
    FigureMapping(PAPER_FIGURES, "b74_coherence_law", "coherence_to_budget_law"),
    FigureMapping(PAPER_FIGURES, "b43_b46_operating_boundaries", "operating_boundaries"),
    FigureMapping(REPO_ROOT / "figures", "b57_coherence_budget_law", "coherence_budget_relationship"),
    FigureMapping(PAPER_FIGURES, "b80_synthetic_susceptibility_phase_diagram", "synthetic_susceptibility_phase_diagram"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b23_mechanism_diagnostics", "round0_mechanism_diagnostics"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b22_gfp_dose_response", "gfp_dose_response"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b11_dose_response", "materials_dose_response"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b12_distributed_trigger_ablation", "distributed_trigger_ablation"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b12_conditionality_diagnostics", "conditionality_diagnostics"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b14_long_loop_persistence", "long_loop_persistence"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b20_long_loop_trajectories", "long_loop_trajectories"),
    FigureMapping(PAPER_FIGURES, "b70_bear_physical_sdl_replay", "bear_autonomous_replay"),
    FigureMapping(REPO_ROOT / "docs" / "figures", "b31_cameo_retrospective", "cameo_retrospective_replay"),
    FigureMapping(REPO_ROOT / "figures", "b55_natural_coherence_audit", "natural_coherence_audit"),
    FigureMapping(REPO_ROOT / "figures", "b58_sample_metadata_shift_replay", "sample_metadata_shift_replay"),
]


def copy_variant(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"figure source not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def sync_mapping(mapping: FigureMapping) -> list[Path]:
    copied: list[Path] = []
    for suffix in ["pdf", "png", "svg"]:
        source = mapping.source_dir / f"{mapping.source_stem}.{suffix}"
        if source.is_file():
            target = PAPER_FIGURES / f"{mapping.target_stem}.{suffix}"
            copy_variant(source, target)
            copied.append(target)
    pdf_source = PAPER_FIGURES / f"{mapping.target_stem}.pdf"
    if not pdf_source.is_file():
        raise FileNotFoundError(f"missing synchronized PDF: {pdf_source}")
    for target_dir in [OVERLEAF_FLAT, SUBMISSION_FIGURES]:
        target = target_dir / f"{mapping.target_stem}.pdf"
        copy_variant(pdf_source, target)
        copied.append(target)
    return copied


def main() -> int:
    copied: list[Path] = []
    for mapping in FIGURE_MAPPINGS:
        copied.extend(sync_mapping(mapping))
    for path in copied:
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
