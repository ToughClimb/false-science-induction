#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pytest -q

BASE_CONFIG="configs/smoke_b80_synthetic_susceptibility_phase_diagram_20260531.json"
SMOKE_RUN_DIR="${SMOKE_RUN_DIR:-runs/smoke_b80_synthetic_susceptibility_phase_diagram_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
SMOKE_CONFIG="${SMOKE_RUN_DIR}/config.json"
mkdir -p "$SMOKE_RUN_DIR"
python - "$BASE_CONFIG" "$SMOKE_RUN_DIR" "$SMOKE_CONFIG" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

base_config = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
smoke_config = Path(sys.argv[3])

cfg = json.loads(base_config.read_text(encoding="utf-8"))
cfg["run_id"] = run_dir.name
cfg["output_dir"] = str(run_dir)
cfg["output_detail_csv"] = str(run_dir / "detail.csv")
cfg["output_summary_csv"] = str(run_dir / "summary.csv")
cfg["output_threshold_csv"] = str(run_dir / "thresholds.csv")
cfg["output_json"] = str(run_dir / "result.json")
cfg["output_md"] = str(run_dir / "result.md")
cfg["output_figure_pdf"] = str(run_dir / "b80_synthetic_susceptibility_phase_diagram.pdf")
smoke_config.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
PY

python scripts/b80_synthetic_susceptibility_phase_diagram.py \
  --config "$SMOKE_CONFIG"

echo "Smoke reproduction completed: $SMOKE_RUN_DIR"
