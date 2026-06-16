#!/usr/bin/env python3
"""
Flatten an existing run's grounded extractions into CSV columns.

    python to_csv.py outputs/extract/ml_demo            # uses schema from run_config.yaml
    python to_csv.py outputs/extract/ml_demo --schema compute_efficiency

Writes evidence.csv (one row per paper) and extractions_long.csv (one row per grounded span).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from extraction.exporters import CsvExporter


def _load_manifest(run_dir: Path) -> list[dict]:
    rows = []
    for ln in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if "_meta" not in rec:
            rows.append(rec)
    return rows


def _schema_of(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    cfg = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
    return cfg["extractor"]["schema"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a run's extractions to CSV")
    ap.add_argument("run_dir", help="outputs/extract/<run_name>")
    ap.add_argument("--schema", default=None, help="Override schema name")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest = _load_manifest(run_dir)
    wide, long = CsvExporter(_schema_of(run_dir, args.schema)).export(manifest, run_dir)
    print(f"Wrote {wide}\nWrote {long}")


if __name__ == "__main__":
    main()
