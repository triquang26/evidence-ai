#!/usr/bin/env python3
"""
Aggregate every run's evidence.csv into one cross-paper comparison table.

    python build_comparison.py                       # scans outputs/extract/*/evidence.csv
    python build_comparison.py --out-root outputs/extract

Writes <out-root>/comparison.csv — one row per (paper, subject, metric) across all runs, with
paper_id / arxiv_id / arxiv_url / title so the source PDF stays retrievable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extraction.exporters import build_corpus_comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cross-paper comparison.csv")
    ap.add_argument("--out-root", default="outputs/extract", help="dir holding run subfolders")
    args = ap.parse_args()
    target = build_corpus_comparison(Path(args.out_root))
    print(f"Wrote {target}" if target else "No evidence.csv files found.")


if __name__ == "__main__":
    main()
