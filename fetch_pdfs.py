#!/usr/bin/env python3
"""
Download PDFs for a papers.jsonl, naming each <paper_id>.pdf so the extraction
pipeline (LocalDirSource) picks them up by id.

Run this on a host that can reach arxiv.org (NOT the firewalled GPU box), then
push the folder to the bucket so the GPU box can sync it back:

    python fetch_pdfs.py --input test/papers/papers.jsonl --out test_pdfs
    python fetch_pdfs.py --input test/survey/papers.jsonl --out test_pdfs   # add surveys too
    hf sync test_pdfs hf://buckets/twanghcmut/evidence-ai/test/pdfs

On the GPU box, sync them down and parse:
    hf sync hf://buckets/twanghcmut/evidence-ai/test/pdfs test_pdfs
    python extract_pipeline.py --run-name test_set --input test_pdfs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.config import Config
from harvester.downloader import PDFDownloader


def _load(input_path: Path) -> list[dict]:
    return [json.loads(ln) for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch PDFs for a papers.jsonl by arxiv id")
    ap.add_argument("--input", required=True, help="papers.jsonl with arxiv_id/url fields")
    ap.add_argument("--out", default="test_pdfs", help="output folder of <paper_id>.pdf")
    args = ap.parse_args()

    papers = _load(Path(args.input))
    config = Config(pdf_dir=Path(args.out))
    print(f"Fetching {len(papers)} PDFs -> {args.out}")
    PDFDownloader(config).download(papers)


if __name__ == "__main__":
    main()
