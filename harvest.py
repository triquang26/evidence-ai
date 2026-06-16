#!/usr/bin/env python3
"""
Harvest ~2000 time-series anomaly-detection / forecasting papers.

Sources (in order):
  A. Awesome GitHub lists   — curated, venue-labelled
  B. HuggingFace Papers API — mirrors arXiv, accessible behind firewall

Usage:
    python harvest.py                   # harvest -> outputs/papers.jsonl + .csv
    python harvest.py --pdfs            # also download PDFs to outputs/pdfs/
    python harvest.py --max 1000        # limit per HF-papers query (default 1000)
    python harvest.py --out my_outputs  # custom output directory

After harvest, sync to HuggingFace bucket:
    python sync_to_hf.py --src outputs --dst papers/v1
"""

import argparse
from pathlib import Path

from harvester.config import Config
from harvester.downloader import PDFDownloader
from harvester.hf_papers_scraper import HFPapersScraper
from harvester.pipeline import Deduplicator, Exporter
from harvester.scraper import AwesomeScraper


class Harvester:
    def __init__(self, config: Config, hf_max_per_query: int = 1000):
        self.config = config
        self.hf_max_per_query = hf_max_per_query
        self._dedup = Deduplicator(config)
        self._exporter = Exporter(config)

    def run(self, download_pdfs: bool = False) -> list[dict]:
        raw: list[dict] = []

        print("[A] Awesome-lists ...")
        raw += AwesomeScraper(self.config).harvest()

        print("[B] HuggingFace Papers ...")
        raw += HFPapersScraper(self.config, max_per_query=self.hf_max_per_query).harvest()

        print(f"\nRaw total: {len(raw)}")
        papers = self._dedup.run(raw)
        self._print_stats(papers)
        self._exporter.save(papers)

        if download_pdfs:
            print("\n[PDF] downloading to", self.config.pdf_dir, "...")
            PDFDownloader(self.config).download(papers)

        return papers

    @staticmethod
    def _print_stats(papers: list[dict]) -> None:
        with_venue = sum(1 for p in papers if p.get("venue"))
        surveys = sum(1 for p in papers if p.get("is_survey"))
        print(f"After dedup+filter: {len(papers)}  | with venue: {with_venue}  | surveys: {surveys}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Harvest time-series papers")
    ap.add_argument("--pdfs", action="store_true", help="Download PDFs after harvest")
    ap.add_argument("--max", type=int, default=1000, dest="max_per_query",
                    help="Max results per HF-papers query (default: 1000)")
    ap.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    config = Config(
        out_dir=Path(args.out),
        pdf_dir=Path(args.out) / "pdfs",
    )
    Harvester(config, hf_max_per_query=args.max_per_query).run(
        download_pdfs=args.pdfs,
    )


if __name__ == "__main__":
    main()
