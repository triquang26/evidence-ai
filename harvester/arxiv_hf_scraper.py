from __future__ import annotations
import re
import sys

from .config import Config

_SURVEY_RE = re.compile(r"\b(survey|a review|review on|benchmark)\b", re.I)
# librarian-bots dataset streams newest-first (2026→older), so we find recent
# papers quickly without scanning millions of old records.
_DATASET_ID = "librarian-bots/arxiv-metadata-snapshot"
_YEAR_CUTOFF = 2015


class ArxivHFScraper:
    """Stream the full arXiv metadata snapshot from HuggingFace and filter by topic."""

    def __init__(self, config: Config, max_papers: int = 5000):
        self.config = config
        self.max_papers = max_papers

    def harvest(self) -> list[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  [skip] pip install datasets", file=sys.stderr)
            return []

        print(f"  streaming {_DATASET_ID} (this may take a few minutes)...")
        ds = load_dataset(_DATASET_ID, split="train", streaming=True)

        rows: list[dict] = []
        scanned = 0

        for record in ds:
            scanned += 1
            if scanned % 100_000 == 0:
                print(f"  scanned {scanned:,}  matched {len(rows)}")
            if len(rows) >= self.max_papers:
                break

            if not self._passes(record):
                continue

            arxiv_id = record.get("id", "")
            title = " ".join((record.get("title") or "").split())
            abstract = " ".join((record.get("abstract") or "").split())
            jref = record.get("journal-ref")
            doi = record.get("doi")
            year = self._extract_year(record)

            if year and year < _YEAR_CUTOFF:
                continue

            rows.append({
                "title": title,
                "venue": jref,
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "doi": doi,
                "abstract": abstract,
                "is_survey": bool(_SURVEY_RE.search(title + " " + abstract)),
                "source": "arxiv_hf",
            })

        print(f"  scanned {scanned:,} total  →  {len(rows)} matched")
        return rows

    def _passes(self, record: dict) -> bool:
        cats = set((record.get("categories") or "").split())
        if not (cats & self.config.arxiv_cats):
            return False
        text = ((record.get("title") or "") + " " + (record.get("abstract") or "")).lower()
        return any(k in text for k in self.config.keep_keywords)

    @staticmethod
    def _extract_year(record: dict) -> int | None:
        versions = record.get("versions") or []
        if versions:
            v1 = versions[0].get("created", "")
            m = re.search(r"\b(20\d{2})\b", v1)
            if m:
                return int(m.group(1))
        update = record.get("update_date") or ""
        m = re.search(r"^(20\d{2})", update)
        return int(m.group(1)) if m else None
