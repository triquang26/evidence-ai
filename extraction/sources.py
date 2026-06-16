from __future__ import annotations

import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from harvester.config import Config as HarvesterConfig
from harvester.downloader import PDFDownloader, _norm_title

_HF_TEXT_DATASET = "jamescalam/ai-arxiv"


def paper_id(paper: dict) -> str:
    base = paper.get("arxiv_id") or _norm_title(paper.get("title", ""))[:60]
    return base.replace("/", "_")


class BaseSource(ABC):
    """Strategy interface for getting PDFs (or pre-parsed text) into the pipeline.

    Decouples acquisition from the rest of the pipeline so a firewalled host
    swaps the source via config instead of code.

    `fetch(papers, pdf_dir)` is the primary method.  Sources that can also
    pre-supply parsed markdown (bypassing MinerU) may implement
    `pre_parse(papers, parsed_dir)` — the orchestrator calls it before MinerU
    so MinerU's idempotent cache check finds the file and skips re-parsing.
    """

    @abstractmethod
    def fetch(self, papers: list[dict], pdf_dir: Path) -> None: ...

    def pre_parse(self, papers: list[dict], parsed_dir: Path) -> None:
        """Override to write pre-parsed markdown, bypassing MinerU."""


class ArxivSource(BaseSource):
    """Direct arxiv.org download (reuses harvester.PDFDownloader). Needs arxiv reachable."""

    def fetch(self, papers: list[dict], pdf_dir: Path) -> None:
        PDFDownloader(HarvesterConfig(pdf_dir=pdf_dir)).download(papers)


class LocalDirSource(BaseSource):
    """Copy pre-supplied PDFs named <paper_id>.pdf from a local/bucket-synced dir.

    The firewall-safe path: user drops PDFs (or `hf download`s them) into one
    folder, the pipeline picks up whatever matches.
    """

    def __init__(self, src_dir: Path):
        self.src_dir = Path(src_dir)

    def fetch(self, papers: list[dict], pdf_dir: Path) -> None:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for p in papers:
            pid = paper_id(p)
            src = self.src_dir / f"{pid}.pdf"
            dst = pdf_dir / f"{pid}.pdf"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)


class HFTextSource(BaseSource):
    """Firewalled-host source: stream full paper text from jamescalam/ai-arxiv on HuggingFace.

    For each paper whose arxiv_id exists in the dataset, writes the pre-extracted
    `content` field as a markdown file into parsed_dir — MinerU's idempotent check
    then finds the file and skips re-parsing.  A sentinel empty PDF is also written
    to pdf_dir so the pipeline records the paper as "downloaded".

    Papers whose arxiv_id is not in the dataset are silently skipped (no PDF, no parse).
    Dataset coverage: ~100k ML papers, 2015–2023.
    """

    def __init__(self, dataset_id: str = _HF_TEXT_DATASET):
        self.dataset_id = dataset_id
        self._content_cache: dict[str, str] = {}  # arxiv_id → content text

    def fetch(self, papers: list[dict], pdf_dir: Path) -> None:
        # Called first by the orchestrator; cache is populated in pre_parse.
        # Write sentinel PDFs for papers whose content was found.
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for p in papers:
            pid = paper_id(p)
            arxiv_id = p.get("arxiv_id") or ""
            if arxiv_id in self._content_cache:
                sentinel = pdf_dir / f"{pid}.pdf"
                if not sentinel.exists():
                    sentinel.write_bytes(b"")

    def pre_parse(self, papers: list[dict], parsed_dir: Path) -> None:
        """Stream dataset, match by arxiv_id, write content files.  Must be called before fetch."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("  [HFTextSource] pip install datasets required", file=sys.stderr)
            return

        want: dict[str, dict] = {
            p["arxiv_id"]: p for p in papers if p.get("arxiv_id")
        }
        if not want:
            return

        print(f"  [HFTextSource] streaming {self.dataset_id} for {len(want)} papers …")
        ds = load_dataset(self.dataset_id, split="train", streaming=True)
        found = 0
        for row in ds:
            row_id = (row.get("id") or "").strip()
            if row_id not in want:
                continue
            content = (row.get("content") or "").strip()
            if not content:
                continue
            self._content_cache[row_id] = content

            p = want[row_id]
            pid = paper_id(p)
            md_dir = parsed_dir / pid
            md_dir.mkdir(parents=True, exist_ok=True)
            md_path = md_dir / f"{pid}.md"
            if not md_path.exists():
                md_path.write_text(content, encoding="utf-8")

            found += 1
            if found >= len(want):
                break  # all papers matched — stop streaming

        print(f"  [HFTextSource] found {found}/{len(want)} papers in dataset")


def build_source(kind: str, local_dir: str | None = None) -> BaseSource:
    if kind == "arxiv":
        return ArxivSource()
    if kind == "local":
        if not local_dir:
            raise ValueError("source.local_dir is required for kind='local'")
        return LocalDirSource(Path(local_dir))
    if kind == "hf_text":
        return HFTextSource()
    raise ValueError(f"Unknown source kind '{kind}' (expected: arxiv | local | hf_text)")
