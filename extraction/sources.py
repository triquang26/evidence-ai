from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from harvester.config import Config as HarvesterConfig
from harvester.downloader import PDFDownloader, _norm_title


def paper_id(paper: dict) -> str:
    base = paper.get("arxiv_id") or _norm_title(paper.get("title", ""))[:60]
    return base.replace("/", "_")


class BaseSource(ABC):
    """Strategy interface for getting PDFs into the run's pdf_dir.

    Decouples acquisition from the rest of the pipeline so a firewalled host
    (no arxiv.org) swaps the source via config instead of code.
    """

    @abstractmethod
    def fetch(self, papers: list[dict], pdf_dir: Path) -> None:
        ...


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


def build_source(kind: str, local_dir: str | None = None) -> BaseSource:
    if kind == "arxiv":
        return ArxivSource()
    if kind == "local":
        if not local_dir:
            raise ValueError("source.local_dir is required for kind='local'")
        return LocalDirSource(Path(local_dir))
    raise ValueError(f"Unknown source kind '{kind}' (expected: arxiv | local)")
