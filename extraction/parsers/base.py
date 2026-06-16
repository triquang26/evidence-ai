from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing one PDF.

    markdown_path feeds the extractor; content_list_path keeps MinerU's
    per-block page_idx/bbox for tracing an extraction back to the PDF page.
    """

    paper_id: str
    markdown_path: Path | None
    content_list_path: Path | None
    ok: bool
    error: str | None = None

    @property
    def markdown(self) -> str:
        if not self.markdown_path or not self.markdown_path.exists():
            return ""
        return self.markdown_path.read_text(encoding="utf-8")


class BaseParser(ABC):
    """Strategy interface for PDF -> markdown backends."""

    @abstractmethod
    def parse(self, pdf_path: Path, paper_id: str, out_dir: Path) -> ParseResult:
        """Parse one PDF, writing artifacts under out_dir/<paper_id>/."""

    def parse_many(self, pdfs: list[tuple[str, Path]], out_dir: Path) -> list[ParseResult]:
        return [self.parse(path, paper_id, out_dir) for paper_id, path in pdfs]
