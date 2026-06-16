from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractionResult:
    paper_id: str
    jsonl_path: Path | None  # LangExtract annotated doc (grounded)
    n_extractions: int
    ok: bool
    error: str | None = None


class BaseExtractor(ABC):
    """Strategy interface for markdown -> grounded structured output backends."""

    @abstractmethod
    def extract(self, text: str, paper_id: str, out_dir: Path) -> ExtractionResult:
        ...
