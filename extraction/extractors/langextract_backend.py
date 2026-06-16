from __future__ import annotations

from pathlib import Path

import langextract as lx
from langextract.factory import ModelConfig

from ..config import ExtractorConfig
from ..schemas import get_schema
from .base import BaseExtractor, ExtractionResult


class LangExtractExtractor(BaseExtractor):
    """LangExtract over a local vLLM OpenAI-compatible endpoint.

    Grounding is the anti-hallucination guarantee: every extraction carries a
    char-offset back into the source markdown, auditable in the HTML viewer.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.schema = get_schema(config.schema)
        self._model_config = ModelConfig(
            model_id=config.model_id,
            provider="openai",
            provider_kwargs={"api_key": config.api_key, "base_url": config.base_url},
        )

    def extract(self, text: str, paper_id: str, out_dir: Path) -> ExtractionResult:
        if not text.strip():
            return ExtractionResult(paper_id, None, 0, ok=False, error="empty markdown")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=self.schema.prompt,
                examples=self.schema.examples,
                config=self._model_config,
                extraction_passes=self.config.extraction_passes,
                max_workers=self.config.max_workers,
                max_char_buffer=self.config.max_char_buffer,
                fence_output=self.config.fence_output,
                use_schema_constraints=self.config.use_schema_constraints,
            )
        except Exception as exc:  # noqa: BLE001 - record any backend failure per-paper
            return ExtractionResult(paper_id, None, 0, ok=False, error=str(exc)[:500])

        name = f"{paper_id}.jsonl"
        lx.io.save_annotated_documents([result], output_name=name, output_dir=str(out_dir))
        n = len(getattr(result, "extractions", []) or [])
        return ExtractionResult(paper_id, out_dir / name, n, ok=True)
