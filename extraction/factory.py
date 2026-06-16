from __future__ import annotations

from .config import PipelineConfig
from .extractors import BaseExtractor, GuidedJsonValidator, LangExtractExtractor
from .parsers import BaseParser, MinerUParser

_PARSERS = {"mineru": MinerUParser}
_EXTRACTORS = {"langextract": LangExtractExtractor}


def build_parser(config: PipelineConfig, kind: str = "mineru") -> BaseParser:
    return _PARSERS[kind](config.parser)


def build_extractor(config: PipelineConfig, kind: str = "langextract") -> BaseExtractor:
    return _EXTRACTORS[kind](config.extractor)


def build_validator(config: PipelineConfig) -> GuidedJsonValidator | None:
    if not config.extractor.guided_json:
        return None
    return GuidedJsonValidator(config.extractor)
