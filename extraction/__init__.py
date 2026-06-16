"""Grounded extraction pipeline: PDF -> markdown (MinerU/vLLM) -> JSON (LangExtract/vLLM)."""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
