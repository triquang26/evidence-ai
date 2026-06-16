from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class ParserConfig:
    backend: str = "vlm-engine"  # MinerU -b: pipeline|vlm-engine|hybrid-engine|vlm-http-client|hybrid-http-client
    model: str = "opendatalab/MinerU2.5-Pro-2605-1.2B"
    server_url: str | None = None  # set to use vlm-http-client against an external server
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractorConfig:
    model_id: str = "Qwen/Qwen3-30B-A3B-Instruct"
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "EMPTY"
    schema: str = "evidence_eval"
    extraction_passes: int = 3
    max_workers: int = 20
    max_char_buffer: int = 8000
    fence_output: bool = True
    use_schema_constraints: bool = False
    guided_json: bool = False  # optional vLLM xgrammar validation layer


@dataclass(frozen=True)
class SourceConfig:
    kind: str = "arxiv"  # arxiv | local | hf_text
    # local: dir of <paper_id>.pdf pre-downloaded and synced via HF bucket
    # hf_text: stream full paper text from jamescalam/ai-arxiv on HuggingFace
    #          (firewalled host, ~100k ML papers 2015-2023, no PDF needed)
    local_dir: str | None = None  # dir of <paper_id>.pdf when kind == 'local'


@dataclass(frozen=True)
class ServerConfig:
    enabled: bool = True  # let the pipeline manage the vLLM server lifecycle
    port: int = 8001
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    startup_timeout_s: int = 600
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str = "test_smoke"
    seed: int = 42
    resolve_citations: bool = True  # link each subject/baseline to its own source paper (arxiv id)
    input_path: Path = REPO_ROOT / "test" / "papers" / "papers.jsonl"
    out_root: Path = REPO_ROOT / "outputs" / "extract"
    corpus_size: int = 5692  # for the full-corpus time projection
    source: SourceConfig = field(default_factory=SourceConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def run_dir(self) -> Path:
        return self.out_root / self.run_name

    @property
    def pdf_dir(self) -> Path:
        return self.run_dir / "pdfs"

    @property
    def parsed_dir(self) -> Path:
        return self.run_dir / "parsed"

    @property
    def extracted_dir(self) -> Path:
        return self.run_dir / "extracted"

    @property
    def validated_dir(self) -> Path:
        return self.run_dir / "validated"

    @property
    def audit_dir(self) -> Path:
        return self.run_dir / "audit"

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "PipelineConfig":
        path = Path(path) if path else DEFAULT_CONFIG
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "PipelineConfig":
        data = dict(data)
        for key, path_field in (("input_path", "input_path"), ("out_root", "out_root")):
            if key in data:
                data[path_field] = Path(data[key])
        if "source" in data:
            data["source"] = SourceConfig(**data["source"])
        if "parser" in data:
            data["parser"] = ParserConfig(**data["parser"])
        if "extractor" in data:
            data["extractor"] = ExtractorConfig(**data["extractor"])
        if "server" in data:
            data["server"] = ServerConfig(**data["server"])
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})

    def make_dirs(self) -> None:
        for d in (self.pdf_dir, self.parsed_dir, self.extracted_dir, self.audit_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.extractor.guided_json:
            self.validated_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> Path:
        """Persist the resolved config next to the run for reproducibility."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "run_config.yaml"
        target.write_text(yaml.safe_dump(self._serializable(), sort_keys=False), encoding="utf-8")
        return target

    def _serializable(self) -> dict:
        d = asdict(self)
        d["input_path"] = str(self.input_path)
        d["out_root"] = str(self.out_root)
        for sub in ("source", "parser", "extractor", "server"):
            d[sub] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d[sub].items()}
        return d

    def set_seed(self) -> None:
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        try:
            import numpy as np

            np.random.seed(self.seed)
        except ImportError:
            pass
