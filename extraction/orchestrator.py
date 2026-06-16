from __future__ import annotations

import json
from pathlib import Path

from .audit import AuditReporter
from .config import PipelineConfig
from .exporters import CsvExporter, build_corpus_comparison
from .factory import build_extractor, build_parser, build_validator
from .sources import build_source, paper_id
from .timing import StageTimer, write_timing_report
from .vllm_server import VLLMServer


class ExtractionPipeline:
    """Acquire -> parse (GPU phase 1) -> extract+audit (GPU phase 2) -> manifest.

    Sequential serving: parse runs MinerU VLM in-process (frees the GPU on
    exit); extraction then starts a single Qwen vLLM server via VLLMServer.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.timer = StageTimer()

    def run(self) -> list[dict]:
        self.cfg.set_seed()
        self.cfg.make_dirs()
        self.cfg.snapshot()

        papers = self._load_papers()
        records = {paper_id(p): {"id": paper_id(p), "title": p.get("title"),
                                 "arxiv_id": p.get("arxiv_id"), "url": p.get("url"),
                                 "status": "pending"}
                   for p in papers}

        downloaded = self._acquire(papers, records)
        parsed = self._parse(downloaded, records)
        self._extract_and_audit(parsed, records)

        manifest = list(records.values())
        self._write_manifest(manifest)
        self._write_timing(manifest)
        CsvExporter(self.cfg.extractor.schema).export(manifest, self.cfg.run_dir)
        build_corpus_comparison(self.cfg.out_root)
        return manifest

    def _load_papers(self) -> list[dict]:
        path = self.cfg.input_path
        if path.is_dir():  # a folder of PDFs -> one paper per file (id = filename stem)
            return [{"title": p.stem, "arxiv_id": p.stem} for p in sorted(path.glob("*.pdf"))]
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]

    def _acquire(self, papers: list[dict], records: dict) -> list[tuple[str, Path]]:
        if self.cfg.input_path.is_dir():  # PDFs already on disk at the input dir
            source = build_source("local", str(self.cfg.input_path))
        else:
            source = build_source(self.cfg.source.kind, self.cfg.source.local_dir)
        with self.timer("download"):
            source.fetch(papers, self.cfg.pdf_dir)
        downloaded = []
        for p in papers:
            pid = paper_id(p)
            pdf = self.cfg.pdf_dir / f"{pid}.pdf"
            if pdf.exists():
                downloaded.append((pid, pdf))
            else:
                records[pid]["status"] = "no_pdf"
        return downloaded

    def _parse(self, downloaded: list[tuple[str, Path]], records: dict) -> list[tuple[str, str]]:
        parser = build_parser(self.cfg)
        parsed = []
        with self.timer("parse"):
            for pid, pdf in downloaded:
                result = parser.parse(pdf, pid, self.cfg.parsed_dir)
                rec = records[pid]
                if result.ok and result.markdown_path:
                    rec["markdown"] = str(result.markdown_path)
                    rec["content_list"] = str(result.content_list_path) if result.content_list_path else None
                    rec["status"] = "parsed"
                    parsed.append((pid, result.markdown))
                else:
                    rec["status"] = "parse_failed"
                    rec["error"] = result.error
        return parsed

    def _extract_and_audit(self, parsed: list[tuple[str, str]], records: dict) -> None:
        if not parsed:
            return
        extractor = build_extractor(self.cfg)
        validator = build_validator(self.cfg)
        auditor = AuditReporter()
        with VLLMServer(self.cfg.server, self.cfg.extractor):
            for pid, markdown in parsed:
                rec = records[pid]
                with self.timer("extract"):
                    ex = extractor.extract(markdown, pid, self.cfg.extracted_dir)
                if not ex.ok:
                    rec["status"] = "extract_failed"
                    rec["error"] = ex.error
                    continue
                rec["status"] = "extracted"
                rec["extracted"] = str(ex.jsonl_path)
                rec["n_extractions"] = ex.n_extractions
                if validator:
                    vpath = validator.validate(markdown, pid, self.cfg.validated_dir)
                    rec["validated"] = str(vpath) if vpath else None
                with self.timer("audit"):
                    html = auditor.render(ex.jsonl_path, self.cfg.audit_dir, pid)
                rec["audit"] = str(html) if html else None

    def _write_manifest(self, manifest: list[dict]) -> None:
        meta = {
            "parser_model": self.cfg.parser.model,
            "parser_backend": self.cfg.parser.backend,
            "extractor_model": self.cfg.extractor.model_id,
            "schema": self.cfg.extractor.schema,
            "seed": self.cfg.seed,
        }
        path = self.cfg.run_dir / "manifest.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for rec in manifest:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_timing(self, manifest: list[dict]) -> None:
        parsed = sum(1 for r in manifest if r["status"] in ("parsed", "extracted", "extract_failed"))
        extracted = sum(1 for r in manifest if r["status"] == "extracted")
        write_timing_report(
            self.cfg.run_dir / "timing_estimate.txt",
            run_name=self.cfg.run_name,
            seed=self.cfg.seed,
            attempted=len(manifest),
            parsed=parsed,
            extracted=extracted,
            elapsed=self.timer.elapsed,
            corpus_size=self.cfg.corpus_size,
        )
