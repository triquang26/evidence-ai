#!/usr/bin/env python3
"""
Parse PDFs and extract grounded system/compute/efficiency evidence.

Pipeline (sequential on 1x H100):
  acquire PDFs -> MinerU2.5 VLM parse (vLLM, in-process) -> free GPU
  -> Qwen3 vLLM server -> LangExtract grounded JSON -> audit HTML -> manifest

Usage:
    python extract_pipeline.py                                  # uses configs/pipeline.yaml
    python extract_pipeline.py --config configs/full_run.yaml   # full corpus with local PDFs
    python extract_pipeline.py --run-name test_smoke
    python extract_pipeline.py --input outputs/papers.jsonl
    python extract_pipeline.py --pdf-dir ./pdfs                 # override local PDF dir
    python extract_pipeline.py --guided                         # add guided_json validation
    python extract_pipeline.py --no-server                      # Qwen server already running

After the run, sync to HuggingFace:
    python sync_to_hf.py --src outputs/extract/<run-name> --dst extract/<run-name>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from extraction.config import PipelineConfig
from extraction.orchestrator import ExtractionPipeline


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grounded extraction pipeline")
    ap.add_argument("--config", default=None, help="Path to pipeline.yaml (default: configs/pipeline.yaml)")
    ap.add_argument("--run-name", default=None, help="Override run name / output subdir")
    ap.add_argument("--input", default=None, help="Override input papers.jsonl")
    ap.add_argument("--pdf-dir", default=None, help="Override source.local_dir (dir of <arxiv_id>.pdf files)")
    ap.add_argument("--guided", action="store_true", help="Enable guided_json validation layer")
    ap.add_argument("--no-server", action="store_true", help="Do not manage the vLLM server (already running)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    cfg = cfg.with_overrides(
        run_name=args.run_name,
        input_path=Path(args.input) if args.input else None,
    )
    if args.pdf_dir:
        from extraction.config import SourceConfig
        cfg = replace(cfg, source=SourceConfig(kind="local", local_dir=args.pdf_dir))
    if args.guided:
        cfg = replace(cfg, extractor=replace(cfg.extractor, guided_json=True))
    if args.no_server:
        cfg = replace(cfg, server=replace(cfg.server, enabled=False))

    manifest = ExtractionPipeline(cfg).run()
    extracted = sum(1 for r in manifest if r["status"] == "extracted")
    print(f"\nDone: {extracted}/{len(manifest)} extracted -> {cfg.run_dir}")
    print(f"Timing report: {cfg.run_dir / 'timing_estimate.txt'}")


if __name__ == "__main__":
    main()
