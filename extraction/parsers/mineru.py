from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import ParserConfig
from .base import BaseParser, ParseResult


class MinerUParser(BaseParser):
    """Wraps the `mineru` CLI (MinerU2.5 VLM on vLLM).

    Runs in-process VLM by default (`-b vlm`), so the GPU is released when the
    process exits — this is what makes the sequential parse->extract handoff
    work on a single H100. Set ParserConfig.server_url to instead drive an
    external MinerU server via the http-client backend.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse(self, pdf_path: Path, paper_id: str, out_dir: Path) -> ParseResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        md, content_list = self._locate_outputs(out_dir, pdf_path.stem)
        if md is not None:  # idempotent: reuse a previous parse (resumable re-runs)
            return ParseResult(paper_id, md, content_list, ok=True)
        cmd = self._build_command(pdf_path, out_dir)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            return ParseResult(paper_id, None, None, ok=False, error=(exc.stderr or str(exc))[:500])
        except FileNotFoundError:
            return ParseResult(paper_id, None, None, ok=False, error="mineru CLI not found")

        md, content_list = self._locate_outputs(out_dir, pdf_path.stem)
        if md is None:
            return ParseResult(paper_id, None, None, ok=False, error="no markdown produced")
        return ParseResult(paper_id, md, content_list, ok=True)

    def _build_command(self, pdf_path: Path, out_dir: Path) -> list[str]:
        cmd = ["mineru", "-p", str(pdf_path), "-o", str(out_dir), "-b", self.config.backend]
        if self.config.server_url:
            cmd += ["-u", self.config.server_url]
        cmd += list(self.config.extra_args)
        return cmd

    @staticmethod
    def _locate_outputs(out_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
        """MinerU writes <out>/<stem>/<auto|vlm>/<stem>.md (+ _content_list.json).

        Search is scoped strictly to this paper's own subdir so a shared parsed
        dir never lets one paper pick up another paper's markdown.
        """
        base = out_dir / stem
        if not base.exists():
            return None, None
        md = next(iter(sorted(base.rglob(f"{stem}.md")) or sorted(base.rglob("*.md"))), None)
        content_list = next(iter(sorted(base.rglob("*_content_list.json"))), None)
        return md, content_list
