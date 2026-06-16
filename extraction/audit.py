from __future__ import annotations

from pathlib import Path

import langextract as lx


class AuditReporter:
    """Render LangExtract's interactive grounding viewer to a standalone HTML."""

    def render(self, jsonl_path: Path, out_dir: Path, paper_id: str) -> Path | None:
        if not jsonl_path or not jsonl_path.exists():
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            html = lx.visualize(str(jsonl_path))
        except Exception:  # noqa: BLE001 - audit is non-fatal
            return None
        content = html.data if hasattr(html, "data") else html
        target = out_dir / f"{paper_id}.html"
        target.write_text(content, encoding="utf-8")
        return target
